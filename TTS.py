import os
import torch
import torchaudio
import soundfile as sf
import pandas as pd
import zipfile
import tempfile
import shutil
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoProcessor, VitsModel
import logging
import matplotlib.pyplot as plt
import librosa
import librosa.display
from datetime import datetime
import json
import numpy as np
import seaborn as sns


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


def split_batch(waveforms, transcriptions, speaker_genders):
    """
    Split batch into support and query sets while maintaining gender balance
    """
    # Separate male and female samples
    male_indices = [i for i, gender in enumerate(speaker_genders) if gender == "male"]
    female_indices = [i for i, gender in enumerate(speaker_genders) if gender == "female"]
    
    # Ensure balanced split for each gender
    male_split = len(male_indices) // 2
    female_split = len(female_indices) // 2
    
    # Create support and query sets with gender balance
    support_indices = male_indices[:male_split] + female_indices[:female_split]
    query_indices = male_indices[male_split:] + female_indices[female_split:]
    
    # Create support and query sets
    support_waveforms = torch.stack([waveforms[i] for i in support_indices])
    support_transcriptions = [transcriptions[i] for i in support_indices]
    support_genders = [speaker_genders[i] for i in support_indices]
    
    query_waveforms = torch.stack([waveforms[i] for i in query_indices])
    query_transcriptions = [transcriptions[i] for i in query_indices]
    query_genders = [speaker_genders[i] for i in query_indices]
    
    return (support_waveforms, support_transcriptions, support_genders), \
           (query_waveforms, query_transcriptions, query_genders)
           
           
class CatalanTTSModel:
    def __init__(self, pretrained_model_name="facebook/mms-tts-spa", device='cuda'):
        """
        Initialize with a pre-trained Spanish TTS model from Facebook's MMS
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load pre-trained model and processor
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        self.model = VitsModel.from_pretrained(pretrained_model_name).to(self.device)
        
        # Print model configuration for debugging
        print("Model config:", self.model.config)
        
        # Add training history tracking
        self.training_history = {
            'epoch_losses': [],
            'batch_losses': [],
            'timestamps': [],
            'model_config': self.model.config.to_dict()
        }
        
        # Create output directory with timestamp
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f"tts_output_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize with better learning rates and optimizer
        self.meta_learning_rate = 1e-4  # Reduced from previous value
        self.meta_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-6,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.meta_optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )

    def prepare_catalan_dataset(self, combined_dataset):
        """
        Prepare the Catalan dataset for fine-tuning
        """
        self.train_loader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            collate_fn=pad_collate_fn  # Use the custom collate function
        )
        return self.train_loader

    def compute_loss(self, batch):
        """
        Compute loss with memory-efficient processing
        """
        waveforms, transcriptions, speaker_genders = batch
        
        try:
            # Process text inputs in smaller chunks if needed
            batch_size = len(transcriptions)
            chunk_size = 2  # Process 2 samples at a time
            total_loss = 0
            
            for i in range(0, batch_size, chunk_size):
                chunk_end = min(i + chunk_size, batch_size)
                chunk_transcriptions = transcriptions[i:chunk_end]
                chunk_waveforms = waveforms[i:chunk_end]
                
                # Clear cache periodically
                if i % (chunk_size * 2) == 0:
                    torch.cuda.empty_cache()
                
                # Process text inputs
                inputs = self.processor(
                    text=chunk_transcriptions,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs)
                predicted_waveforms = outputs.waveform
                
                # Ensure target waveforms are on the correct device
                target_waveforms = chunk_waveforms.to(self.device)
                
                # Get minimum length between predicted and target
                min_length = min(predicted_waveforms.size(-1), target_waveforms.size(-1))
                
                # Truncate both to the minimum length
                predicted_waveforms = predicted_waveforms[..., :min_length]
                target_waveforms = target_waveforms[..., :min_length]
                
                # Normalize waveforms
                predicted_waveforms = predicted_waveforms / (torch.max(torch.abs(predicted_waveforms)) + 1e-7)
                target_waveforms = target_waveforms / (torch.max(torch.abs(target_waveforms)) + 1e-7)
                
                # Log shapes for debugging
                logger.debug(f"Predicted shape after truncation: {predicted_waveforms.shape}")
                logger.debug(f"Target shape after truncation: {target_waveforms.shape}")
                
                # Compute L1 loss with matched shapes
                loss = F.l1_loss(
                    predicted_waveforms.view(predicted_waveforms.shape[0], -1),
                    target_waveforms.view(target_waveforms.shape[0], -1)
                )
                
                # Add regularization term
                l2_lambda = 1e-6
                l2_reg = torch.tensor(0., device=self.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
                
                # Clip loss value
                loss = torch.clamp(loss, 0.0, 10.0)
                
                # Free memory explicitly
                del outputs
                torch.cuda.empty_cache()
                
                total_loss += loss
            
            return total_loss / ((batch_size + chunk_size - 1) // chunk_size)
            
        except Exception as e:
            logger.error(f"Error in compute_loss: {str(e)}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def prepare_batch(self, batch):
        """
        Prepare batch with consistent lengths
        """
        waveforms, transcriptions, speaker_genders = batch
        
        try:
            # Process text inputs to get expected output length
            inputs = self.processor(
                text=transcriptions,
                return_tensors="pt",
                padding=True
            )
            
            # Get model's expected output length
            with torch.no_grad():
                test_outputs = self.model(**inputs.to(self.device))
                expected_length = test_outputs.waveform.size(-1)
            
            # Pad or truncate waveforms to match expected length
            processed_waveforms = []
            for waveform in waveforms:
                if waveform.size(-1) < expected_length:
                    # Pad if too short
                    padding = torch.zeros(expected_length - waveform.size(-1))
                    waveform = torch.cat([waveform, padding])
                else:
                    # Truncate if too long
                    waveform = waveform[:expected_length]
                processed_waveforms.append(waveform)
            
            processed_waveforms = torch.stack(processed_waveforms)
            
            return processed_waveforms, transcriptions, speaker_genders
            
        except Exception as e:
            logger.error(f"Error in prepare_batch: {str(e)}")
            return batch

    def meta_train_step(self, support_set, query_set):
        """
        Enhanced meta-learning step with gender-aware training
        """
        max_grad_norm = 1.0
        
        try:
            # Prepare batches
            support_set = self.prepare_batch(support_set)
            query_set = self.prepare_batch(query_set)
            
            # Separate losses by gender for monitoring
            support_loss_male = 0
            support_loss_female = 0
            query_loss_male = 0
            query_loss_female = 0
            
            # Inner loop (support set)
            support_loss = self.compute_loss(support_set)
            
            # Track gender-specific losses
            waveforms, _, genders = support_set
            male_mask = torch.tensor([g == "male" for g in genders]).to(self.device)
            female_mask = torch.tensor([g == "female" for g in genders]).to(self.device)
            
            if male_mask.any():
                support_loss_male = support_loss * male_mask.float().mean()
            if female_mask.any():
                support_loss_female = support_loss * female_mask.float().mean()
            
            # Clip gradients for support set
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            # Compute and apply gradients
            grads = torch.autograd.grad(
                support_loss, 
                [p for p in self.model.parameters() if p.requires_grad],
                create_graph=True,
                allow_unused=True
            )
            
            # Update adapted parameters
            for param, grad in zip([p for p in self.model.parameters() if p.requires_grad], grads):
                if grad is not None:
                    param.data = param.data - self.meta_learning_rate * torch.clamp(grad, -1.0, 1.0)
            
            # Outer loop (query set)
            query_loss = self.compute_loss(query_set)
            
            # Track gender-specific losses for query set
            waveforms, _, genders = query_set
            male_mask = torch.tensor([g == "male" for g in genders]).to(self.device)
            female_mask = torch.tensor([g == "female" for g in genders]).to(self.device)
            
            if male_mask.any():
                query_loss_male = query_loss * male_mask.float().mean()
            if female_mask.any():
                query_loss_female = query_loss * female_mask.float().mean()
            
            # Optimize
            self.meta_optimizer.zero_grad()
            query_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.meta_optimizer.step()
            
            # Log gender-specific losses
            logger.debug(f"Support losses - Male: {support_loss_male:.4f}, Female: {support_loss_female:.4f}")
            logger.debug(f"Query losses - Male: {query_loss_male:.4f}, Female: {query_loss_female:.4f}")
            
            if torch.isnan(query_loss) or torch.isinf(query_loss):
                logger.warning("NaN or Inf loss detected! Using previous valid loss.")
                return self.previous_valid_loss if hasattr(self, 'previous_valid_loss') else 0.0
            
            self.previous_valid_loss = query_loss.item()
            return query_loss.item()
            
        except Exception as e:
            logger.error(f"Error in meta_train_step: {str(e)}")
            return self.previous_valid_loss if hasattr(self, 'previous_valid_loss') else 0.0

    def finetune(self, num_epochs=100):
        """
        Fine-tune with memory optimization
        """
        logger.info("Starting fine-tuning process...")
        
        # Clear memory before starting
        torch.cuda.empty_cache()
        
        # Reduce memory usage for training history
        self.training_history = {
            'epoch_losses': [],
            'batch_losses': [],
            'timestamps': []
        }
        
        consecutive_bad_epochs = 0
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Clear memory at start of each epoch
            torch.cuda.empty_cache()
            
            epoch_loss = 0
            batch_losses = []
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Split batch into support and query sets
                support_set, query_set = split_batch(*batch)
                
                # Perform meta-learning step
                loss = self.meta_train_step(support_set, query_set)
                epoch_loss += loss
                batch_losses.append(loss)
                
                # Log batch progress
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")
            
            # Calculate and store epoch metrics
            avg_loss = epoch_loss / len(self.train_loader)
            self.training_history['epoch_losses'].append(avg_loss)
            self.training_history['batch_losses'].extend(batch_losses)
            self.training_history['timestamps'].append(datetime.now().isoformat())
            
            logger.info(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
            
            # Check if this is the best model so far
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                logger.info(f"New best model achieved with loss: {best_loss:.4f}")
            
            # Check for bad training
            if avg_loss > 0.9:  # Loss is suspiciously high
                consecutive_bad_epochs += 1
            else:
                consecutive_bad_epochs = 0
            
            # Restart training if necessary
            if consecutive_bad_epochs >= 3:
                logger.warning("Loss stuck at high value. Restarting training...")
                self.reset_model()
                consecutive_bad_epochs = 0
                continue
            
            # Update learning rate
            self.scheduler.step(avg_loss)
            
            # Generate example audio every epoch
            logger.info(f"Generating example audio for epoch {epoch}")
            self.generate_example_audio(epoch, batch_idx)
            
            # Save checkpoint and plots
            if epoch % 5 == 0:  # Save every 5 epochs instead of every epoch
                if not self.save_checkpoint(epoch, avg_loss, is_best):
                    logger.warning(f"Failed to save checkpoint for epoch {epoch}")
            
            if epoch % 5 == 0:  # Generate examples every 5 epochs
                self.generate_example_audio(epoch, batch_idx)
            
            if epoch % 5 == 0:  # Plot every 5 epochs
                self.plot_training_history()
            
            # Save training history to JSON with error handling
            try:
                history_path = os.path.join(self.output_dir, 'training_history.json')
                with open(history_path, 'w') as f:
                    json.dump(self.training_history, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to save training history: {str(e)}")

    def generate_speech(self, text, output_path="output.mp3"):
        """
        Generate speech from text input
        """
        try:
            # Prepare input
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)

            # Generate speech using the forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                speech = outputs.waveform.squeeze().cpu().numpy()

            # Save the audio file
            sf.write(output_path, speech, samplerate=16000)
            logger.info(f"Generated speech saved to {output_path}")

            return speech

        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            raise

    def save_checkpoint(self, epoch, loss, is_best=False):
        """
        Save model checkpoint with metadata and error handling
        """
        try:
            # First verify write permissions and disk space
            checkpoint_dir = os.path.dirname(self.output_dir)
            if not os.access(checkpoint_dir, os.W_OK):
                logger.error(f"No write permission for directory: {checkpoint_dir}")
                return False
            
            # Check available disk space (require at least 1GB)
            free_space = shutil.disk_usage(checkpoint_dir).free
            if free_space < 1_000_000_000:  # 1GB in bytes
                logger.error(f"Insufficient disk space. Available: {free_space / 1_000_000_000:.2f}GB")
                return False

            # Create checkpoint data
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.meta_optimizer.state_dict(),
                'training_history': self.training_history,
                'model_config': self.model.config.to_dict(),
                'loss': loss,
                'timestamp': datetime.now().isoformat()
            }
            
            checkpoint_path = os.path.join(
                self.output_dir,
                f'checkpoint_epoch_{epoch}.pth'
            )
            
            # Save to temporary file first
            temp_path = checkpoint_path + '.tmp'
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, checkpoint_path)
            
            # If this is the best model, save it separately
            if is_best:
                best_model_path = os.path.join(self.output_dir, 'best_model.pth')
                best_temp_path = best_model_path + '.tmp'
                torch.save(checkpoint, best_temp_path)
                os.replace(best_temp_path, best_model_path)
                logger.info(f"Saved new best model with loss: {loss:.4f}")
            
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            # Clean up temporary files if they exist
            for temp_file in [temp_path, best_temp_path]:
                if 'temp_file' in locals() and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            return False

    def load_checkpoint(self, path):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def plot_spectrogram(self, waveform, title="Spectrogram", save_path=None):
        """
        Generate publication-quality spectrogram visualization
        """
        plt.figure(figsize=(12, 6))
        
        # Convert to numpy array if it's a torch tensor
        if torch.is_tensor(waveform):
            waveform = waveform.squeeze().cpu().numpy()
        else:
            waveform = np.asarray(waveform).squeeze()
        
        # Generate mel spectrogram with improved parameters
        spectrogram = librosa.feature.melspectrogram(
            y=waveform,
            sr=16000,
            n_mels=128,  # Increased for better resolution
            fmax=8000,
            n_fft=2048,
            hop_length=512
        )
        
        # Convert to log scale
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
        # Plot spectrogram with enhanced visualization
        img = librosa.display.specshow(
            spectrogram_db,
            y_axis='mel',
            x_axis='time',
            sr=16000,
            fmax=8000,
            cmap='magma'  # Using a perceptually uniform colormap
        )
        
        # Add colorbar with proper formatting
        cbar = plt.colorbar(img, format='%+2.0f dB')
        cbar.set_label('Intensity (dB)', rotation=270, labelpad=15)
        
        # Enhance plot aesthetics
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Frequency (Hz)', fontsize=12)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3, linestyle='--')
        
        if save_path:
            # Save in multiple formats for different use cases
            base_path = os.path.splitext(save_path)[0]
            for format in ['png', 'pdf']:
                full_path = f"{base_path}.{format}"
                plt.savefig(
                    full_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.1
                )
            plt.close()
        else:
            plt.show()

    def plot_training_history(self):
        """
        Create detailed, publication-quality training visualization plots
        """
        try:
            # Skip plotting if there's not enough data
            if len(self.training_history['epoch_losses']) < 2:
                logger.warning("Not enough data to plot training history")
                return
            
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(15, 10))
            gs = fig.add_gridspec(2, 2)
            
            # 1. Epoch Loss Plot (smoothed and raw)
            ax1 = fig.add_subplot(gs[0, 0])
            epoch_losses = np.array(self.training_history['epoch_losses'])
            epochs = np.arange(len(epoch_losses))  # Create matching x-axis array
            
            # Plot raw data
            ax1.plot(epochs, epoch_losses, 'lightgray', label='Raw Loss', alpha=0.3)
            
            # Plot smoothed data if enough points
            window_size = min(5, len(epoch_losses) // 2)
            if window_size > 1:
                smoothed_losses = np.convolve(epoch_losses, np.ones(window_size)/window_size, mode='valid')
                valid_epochs = epochs[window_size-1:len(smoothed_losses)+window_size-1]
                ax1.plot(valid_epochs, smoothed_losses, 'b-', label='Smoothed Loss', linewidth=2)
            
            ax1.set_title('Training Loss per Epoch', fontsize=12, pad=10)
            ax1.set_xlabel('Epoch', fontsize=10)
            ax1.set_ylabel('Loss', fontsize=10)
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # 2. Batch Loss Plot with Moving Average
            ax2 = fig.add_subplot(gs[0, 1])
            batch_losses = np.array(self.training_history['batch_losses'])
            batches = np.arange(len(batch_losses))
            
            # Plot raw batch data
            ax2.plot(batches, batch_losses, 'lightgray', label='Raw Batch Loss', alpha=0.3)
            
            # Calculate and plot moving average if enough points
            window_size = min(50, len(batch_losses) // 4)
            if window_size > 1:
                moving_avg = np.convolve(batch_losses, np.ones(window_size)/window_size, mode='valid')
                valid_batches = batches[window_size-1:len(moving_avg)+window_size-1]
                ax2.plot(valid_batches, moving_avg, 'r-', label='Moving Average', linewidth=2)
            
            ax2.set_title('Training Loss per Batch', fontsize=12, pad=10)
            ax2.set_xlabel('Batch', fontsize=10)
            ax2.set_ylabel('Loss', fontsize=10)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 3. Loss Distribution Plot
            ax3 = fig.add_subplot(gs[1, 0])
            sns.histplot(data=epoch_losses, bins=min(30, len(epoch_losses)), kde=True, ax=ax3)
            ax3.set_title('Loss Distribution', fontsize=12, pad=10)
            ax3.set_xlabel('Loss Value', fontsize=10)
            ax3.set_ylabel('Frequency', fontsize=10)
            
            # 4. Learning Rate Plot
            ax4 = fig.add_subplot(gs[1, 1])
            if hasattr(self, 'learning_rates') and len(self.learning_rates) > 0:
                lr_history = self.learning_rates
                ax4.plot(lr_history, 'g-', linewidth=2)
                ax4.set_title('Learning Rate Schedule', fontsize=12, pad=10)
                ax4.set_xlabel('Epoch', fontsize=10)
                ax4.set_ylabel('Learning Rate', fontsize=10)
                ax4.set_yscale('log')
                ax4.grid(True, linestyle='--', alpha=0.7)
            
            # Add timestamp and model info
            plt.figtext(0.02, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                        fontsize=8, style='italic')
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Save plots
            try:
                for format in ['png', 'pdf']:
                    save_path = os.path.join(self.output_dir, f'training_analysis.{format}')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved training plot to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save training plots: {str(e)}")
            finally:
                plt.close()

        except Exception as e:
            logger.error(f"Error in plot_training_history: {str(e)}")
            # Ensure we close any open figures to prevent memory leaks
            try:
                plt.close()
            except:
                pass

    def generate_example_audio(self, epoch, batch_idx, texts=None):
        """
        Generate and save example audio samples during training
        """
        if texts is None:
            texts = [
                "Hola, com estàs?",  # Hello, how are you?
                "Barcelona és una ciutat bonica.",  # Barcelona is a beautiful city
                "M'agrada molt la música catalana."  # I really like Catalan music
            ]
        
        example_dir = os.path.join(self.output_dir, f"examples_epoch_{epoch}")  # Simplified directory name
        os.makedirs(example_dir, exist_ok=True)
        
        try:
            logger.info(f"Generating example audio for epoch {epoch}")
            for idx, text in enumerate(texts):
                output_path = os.path.join(example_dir, f"sample_{idx}.mp3")
                waveform = self.generate_speech(text, output_path)
                
                if waveform is not None:
                    # Generate and save spectrogram
                    spec_path = os.path.join(example_dir, f"sample_{idx}_spec.png")
                    self.plot_spectrogram(
                        waveform,
                        title=f"Spectrogram - Epoch {epoch} - {text}",  # Added text to title
                        save_path=spec_path
                    )
                    logger.info(f"Generated audio and spectrogram for text: {text}")
                else:
                    logger.warning(f"No waveform generated for text: {text}")
        
        except Exception as e:
            logger.error(f"Error in generate_example_audio: {str(e)}")
            # Continue training even if example generation fails
            pass

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        """
        Load model from checkpoint with explicit CPU mapping
        """
        try:
            logger.info("Loading checkpoint...")
            
            # Explicitly map to CPU
            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu',  # Force CPU mapping
                weights_only=True    # Address the FutureWarning
            )
            
            # Initialize model on CPU
            model = cls(device='cpu')
            
            # Load state dictionaries
            try:
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        # If the checkpoint is just the model state
                        model.model.load_state_dict(checkpoint)
                    
                    # Load optimizer state if available
                    if 'optimizer_state_dict' in checkpoint:
                        model.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    # Load training history if available
                    if 'training_history' in checkpoint:
                        model.training_history = checkpoint['training_history']
                else:
                    # If checkpoint is just the model state
                    model.model.load_state_dict(checkpoint)
            
            except Exception as e:
                logger.error(f"Error loading state dictionaries: {str(e)}")
                raise
            
            # Ensure model is in eval mode
            model.model.eval()
            
            logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise

    def reset_model(self):
        """
        Reset the model and optimizer when training gets stuck
        """
        try:
            logger.info("Resetting model and optimizer states...")
            
            # Re-initialize model weights
            self.model = VitsModel.from_pretrained("facebook/mms-tts-spa").to(self.device)
            
            # Reset optimizer with initial learning rate
            self.meta_optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=1e-6,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            # Reset scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.meta_optimizer,
                mode='min',
                factor=0.5,
                patience=2,
                verbose=True
            )
            
            # Clear previous loss history
            self.previous_valid_loss = None
            
            logger.info("Model and optimizer successfully reset")
            
        except Exception as e:
            logger.error(f"Error resetting model: {str(e)}")
            raise

    def generate_comprehensive_report(self):
        """
        Generate a comprehensive training report with all relevant metrics and visualizations
        """
        report_dir = os.path.join(self.output_dir, 'analysis_report')
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. Training Statistics
        stats = {
            'total_epochs': len(self.training_history['epoch_losses']),
            'best_loss': min(self.training_history['epoch_losses']),
            'final_loss': self.training_history['epoch_losses'][-1],
            'improvement': (self.training_history['epoch_losses'][0] - 
                           self.training_history['epoch_losses'][-1]) / 
                           self.training_history['epoch_losses'][0] * 100,
            'training_duration': (
                datetime.fromisoformat(self.training_history['timestamps'][-1]) -
                datetime.fromisoformat(self.training_history['timestamps'][0])
            ).total_seconds() / 3600  # Convert to hours
        }
        
        # Save statistics
        with open(os.path.join(report_dir, 'training_stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)
        
        # 2. Generate Loss Analysis Plots
        self.plot_training_history()  # This will save the plots
        
        # 3. Generate Learning Curve Analysis
        plt.figure(figsize=(12, 6))
        epochs = range(len(self.training_history['epoch_losses']))
        
        # Plot learning curve with confidence intervals
        losses = np.array(self.training_history['epoch_losses'])
        window = min(50, len(losses) // 10)
        rolling_mean = pd.Series(losses).rolling(window=window).mean()
        rolling_std = pd.Series(losses).rolling(window=window).std()
        
        plt.plot(epochs, rolling_mean, 'b-', label='Moving Average')
        plt.fill_between(epochs, 
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        alpha=0.2, color='b')
        
        plt.title('Learning Curve Analysis', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save learning curve analysis
        plt.savefig(os.path.join(report_dir, 'learning_curve_analysis.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        return report_dir
        
        
        
def create_combined_dataset(data_dir):
    """
    Create a combined dataset from male and female directories and their corresponding TSV files.
    """
    class CatalanDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir):
            self.audio_files = []
            self.transcriptions = []
            self.speaker_genders = []

            # Process female data
            female_dir = os.path.join(root_dir, 'female')
            print(f"Processing female directory: {female_dir}")

            female_wavs = []
            for root, _, files in os.walk(female_dir):
                for file in files:
                    if file.endswith('.mp3'):
                        wav_path = os.path.join(root, file)
                        female_wavs.append(wav_path)
                        self.audio_files.append(wav_path)
                        self.transcriptions.append("Placeholder text")  # We'll update this later
                        self.speaker_genders.append("female")

            print(f"Found {len(female_wavs)} female audio files")

            # Process male data similarly
            male_dir = os.path.join(root_dir, 'male')
            print(f"Processing male directory: {male_dir}")

            male_wavs = []
            for root, _, files in os.walk(male_dir):
                for file in files:
                    if file.endswith('.mp3'):
                        wav_path = os.path.join(root, file)
                        male_wavs.append(wav_path)
                        self.audio_files.append(wav_path)
                        self.transcriptions.append("Placeholder text")  # We'll update this later
                        self.speaker_genders.append("male")

            print(f"Found {len(male_wavs)} male audio files")
            print(f"Total dataset size: {len(self.audio_files)} files")

            if len(self.audio_files) == 0:
                raise ValueError("No audio files found in the dataset")

        def __len__(self):
            return len(self.audio_files)

        def __getitem__(self, idx):
            try:
                waveform, sample_rate = torchaudio.load(self.audio_files[idx])
                return waveform, self.transcriptions[idx], self.speaker_genders[idx]
            except Exception as e:
                print(f"Error loading file {self.audio_files[idx]}: {str(e)}")
                raise

    # Create and return the dataset
    return CatalanDataset(data_dir)



def pad_collate_fn(batch):
    """
    Custom collate function to pad audio waveforms to the same length.
    """
    waveforms, transcriptions, speaker_genders = zip(*batch)
    
    # Find the maximum length in the batch
    max_length = max(waveform.size(1) for waveform in waveforms)
    
    # Pad each waveform to the maximum length
    padded_waveforms = [F.pad(waveform, (0, max_length - waveform.size(1))) for waveform in waveforms]
    
    # Stack the padded waveforms into a single tensor
    padded_waveforms = torch.stack(padded_waveforms)
    
    return padded_waveforms, list(transcriptions), list(speaker_genders)
    
    
    
if __name__ == "__main__":
    # Initialize the model
    model = CatalanTTSModel()
    
    # Create dataset
    data_dir ="/export/fhome/amlai04/data2/" # Replace with your actual data directory
    combined_dataset = create_combined_dataset(data_dir)
    
    # Prepare dataset
    model.prepare_catalan_dataset(combined_dataset)
    
    # Start training
    model.finetune()
    # During or after training
    model.plot_training_history()  # Creates detailed training plots

    
    # Generate full report
    report_dir = model.generate_comprehensive_report()
    
    