import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
import wandb
from pathlib import Path
import logging
from tqdm import tqdm
import math
from typing import Dict, Tuple, Optional
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.gradient_compressor import ORGradientCompressor
from training.optimizer import configure_optimizer

class Trainer:
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(config['output_dir'])
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.setup()
        
    def setup(self):
        """Initialize model, tokenizer, and other components."""
        self.setup_tokenizer()
        self.setup_model()
        self.setup_optimization()
        self.setup_compression()
        
    def setup_tokenizer(self):
        """Initialize and configure tokenizer."""
        self.logger.info("Setting up tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            cache_dir=self.config['cache_dir']
        )
        
        # Set up padding token
        if self.tokenizer.pad_token is None:
            self.logger.info("Setting up padding token...")
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
        self.logger.info(f"Vocabulary size: {len(self.tokenizer)}")
        self.logger.info(f"Padding token: {self.tokenizer.pad_token}")
                
    def setup_model(self):
        """Initialize and configure model."""
        self.logger.info("Setting up model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            cache_dir=self.config['cache_dir']
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(self.device)
    def setup_compression(self):
        """Setup gradient compression."""
        self.logger.info("Setting up gradient compression...")
        
        # Extract compression parameters from nested dictionary
        compression_config = self.config['compression']
        compression_kwargs = {
            'block_size': compression_config['block_size'],
            'coupling_threshold': compression_config['coupling_threshold'],
            'min_compression_ratio': compression_config['min_compression_ratio']
        }
        
        self.logger.info(f"Compression config: {compression_kwargs}")
        
        self.compressor = ORGradientCompressor(
            self.model,
            **compression_kwargs
        ) 
    def setup_optimization(self):
        """Setup optimizer and scheduler."""
        self.logger.info("Setting up optimization...")
        self.optimizer = configure_optimizer(
            self.model,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        total_steps = self.config['max_steps']
        warmup_steps = int(total_steps * self.config['warmup_ratio'])
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        
    def save_checkpoint(self, path: Path, val_loss: Optional[float] = None):
        """Save model checkpoint."""
        checkpoint = {
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
            
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """Execute a single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss / self.config['gradient_accumulation_steps']
        loss.backward()
        
        return loss.item(), math.exp(loss.item() * self.config['gradient_accumulation_steps'])
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        self.model.train()
        self.optimizer.zero_grad()
        
        try:
            while self.global_step < self.config['max_steps']:
                epoch_loss = 0
                num_batches = 0
                
                for batch in tqdm(train_loader, desc=f"Step {self.global_step}"):
                    loss, perplexity = self.train_step(batch)
                    
                    if (num_batches + 1) % self.config['gradient_accumulation_steps'] == 0:
                        # Compress gradients
                        compression_stats = self.compressor.compress_gradients()
                        
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        # Update weights
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                        self.global_step += 1
                        
                        # Log training progress
                        if self.global_step % self.config['logging_steps'] == 0:
                            self.log_training_step(loss, perplexity, compression_stats)
                        
                        # Evaluation
                        if self.global_step % self.config['eval_steps'] == 0:
                            self.run_evaluation(val_loader)
                        
                        # Regular model saving
                        if self.global_step % self.config['save_steps'] == 0:
                            self.save_checkpoint(
                                self.output_dir / f'model_step_{self.global_step}.pt'
                            )
                    
                    num_batches += 1
                    epoch_loss += loss
                    
                    if self.global_step >= self.config['max_steps']:
                        break
                        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user. Saving checkpoint...")
            self.save_checkpoint(self.output_dir / 'interrupted_checkpoint.pt')
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            raise
            
        finally:
            # Save final model
            self.save_checkpoint(self.output_dir / 'final_model.pt')
            
    def log_training_step(self, loss: float, perplexity: float, compression_stats: Dict):
        """Log training metrics."""
        lr = self.scheduler.get_last_lr()[0]
        
        wandb.log({
            'train/loss': loss * self.config['gradient_accumulation_steps'],
            'train/perplexity': perplexity,
            'train/learning_rate': lr,
            'train/compression_ratio': compression_stats['avg_compression_ratio'],
            'train/preserved_energy': compression_stats['avg_preserved_energy'],
            'global_step': self.global_step
        })
        
        self.logger.info(
            f"Step {self.global_step}: "
            f"Loss: {loss * self.config['gradient_accumulation_steps']:.4f}, "
            f"Perplexity: {perplexity:.4f}, "
            f"LR: {lr:.2e}"
        )
        
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                total_loss += outputs.loss.item()
                num_batches += 1
        
        self.model.train()
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        return avg_loss, perplexity
        
    def run_evaluation(self, val_loader: DataLoader):
        """Run evaluation and log results."""
        val_loss, val_perplexity = self.evaluate(val_loader)
        
        wandb.log({
            'val/loss': val_loss,
            'val/perplexity': val_perplexity,
            'global_step': self.global_step
        })
        
        self.logger.info(
            f"Validation: Loss: {val_loss:.4f}, "
            f"Perplexity: {val_perplexity:.4f}"
        )
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(
                self.output_dir / f'best_model_step_{self.global_step}.pt',
                val_loss
            )
            self.logger.info(f"Saved new best model with val_loss: {val_loss:.4f}")

