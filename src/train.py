from dataclasses import asdict
import torch
from pathlib import Path
import wandb
from datetime import datetime
import typer
from typing import Optional

from data.dataset import TinyStoriesDataset
from training.trainer import Trainer
from utils.logging_config import setup_logging
from config import TrainingConfig
from torch.utils.data import DataLoader

app = typer.Typer()

@app.command()
def train(
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_steps: Optional[int] = None,
    learning_rate: Optional[float] = None,
    train_subset: Optional[int] = None,
    resume_from: Optional[str] = None,
):
    """
    Train SmolLM model with gradient compression.
    
    Args:
        config_path: Path to config file (optional)
        output_dir: Output directory (overrides config)
        batch_size: Batch size (overrides config)
        max_steps: Maximum training steps (overrides config)
        learning_rate: Learning rate (overrides config)
        train_subset: Number of training examples (overrides config)
        resume_from: Path to checkpoint to resume from
    """
    # Load or create config
    if config_path:
        config = TrainingConfig.load(config_path)
    else:
        config = TrainingConfig()
    
    # Override config with command line arguments
    if output_dir:
        config.output_dir = output_dir
    if batch_size:
        config.batch_size = batch_size
    if max_steps:
        config.max_steps = max_steps
    if learning_rate:
        config.learning_rate = learning_rate
    if train_subset:
        config.train_subset_size = train_subset
    
    # Create directories
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(
        name='smollm_training',
        log_dir=config.log_dir
    )
    
    # Save config
    config.save(output_dir / 'config.json')
    
    # Initialize wandb
    run_name = f"smollm_tinystories_{datetime.now():%Y%m%d_%H%M%S}"
    if resume_from:
        run_name += "_resumed"
    
    wandb.init(
        project=config.wandb_project,
        config=asdict(config),
        name=run_name
    )
    
    config_dict = asdict(config)
    # Restructure compression config
    config_dict['compression_kwargs'] = {
        'block_size': config.compression.block_size,
        'coupling_threshold': config.compression.coupling_threshold,
        'min_compression_ratio': config.compression.min_compression_ratio
    }
    
    # Initialize trainer with restructured config
    trainer = Trainer(config_dict, logger)
    
    # Load checkpoint if resuming
    if resume_from:
        trainer.load_checkpoint(resume_from)
    
    # Setup datasets
    logger.info("Preparing datasets...")
    train_dataset = TinyStoriesDataset(
        split='train',
        tokenizer=trainer.tokenizer,
        max_length=config.max_seq_length,
        subset_size=config.train_subset_size
    )
    
    val_dataset = TinyStoriesDataset(
        split='validation',
        tokenizer=trainer.tokenizer,
        max_length=config.max_seq_length,
        subset_size=config.val_subset_size
    )
    
    # Setup dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Train
    logger.info("Starting training...")
    try:
        trainer.train(train_loader, val_loader)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        wandb.finish()
        logger.info("Training finished")

if __name__ == "__main__":
    app()