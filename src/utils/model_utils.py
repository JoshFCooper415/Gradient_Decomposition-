import torch
from pathlib import Path
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Dict:
    """
    Load model checkpoint with error handling.
    
    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        device: Device to load model to
    
    Returns:
        Dictionary containing checkpoint info
    """
    try:
        checkpoint = torch.load(path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state if provided
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        logger.info(f"Successfully loaded checkpoint from {path}")
        return checkpoint
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {path}: {str(e)}")
        raise
