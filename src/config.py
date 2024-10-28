from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json
from pathlib import Path
from enum import Enum
from typing import Union

class CompressionStrategy(str, Enum):
    """Enumeration of supported compression strategies."""
    SVD = "svd"
    BLOCK_DIAGONAL = "block_diagonal"
    ADAPTIVE = "adaptive"
    
    def __str__(self):
        return self.value

@dataclass
class CompressionConfig:
    """Configuration for gradient compression."""
    block_size: int = 64
    coupling_threshold: float = 0.1
    min_compression_ratio: float = 1.2
    strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE

@dataclass
class TrainingConfig:
    """Complete training configuration."""
    # Model settings
    model_name: str = "HuggingFaceTB/SmolLM-135M"
    max_seq_length: int = 256
    
    # Training hyperparameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_steps: int = 10_000
    
    # Dataset settings
    train_subset_size: Optional[int] = 10000
    val_subset_size: Optional[int] = 1000
    
    # Compression settings
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    
    # Training loop settings
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 10
    
    # Output settings
    output_dir: str = './outputs'
    log_dir: str = './logs'
    cache_dir: str = './cache'
    wandb_project: str = 'smollm-training'
    
    def save(self, path: Union[str, Path]):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrainingConfig':
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
            # Convert compression strategy back to enum
            if 'compression' in data:
                data['compression']['strategy'] = CompressionStrategy(
                    data['compression']['strategy']
                )
            return cls(**data)