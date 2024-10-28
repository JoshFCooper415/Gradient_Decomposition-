import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict
from .cuda_kernels import CUDABlockAngularDecomposition

class ORGradientCompressor:
    """CUDA-optimized gradient compressor."""
    
    def __init__(
        self,
        model: nn.Module,
        block_size: int = 64,
        coupling_threshold: float = 0.1,
        min_compression_ratio: float = 1.2
    ):
        self.model = model
        self.min_compression_ratio = min_compression_ratio
        self.decomposer = CUDABlockAngularDecomposition(
            block_size=block_size,
            coupling_threshold=coupling_threshold
        )
        self.compression_stats = defaultdict(list)
        
    def compress_gradients(self) -> Dict:
        """Compress gradients using CUDA-optimized implementation."""
        stats = {}
        total_compression_ratio = 0
        total_preserved_energy = 0
        num_params = 0
        
        # Group parameters by size for batch processing
        param_groups = defaultdict(list)
        for name, param in self.model.named_parameters():
            if param.grad is None or param.grad.numel() < 64:
                continue
            shape = tuple(param.grad.shape)
            param_groups[shape].append((name, param))
            
        # Process each group in parallel
        for shape, params in param_groups.items():
            # Stack gradients for batch processing
            grads = torch.stack([p.grad for _, p in params])
            names = [n for n, _ in params]
            
            # Determine if parameters need structure preservation
            preserve_structure = any(
                any(key in n.lower() for key in ['embed', 'pos_encoding'])
                for n in names
            )
            
            # Compress gradients in batch
            compressed_grads, compression_stats = self.decomposer.decompose_and_compress(
                grads,
                preserve_structure=preserve_structure
            )
            
            # Only apply compression if it meets minimum ratio
            if compression_stats['avg_compression_ratio'] >= self.min_compression_ratio:
                # Update gradients and stats
                for i, (name, param) in enumerate(params):
                    param.grad.data.copy_(compressed_grads[i])
                    stats[name] = compression_stats
                    
                total_compression_ratio += compression_stats['avg_compression_ratio']
                total_preserved_energy += compression_stats['avg_preserved_energy']
                num_params += len(params)
                
        # Calculate averages
        return {
            'avg_compression_ratio': total_compression_ratio / max(num_params, 1),
            'avg_preserved_energy': total_preserved_energy / max(num_params, 1)
        }