import torch
import torch.nn as nn
import math
from typing import Tuple, Dict, Optional

class CUDABlockAngularDecomposition:
    """CUDA-optimized block angular decomposition."""
    
    def __init__(self, block_size: int = 64, coupling_threshold: float = 0.1):
        self.block_size = block_size
        self.coupling_threshold = coupling_threshold
        
    def _safe_svd(
        self,
        matrix: torch.Tensor,
        full_matrices: bool = True
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Safely perform SVD with error handling."""
        try:
            U, S, V = torch.svd(matrix, some=not full_matrices)
            return U, S, V
        except RuntimeError:
            return None, None, None
            
    def _compress_matrix(
        self,
        matrix: torch.Tensor,
        preserve_structure: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compress a single matrix using SVD."""
        device = matrix.device
        rows, cols = matrix.shape
        
        if preserve_structure:
            # Handle structure preservation mode
            compressed = matrix.clone()
            preserved_energy = 1.0
            compressed_size = matrix.numel()
        else:
            # Attempt SVD compression
            U, S, V = self._safe_svd(matrix, full_matrices=False)
            
            if U is None or S is None or V is None:
                # SVD failed, return original matrix
                return matrix.clone(), {
                    'compressed_size': matrix.numel(),
                    'avg_preserved_energy': 1.0
                }
                
            # Calculate energy preservation
            total_energy = torch.sum(S ** 2)
            if total_energy == 0:
                return matrix.clone(), {
                    'compressed_size': matrix.numel(),
                    'avg_preserved_energy': 1.0
                }
                
            cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy
            
            # Find rank that preserves 90% energy
            ranks = torch.where(cumulative_energy >= 0.9)[0]
            if len(ranks) == 0:
                rank = len(S)
            else:
                rank = ranks[0].item() + 1
            
            rank = max(1, min(rank, min(matrix.shape) - 1))
            
            # Perform compression
            compressed = torch.mm(
                torch.mm(U[:, :rank], torch.diag(S[:rank])),
                V[:, :rank].t()
            )
            
            preserved_energy = float(cumulative_energy[rank - 1].item())
            compressed_size = rank * (rows + cols)
            
        return compressed, {
            'compressed_size': compressed_size,
            'avg_preserved_energy': preserved_energy
        }
        
    def decompose_and_compress(
        self,
        gradient: torch.Tensor,
        preserve_structure: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Decompose and compress gradient tensor using CUDA acceleration.
        
        Args:
            gradient: Input gradient tensor
            preserve_structure: Whether to preserve matrix structure
            
        Returns:
            Tuple of (compressed gradient, compression statistics)
        """
        original_size = gradient.numel()
        device = gradient.device
        
        # Handle small tensors
        if original_size <= self.block_size * self.block_size:
            return gradient.clone(), {
                'orig_size': original_size,
                'compressed_size': original_size,
                'avg_compression_ratio': 1.0,
                'avg_preserved_energy': 1.0
            }
            
        # Handle different tensor shapes
        orig_shape = gradient.shape
        if len(orig_shape) == 1:
            # Convert 1D to 2D
            gradient = gradient.reshape(1, -1)
        elif len(orig_shape) > 2:
            # For 3+D tensors, reshape preserving first dimension
            gradient = gradient.reshape(orig_shape[0], -1)
            
        # Compress the gradient
        compressed_gradient, stats = self._compress_matrix(
            gradient,
            preserve_structure
        )
        
        # Reshape back to original shape
        if len(orig_shape) != 2:
            compressed_gradient = compressed_gradient.reshape(orig_shape)
            
        # Calculate final statistics
        compression_stats = {
            'orig_size': original_size,
            'compressed_size': stats['compressed_size'],
            'avg_compression_ratio': original_size / max(stats['compressed_size'], 1),
            'avg_preserved_energy': stats['avg_preserved_energy']
        }
        
        return compressed_gradient, compression_stats

# src/compression/gradient_compressor.py
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict
from .cuda_kernels import CUDABlockAngularDecomposition

class ORGradientCompressor:
    """Gradient compressor using OR decomposition techniques."""
    
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
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
                
            if param.grad.numel() < 64:  # Skip small gradients
                continue
                
            # Determine if parameter needs structure preservation
            preserve_structure = any(key in name.lower() 
                                  for key in ['embed', 'pos_encoding'])
            
            try:
                # Apply compression
                compressed_grad, compression_stats = self.decomposer.decompose_and_compress(
                    param.grad,
                    preserve_structure=preserve_structure
                )
                
                # Only apply compression if it meets minimum ratio
                if compression_stats['avg_compression_ratio'] >= self.min_compression_ratio:
                    param.grad.data.copy_(compressed_grad)
                    stats[name] = compression_stats
                    
                    total_compression_ratio += compression_stats['avg_compression_ratio']
                    total_preserved_energy += compression_stats['avg_preserved_energy']
                    num_params += 1
                    
            except Exception as e:
                print(f"Error compressing gradient for {name}: {str(e)}")
                continue
                
        # Calculate averages
        if num_params > 0:
            return {
                'avg_compression_ratio': total_compression_ratio / num_params,
                'avg_preserved_energy': total_preserved_energy / num_params
            }
        else:
            return {
                'avg_compression_ratio': 1.0,
                'avg_preserved_energy': 1.0
            }