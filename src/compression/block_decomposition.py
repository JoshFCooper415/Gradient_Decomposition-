import torch
import torch.nn as nn
from typing import Dict, Tuple

class BlockAngularDecomposition:
    """Implements block angular decomposition for gradient compression."""
    
    def __init__(self, block_size: int = 64, coupling_threshold: float = 0.1):
        """
        Initialize block angular decomposition.
        
        Args:
            block_size: Size of blocks for matrix decomposition
            coupling_threshold: Threshold for coupling detection
        """
        self.block_size = block_size
        self.coupling_threshold = coupling_threshold
        
    def decompose_and_compress(
        self,
        gradient: torch.Tensor,
        preserve_structure: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Decompose and compress gradient tensor.
        
        Args:
            gradient: Input gradient tensor
            preserve_structure: Whether to preserve matrix structure
            
        Returns:
            Tuple of (compressed gradient, compression statistics)
        """
        original_size = gradient.numel()
        compressed_size = 0
        preserved_energy = 0.0
        
        # Handle scalar or empty tensors
        if gradient.numel() <= 1:
            return gradient.clone(), {
                'orig_size': original_size,
                'compressed_size': original_size,
                'avg_compression_ratio': 1.0,
                'avg_preserved_energy': 1.0
            }
        
        # Reshape gradient into 2D matrix if needed
        orig_shape = gradient.shape
        if len(orig_shape) > 2:
            gradient = gradient.reshape(gradient.shape[0], -1)
        elif len(orig_shape) == 1:
            gradient = gradient.reshape(1, -1)
            
        # Split into blocks
        blocks = []
        block_positions = []
        rows = gradient.shape[0]
        cols = gradient.shape[1]
        
        for i in range(0, rows, self.block_size):
            for j in range(0, cols, self.block_size):
                i_end = min(i + self.block_size, rows)
                j_end = min(j + self.block_size, cols)
                block = gradient[i:i_end, j:j_end]
                blocks.append(block)
                block_positions.append((i, i_end, j, j_end))
                
        # Process each block
        compressed_blocks = []
        total_blocks = len(blocks)
        successful_compressions = 0
        
        for block in blocks:
            # Skip tiny blocks or blocks with zero elements
            if min(block.shape) <= 1 or block.numel() == 0 or not torch.any(block):
                compressed_blocks.append(block)
                compressed_size += block.numel()
                preserved_energy += 1.0
                successful_compressions += 1
                continue
                
            try:
                # Compute SVD
                U, S, V = torch.svd(block, some=True)
                
                # Handle empty singular values
                if len(S) == 0:
                    compressed_blocks.append(block)
                    compressed_size += block.numel()
                    preserved_energy += 1.0
                    successful_compressions += 1
                    continue
                
                # Compute energy preservation
                total_energy = torch.sum(S ** 2)
                if total_energy == 0:
                    compressed_blocks.append(block)
                    compressed_size += block.numel()
                    preserved_energy += 1.0
                    successful_compressions += 1
                    continue
                    
                cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy
                
                # Find minimum rank that preserves 90% energy
                energy_threshold = 0.9
                ranks = torch.nonzero(cumulative_energy >= energy_threshold)
                if len(ranks) == 0:  # If no rank meets the threshold
                    rank = len(S)
                else:
                    rank = ranks[0].item() + 1
                    
                rank = max(1, min(rank, min(block.shape)))
                
                # Compress block
                compressed_block = torch.mm(
                    torch.mm(U[:, :rank], torch.diag(S[:rank])),
                    V[:, :rank].t()
                )
                
                compressed_blocks.append(compressed_block)
                compressed_size += rank * sum(block.shape)
                preserved_energy += float(cumulative_energy[rank - 1])
                successful_compressions += 1
                
            except RuntimeError as e:
                # On SVD failure, keep original block
                compressed_blocks.append(block)
                compressed_size += block.numel()
                preserved_energy += 1.0
                successful_compressions += 1
                
        # Reconstruct gradient
        compressed_gradient = torch.zeros_like(gradient)
        for idx, (i_start, i_end, j_start, j_end) in enumerate(block_positions):
            compressed_gradient[i_start:i_end, j_start:j_end] = compressed_blocks[idx]
            
        # Reshape back to original shape if needed
        if len(orig_shape) > 2 or len(orig_shape) == 1:
            compressed_gradient = compressed_gradient.reshape(orig_shape)
            
        # Compute final statistics
        avg_compression_ratio = original_size / max(compressed_size, 1)
        avg_preserved_energy = preserved_energy / max(successful_compressions, 1)
        
        compression_stats = {
            'orig_size': original_size,
            'compressed_size': compressed_size,
            'avg_compression_ratio': avg_compression_ratio,
            'avg_preserved_energy': avg_preserved_energy
        }
        
        return compressed_gradient, compression_stats