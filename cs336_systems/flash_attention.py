"""
Flash Attention V2 Triton Implementation
"""

import torch
import triton
import triton.language as tl
from einops import einsum, rearrange


def flash_attention_forward_pytorch(Q, K, V, is_causal: bool):
    """
    Flash Attention forward pass using tiled computation in pure PyTorch.
    This implementation avoids materializing the full attention matrix.
    
    :param Q: Query (batch size, query seq len, head dim)
    :param K: Key (batch size, key value seq len, head dim)
    :param V: Value (batch size, key value seq len, head dim)
    :param is_causal: whether use causal mask
    :type is_causal: bool

    :return O: Output (batch size, query seq len, head dim)
    :return L: the logsumexp value
    """
    batch_size, n_queries, d = Q.shape
    n_keys = K.shape[1]
    
    # Tile sizes
    Q_TILE_SIZE = 16
    K_TILE_SIZE = 16
    
    # Number of tiles
    n_query_tiles = (n_queries + Q_TILE_SIZE - 1) // Q_TILE_SIZE
    n_key_tiles = (n_keys + K_TILE_SIZE - 1) // K_TILE_SIZE
    
    # Scale factor
    scale = 1.0 / (d ** 0.5)
    
    # Initialize output and logsumexp
    O = torch.zeros_like(Q)
    L = torch.full((batch_size, n_queries), -float('inf'), device=Q.device, dtype=torch.float32)
    
    for b in range(batch_size):
        for qi in range(n_query_tiles):
            q_start = qi * Q_TILE_SIZE
            q_end = min(q_start + Q_TILE_SIZE, n_queries)
            
            # Load query tile
            q_tile = Q[b, q_start:q_end, :]  # (Bq, d)
            
            # Initialize running statistics for this query tile
            o_tile = torch.zeros((q_end - q_start, d), device=Q.device, dtype=torch.float32)
            m_tile = torch.full((q_end - q_start,), -float('inf'), device=Q.device, dtype=torch.float32)
            l_tile = torch.zeros((q_end - q_start,), device=Q.device, dtype=torch.float32)
            
            for kj in range(n_key_tiles):
                k_start = kj * K_TILE_SIZE
                k_end = min(k_start + K_TILE_SIZE, n_keys)
                
                # Load key and value tiles
                k_tile = K[b, k_start:k_end, :]  # (Bk, d)
                v_tile = V[b, k_start:k_end, :]  # (Bk, d)
                
                # Compute attention scores: S = Q @ K^T * scale
                s_tile = einsum(q_tile, k_tile, '... Bq d, ... Bk d -> ... Bq Bk') * scale  # (Bq, Bk)
                
                # Apply causal mask if needed
                if is_causal:
                    # Create causal mask for this tile
                    q_indices = torch.arange(q_start, q_end, device=Q.device)[:, None]
                    k_indices = torch.arange(k_start, k_end, device=Q.device)[None, :]
                    mask = q_indices < k_indices
                    s_tile = torch.where(mask, -1e6, s_tile)
                
                # Update running maximum
                m_tile_new = torch.maximum(m_tile, torch.max(s_tile, dim=-1)[0])
                
                # Compute alpha for updating o and l
                alpha = rearrange(torch.exp(m_tile - m_tile_new), 'b -> b 1')
                
                # Update o
                p_tilde = torch.exp(s_tile - rearrange(m_tile_new, 'b -> b 1'))
                l_tile_new = rearrange(alpha, 'b 1 -> b') * l_tile + torch.sum(p_tilde, dim=-1)
                o_tile = alpha * o_tile + torch.matmul(p_tilde, v_tile)
                
                # Update m and l
                m_tile = m_tile_new
                l_tile = l_tile_new
            
            # Normalize output
            o_tile = o_tile / l_tile[:, None]
            
            # Write output
            O[b, q_start:q_end, :] = o_tile
            
            # Compute and save logsumexp: L = m + log(l)
            L[b, q_start:q_end] = m_tile + torch.log(l_tile)
    
    return O, L


class FlashAttentionPytorch(torch.autograd.Function):
    """
    Flash Attention 2 implemented using pure PyTorch operations.
    Uses tiled computation to avoid materializing the full attention matrix.
    """
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Forward pass of Flash Attention.
        
        Args:
            ctx: Autograd context
            Q: Query tensor (batch_size, n_queries, d)
            K: Key tensor (batch_size, n_keys, d)
            V: Value tensor (batch_size, n_keys, d)
            is_causal: Whether to apply causal masking
        
        Returns:
            O: Output tensor (batch_size, n_queries, d)
        """
        O, L = flash_attention_forward_pytorch(Q, K, V, is_causal)
        
        # Save tensors for backward pass
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.d = Q.shape[-1]
        
        return O
    
    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass of Flash Attention using recomputation.
        
        Args:
            ctx: Autograd context
            dO: Gradient of output (batch_size, n_queries, d)
        
        Returns:
            dQ: Gradient of query
            dK: Gradient of key
            dV: Gradient of value
        """
        pass
