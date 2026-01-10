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
    def backward(ctx: torch.autograd.function.FunctionCtx, dO: torch.Tensor):
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
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        d = ctx.d

        D = torch.sum(O * dO, dim=-1)
        S = einsum(Q, K, "... Q d, ... K d -> ... Q K") / (d ** 0.5)
        if is_causal:
            n_queries = Q.shape[1]
            n_keys = K.shape[1]
            mask = torch.arange(n_queries, device=Q.device)[:, None] < torch.arange(n_keys, device=Q.device)[None, :]
            S = torch.where(mask, -1e6, S)
        P = torch.exp(S - rearrange(L, "... -> ... 1"))
        dV = einsum(P, dO, "... Q K, ... Q d -> ... K d")
        dP = einsum(dO, V, "... Q d, ... K d -> ... Q K")
        dS = P * (dP - rearrange(D, "... -> ... 1"))
        dQ = dS @ K / (d ** 0.5)
        dK = einsum(dS, Q, "... Q K, ... Q d -> ... K d") / (d ** 0.5)
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    m = tl.full([Q_TILE_SIZE], -1e6, dtype=tl.float32)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    o = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES),
        strides=(stride_lq),
        offsets=(query_tile_index * Q_TILE_SIZE),
        block_shape=(Q_TILE_SIZE),
        order=(0),
    )

    l = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        s = tl.dot(q, k.T) * scale

        if is_causal:
            q_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
            k_indices = tl.arange(0, K_TILE_SIZE) + key_tile_index * K_TILE_SIZE
            mask = q_indices[:, None] < k_indices[None, :]
            s = tl.where(mask, -1e6, s)

        m_new = tl.max(m, tl.max(s, axis=-1))

        alpha = tl.exp(m - m_new)

        p = tl.exp(s - m_new[:, None])

        l_new = alpha * l + tl.sum(p, axis=-1)

        o = alpha[:, None] * o + tl.dot(p.to(V_block_ptr.type.element_ty), v)

        m = m_new
        l = l_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o = o / l[:, None]
    
    tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, m + tl.log(l), boundary_check=(0))


class FlashAttentionTriton(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        is_causal: bool = False
    ):
        O = torch.empty_like(Q)
        L = torch.empty_like((Q.shape[0], Q.shape[1]), device=Q.device, dtype=torch.float32)

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        grid = (tl.cdiv(Q.shape[1], Q_TILE_SIZE), Q.shape[0])

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            *Q.stride(), *K.stride(), *V.stride(),
            *O.stride(), *L.stride(),
            Q.shape[1], K.shape[1],
            1.0 / (Q.shape[-1] ** 0.5),
            Q.shape[-1],
            Q_TILE_SIZE, K_TILE_SIZE,
            is_causal,
        )

        # Save tensors for backward pass
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.d = Q.shape[-1]
        
        return O

    def backward(
            ctx: torch.autograd.function.FunctionCtx,
            dO: torch.Tensor,
        ):
        pass