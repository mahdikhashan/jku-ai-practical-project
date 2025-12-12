import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LizardAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size=64, chunk_size=64, alpha=1, m=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.alpha = alpha
        self.m = m  # number of meta-tokens
        
        # Meta-tokens for AWA (learnable)
        self.meta_tokens = nn.Parameter(torch.randn(1, n_heads, m, self.d_head))
        
        # Gating parameters for GLA
        self.W_gamma = nn.Parameter(torch.randn(1, n_heads, 1, self.d_head))
        
        # Projections for phi_q and phi_k (feature maps)
        self.phi_q = nn.Identity()  # Can be replaced with actual feature map
        self.phi_k = nn.Identity()  # Can be replaced with actual feature map

    def forward(self, q, k, v, x=None, g=None, alpha=None, triton_kernel=False, causal=False):
        """
        Args:
            q, k, v: (batch, heads, seq_len, d_head)
            x: (batch, heads, seq_len, d_head) - input for gating, if None uses q
            causal: if True, apply causal masking
        """
        if alpha is None:
            alpha = self.alpha
        
        if x is None:
            x = q  # Use queries as input for gating if not provided
            
        if not triton_kernel:
            gla_out = self.gla_fwd(q, k, v, x, causal=causal)
            awa_out = self.awa_fwd(q, k, v, causal=causal)
            return gla_out + alpha * awa_out

        raise NotImplementedError("custom kernel is not implemented yet!")

    def awa_fwd(self, q, k, v, causal=True):
        """
        Anchor Window Attention with meta-tokens.
        Formula: ŷ_i = Σ[t=i-w+1 to i] exp(q_i^T k_t / √d) v_t / (Σ[j=0 to m-1] t_j + Σ[t=i-w+1 to i] exp(q_i^T k_t / √d))
        
        Args:
            q, k, v: (batch, heads, seq_len, d_head)
            causal: if True, apply causal masking
            
        Returns:
            out: (batch, heads, seq_len, d_head)
        """
        batch, heads, seq_len, d_head = q.shape
        
        # Expand meta-tokens for batch: (batch, heads, m, d_head)
        meta_tokens = self.meta_tokens.expand(batch, -1, -1, -1)
        
        # Output tensor
        out = torch.zeros_like(q)
        
        for i in range(seq_len):
            # Define window boundaries: [i-w+1, i] for causal
            start = max(0, i - self.window_size + 1)
            end = i + 1 if causal else min(seq_len, i + self.window_size)
            
            # Extract current query and window
            q_i = q[:, :, i:i+1, :]  # (batch, heads, 1, d_head)
            k_window = k[:, :, start:end, :]  # (batch, heads, window_len, d_head)
            v_window = v[:, :, start:end, :]  # (batch, heads, window_len, d_head)
            
            # Compute attention scores with keys in window: exp(q_i^T k_t / √d)
            scores_k = torch.matmul(q_i, k_window.transpose(-2, -1)) / math.sqrt(d_head)
            exp_scores_k = torch.exp(scores_k)  # (batch, heads, 1, window_len)
            
            # Compute attention scores with meta-tokens: exp(q_i^T t_j / √d)
            scores_t = torch.matmul(q_i, meta_tokens.transpose(-2, -1)) / math.sqrt(d_head)
            exp_scores_t = torch.exp(scores_t)  # (batch, heads, 1, m)
            
            # Denominator: Σ[j=0 to m-1] exp(q_i^T t_j / √d) + Σ[t=i-w+1 to i] exp(q_i^T k_t / √d)
            sum_meta = exp_scores_t.sum(dim=-1, keepdim=True)  # (batch, heads, 1, 1)
            sum_keys = exp_scores_k.sum(dim=-1, keepdim=True)  # (batch, heads, 1, 1)
            denominator = sum_meta + sum_keys  # (batch, heads, 1, 1)
            
            # Numerator: Σ[t=i-w+1 to i] exp(q_i^T k_t / √d) v_t
            numerator = torch.matmul(exp_scores_k, v_window)  # (batch, heads, 1, d_head)
            
            # Final output: ŷ_i = numerator / denominator
            out[:, :, i:i+1, :] = numerator / (denominator + 1e-8)  # Add epsilon for numerical stability
        
        return out

    def gla_fwd(self, q, k, v, x, causal=True):
        """
        Gated Linear Attention.
        Formula: ŷ_i = (φ_q(q_i)^T Σ[t=1 to i] (Π[l=t+1 to i] Γ_l) φ_k(k_t) v_t^T) / 
                       (φ_q(q_i)^T Σ[j=1 to i] (Π[l=j+1 to i] Γ_l) φ_k(k_j))
        where Γ_i = sigmoid(W_γ x_i) is the learnable gating factor.
        
        Args:
            q, k, v: (batch, heads, seq_len, d_head)
            x: (batch, heads, seq_len, d_head) - input for gating
            causal: if True, apply causal masking (autoregressive)
            
        Returns:
            out: (batch, heads, seq_len, d_head)
        """
        batch, heads, seq_len, d_head = q.shape
        
        # Apply feature maps
        q_feat = self.phi_q(q)  # (batch, heads, seq_len, d_head)
        k_feat = self.phi_k(k)  # (batch, heads, seq_len, d_head)
        
        # Compute gating factors: Γ_i = sigmoid(W_γ x_i)
        # W_gamma: (1, heads, 1, d_head), x: (batch, heads, seq_len, d_head)
        gamma = torch.sigmoid((self.W_gamma * x).sum(dim=-1, keepdim=True))  # (batch, heads, seq_len, 1)
        
        out = torch.zeros_like(q)
        
        if causal:
            # Autoregressive: each position only attends to previous positions
            for i in range(seq_len):
                # Numerator: φ_q(q_i)^T Σ[t=1 to i] (Π[l=t+1 to i] Γ_l) φ_k(k_t) v_t^T
                numerator = torch.zeros(batch, heads, 1, d_head, device=q.device)
                
                # Denominator: φ_q(q_i)^T Σ[j=1 to i] (Π[l=j+1 to i] Γ_l) φ_k(k_j)
                denominator = torch.zeros(batch, heads, 1, 1, device=q.device)
                
                q_i = q_feat[:, :, i:i+1, :]  # (batch, heads, 1, d_head)
                
                for t in range(i + 1):  # t from 0 to i (inclusive)
                    # Compute cumulative product: Π[l=t+1 to i] Γ_l
                    if t < i:
                        cum_gamma = torch.prod(gamma[:, :, t+1:i+1, :], dim=2, keepdim=True)  # (batch, heads, 1, 1)
                    else:
                        cum_gamma = torch.ones(batch, heads, 1, 1, device=q.device)
                    
                    k_t = k_feat[:, :, t:t+1, :]  # (batch, heads, 1, d_head)
                    v_t = v[:, :, t:t+1, :]  # (batch, heads, 1, d_head)
                    
                    # Weighted k-v contribution
                    # φ_k(k_t) v_t^T: outer product -> (batch, heads, d_head, d_head)
                    kv_t = k_t.transpose(-2, -1) @ v_t  # (batch, heads, d_head, d_head)
                    
                    # Add to numerator: (Π Γ_l) * φ_k(k_t) v_t^T
                    numerator += cum_gamma.squeeze(-1).unsqueeze(-1) * (q_i @ kv_t).squeeze(2).unsqueeze(2)
                    
                    # Add to denominator: (Π Γ_l) * φ_q(q_i)^T φ_k(k_j)
                    qk_t = (q_i @ k_t.transpose(-2, -1))  # (batch, heads, 1, 1)
                    denominator += cum_gamma * qk_t
                
                # Final output for position i
                out[:, :, i:i+1, :] = numerator / (denominator + 1e-8)
        else:
            # Non-causal: each position attends to all positions
            for i in range(seq_len):
                numerator = torch.zeros(batch, heads, 1, d_head, device=q.device)
                denominator = torch.zeros(batch, heads, 1, 1, device=q.device)
                
                q_i = q_feat[:, :, i:i+1, :]
                
                for t in range(seq_len):
                    # Compute gating product based on relative positions
                    if t < i:
                        # Product from t+1 to i
                        cum_gamma = torch.prod(gamma[:, :, t+1:i+1, :], dim=2, keepdim=True)
                    elif t > i:
                        # Product from i+1 to t
                        cum_gamma = torch.prod(gamma[:, :, i+1:t+1, :], dim=2, keepdim=True)
                    else:
                        cum_gamma = torch.ones(batch, heads, 1, 1, device=q.device)
                    
                    k_t = k_feat[:, :, t:t+1, :]
                    v_t = v[:, :, t:t+1, :]
                    
                    kv_t = k_t.transpose(-2, -1) @ v_t
                    numerator += cum_gamma.squeeze(-1).unsqueeze(-1) * (q_i @ kv_t).squeeze(2).unsqueeze(2)
                    
                    qk_t = (q_i @ k_t.transpose(-2, -1))
                    denominator += cum_gamma * qk_t
                
                out[:, :, i:i+1, :] = numerator / (denominator + 1e-8)
        
        return out


# Example usage
if __name__ == "__main__":
    batch_size = 2
    n_heads = 8
    seq_len = 128
    d_model = 512
    
    # Create model
    model = LizardAttention(
        d_model=d_model,
        n_heads=n_heads,
        window_size=64,
        chunk_size=64,
        alpha=1.0,
        m=4
    )
    
    # Create dummy inputs
    d_head = d_model // n_heads
    q = torch.randn(batch_size, n_heads, seq_len, d_head)
    k = torch.randn(batch_size, n_heads, seq_len, d_head)
    v = torch.randn(batch_size, n_heads, seq_len, d_head)
    
    # Test different configurations
    print("Testing causal GLA + AWA:")
    out1 = model(q, k, v, causal=True)
    print(f"Output shape: {out1.shape}")
    
    print("\nTesting non-causal GLA + AWA:")
    out2 = model(q, k, v, causal=False)
    print(f"Output shape: {out2.shape}")
