import torch
import triton
import triton.language as tl
import sys

# ============================================================================
# RELOAD KERNEL (Paste the Kernel Here or Import it)
# ============================================================================
# We assume the kernel 'gla_chunk_fwd_kernel' and wrapper 'gla_tensor_core' 
# form the previous script are available. 
# TO RUN THIS: Paste the kernel code from the previous step above this block
# OR run this in the same session/file.
# For simplicity, I will re-define the wrapper wrapper assuming kernel exists.

# ----------------------------------------------------------------------------
# COPY PASTE THE ENTIRE KERNEL DEFINITION HERE IF RUNNING SEPARATELY
# ----------------------------------------------------------------------------
# (If you append this to your previous file, you don't need to repost the kernel)

def run_correctness_test():
    if not torch.cuda.is_available(): return

    print("\n" + "="*80)
    print("VERIFYING NUMERICAL CORRECTNESS")
    print("="*80)

    # Config (Keep small for debugging)
    B, H, S, D = 2, 4, 1024, 128
    CHUNK = 64
    dtype = torch.float16
    device = "cuda"

    print(f"Checking Inputs: [{B}, {H}, {S}, {D}] | Chunk={CHUNK}")

    # 1. Setup Data
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    # Random gates between 0 and 1
    gates = torch.sigmoid(torch.randn(B, H, S, device=device, dtype=dtype))

    # 2. Run Naive PyTorch (The "Ground Truth")
    # This matches the math of the chunk-local kernel
    def naive_chunked_gla(q, k, v, gates):
        # Reshape to verify chunk-by-chunk exactly like the kernel
        # [B, H, NumChunks, ChunkSize, D]
        q_c = q.reshape(B, H, S // CHUNK, CHUNK, D)
        k_c = k.reshape(B, H, S // CHUNK, CHUNK, D)
        v_c = v.reshape(B, H, S // CHUNK, CHUNK, D)
        g_c = gates.reshape(B, H, S // CHUNK, CHUNK)
        
        out_c = torch.zeros_like(q_c)
        
        # Process every chunk independently
        for i in range(S // CHUNK):
            qc = q_c[:, :, i] # [B, H, Chunk, D]
            kc = k_c[:, :, i]
            vc = v_c[:, :, i]
            gc = g_c[:, :, i]
            
            # Attn = Softmax(Q @ K.T)
            scores = torch.matmul(qc, kc.transpose(-1, -2))
            
            # Causal Mask
            mask = torch.tril(torch.ones(CHUNK, CHUNK, device=device, dtype=torch.bool))
            scores.masked_fill_(~mask, float("-inf"))
            
            attn = torch.softmax(scores, dim=-1)
            
            # Apply Gates
            # Gates are [B, H, Chunk] -> unsqueeze to [B, H, Chunk, 1] for broadcast
            # Note: The kernel does `attn * gates[:, None]` which broadcasts 
            # gate across the KEY dimension.
            gated_attn = attn * gc.unsqueeze(-1)
            
            # Output
            out_c[:, :, i] = torch.matmul(gated_attn, vc)
            
        return out_c.reshape(B, H, S, D)

    ref_out = naive_chunked_gla(q, k, v, gates)

    # 3. Run Triton Kernel
    # We use the stateless wrapper to compare exactly with the chunk-local naive logic
    tri_out = gla_tensor_core(q, k, v, gates, chunk_size=CHUNK)

    # 4. Compare
    # Note: FP16 Tensor Cores have lower precision. 
    # We expect errors around 1e-2 to 1e-3.
    diff = (ref_out - tri_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print("-" * 50)
    print(f"Max Difference:  {max_diff:.6f}")
    print(f"Mean Difference: {mean_diff:.6f}")
    print("-" * 50)

    if max_diff < 0.05: # Generous tolerance for FP16 accumulators
        print("✅ PASS: Triton output matches PyTorch baseline!")
    else:
        print("❌ FAIL: Differences are too high. Debug needed.")
        print("Check: Masking logic or Softmax scaling.")

if __name__ == "__main__":
    # Ensure this function is called after your kernel definitions
    run_correctness_test()
    