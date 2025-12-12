import torch  # type: ignore
import torch.nn as nn  # type: ignore
import triton  # type: ignore #

from modules.helper import benchmark

from modules.kernels import anchor_window_fwd_kernel_optimized, gla_chunk_fwd_kernel


class LizardAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size=64, chunk_size=64, alpha=1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.alpha = alpha

    @benchmark(warmup_iterations=10, benchmark_iterations=100, save_results=True)
    def forward(self, q, k, v, g, alpha=1, triton_kernel=False):
        if not triton_kernel:
            return self.gla_fwd(q, k, v, g) + alpha * self.awa_fwd(q, k, v)

        raise NotImplementedError("custom kernel is not implemented yet!")

    def awa_fwd(self, q, k, v):
        # todo(mahdi): implement me in pytorch
        raise NotImplementedError("awa forward: not implemented!")

    def awa_fwd_triton_kernel(self, q, k, v):
        batch, heads, seq, d = q.shape
        out = torch.empty_like(q)

        BLOCK_Q = 64
        BLOCK_K = 64
        BLOCK_D = triton.next_power_of_2(self.d_head)

        grid = (triton.cdiv(seq, BLOCK_Q), batch, heads)

        anchor_window_fwd_kernel_optimized[grid](  # type: ignore
            q,
            k,
            v,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            seq,
            self.d_head,
            self.window_size,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            BLOCK_DMODEL=BLOCK_D,
            num_stages=2,
            num_warps=4,
        )
        return out

    def gla_fwd(self, q, k, v, g):
        # todo(mahdi): implement me in pytorch
        raise NotImplementedError("gla forward: not implemented!")

    def gla_fwd_triton_kernel(self, q, k, v, g):
        batch, heads, seq, d = q.shape
        out = torch.empty_like(q)
        num_chunks = triton.cdiv(seq, self.chunk_size)
        state_in = torch.zeros(
            batch, heads, num_chunks + 1, d, device=q.device, dtype=q.dtype
        )
        state_out = torch.zeros_like(state_in)
        BLOCK_D = triton.next_power_of_2(self.d_head)
        grid = (batch, heads, num_chunks)

        gla_chunk_fwd_kernel[grid](  # type: ignore
            q,
            k,
            v,
            g,
            out,
            state_in,
            state_out,
            seq,
            self.d_head,
            self.chunk_size,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            g.stride(0),
            g.stride(1),
            g.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            state_in.stride(0),
            state_in.stride(1),
            state_in.stride(3),
            BLOCK_D=BLOCK_D,
            BLOCK_CHUNK=self.chunk_size,
        )
        return out


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise Exception(
            """
            to run this experiement, you need a cuda enabled device!
            failed!
            """
        )

    import argparse

    parser = argparse.ArgumentParser(description="Configuration for Model Parameters")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=128, help="Sequence length (default: 128)"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Hidden size (default: 512)"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="Number of attention heads (default: 2)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type (default: bfloat16)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use (default: cpu)"
    )
    parser.add_argument("--experiment_name", type=str, default="Manual_Run")

    args = parser.parse_args()

    batch_size = int(args.batch_size)
    sequence_length = int(args.seq_len)
    hidden_size = int(args.hidden_size)
    num_heads = int(args.num_heads)
    device = args.device
    dtype = args.dtype

    print(f"Batch Size:  {batch_size}")
    print(f"Sequence Length:     {sequence_length}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Num Heads:   {num_heads}")
    print(f"Dtype:       {dtype}")
    print(f"Device:      {device}")

    x = torch.randn(
        args.batch_size,
        args.seq_len,
        args.hidden_size,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )

    ###

    D_MODEL = num_heads * hidden_size
    # Now that we have 64-bit pointers, 16k and 32k should be safe
    SEQ_LENGTH = 1024

    lizard = LizardAttention(D_MODEL, HEADS).cuda().to(DTYPE)

    # lizard = LizardLayer(D_MODEL, HEADS).cuda().to(DTYPE)

    # if FLA_AVAILABLE:
    #     fla_model = (
    #         GatedLinearAttention(
    #             hidden_size=D_MODEL, num_heads=HEADS, mode="fused_chunk"
    #         )
    #         .cuda()
    #         .to(DTYPE)
    #     )

    # print(f"{'Seq':<8} | {'FLA (Ref)':<12} | {'AWA (New)':<12} | {'Lizard Total':<12}")
    # print("-" * 60)

    # results = {"seq": [], "fla": [], "awa": [], "liz": []}

    # for s in SEQ_LENS:

    q = torch.randn(batch_size, HEADS, SEQ_LENGTH, DIM, device="cuda", dtype=DTYPE)
    k = torch.randn(batch_size, HEADS, SEQ_LENGTH, DIM, device="cuda", dtype=DTYPE)
    v = torch.randn(batch_size, HEADS, SEQ_LENGTH, DIM, device="cuda", dtype=DTYPE)
    g = torch.sigmoid(
        torch.randn(batch_size, HEADS, SEQ_LENGTH, device="cuda", dtype=torch.float32)
    ).to(DTYPE)
    x_fla = torch.randn(batch_size, SEQ_LENGTH, D_MODEL, device="cuda", dtype=DTYPE)

    try:
        y = lizard.forward(q, k, v, g)
    except torch.cuda.OutOfMemoryError:
        print(f"{s:<8} | OOM")
    except Exception as e:
        print(f"{s:<8} | Error: {e}")

    try:
        y = forward(
            x,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            device=args.device,
            experiment_name=args.experiment_name,
        )
        print("Success! Output shape:", len(y))
    except Exception as e:
        print("Failed:", e)
