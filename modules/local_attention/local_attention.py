from modules.helper import benchmark


@benchmark(warmup_iterations=10, benchmark_iterations=100, save_results=True)
def forward(
    q, v, k, mask, window_size=256, causal=True, device="cuda:0", experiment_name=None
):
    from local_attention import LocalAttention

    attn = (
        LocalAttention(
            dim=head_dim,  # dimension of each head
            window_size=window_size,  # window size
            causal=True,  # auto-regressive
            look_backward=1,  # each window looks at the window before
            look_forward=0,  # for causal attention
            dropout=0.0,  # no dropout for benchmarking
            exact_windowsize=False,
        )
        .to(device=device)
        .to(torch.bfloat16)
    )
    return attn(q, v, k, mask)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Configuration for Model Parameters")
    parser.add_argument(
        "--head_dim", type=int, default=16, help="Head dim (default: 16)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=1024, help="Sequence length (default: 128)"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Hidden size (default: 512)"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 2)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type (default: bfloat16)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use (default: cpu)"
    )
    parser.add_argument("--experiment_name", type=str, default="Manual_Run")

    args = parser.parse_args()

    head_dim = int(args.head_dim)
    batch_size = int(args.batch_size)
    seq_len = int(args.seq_len)
    hidden_size = int(args.hidden_size)
    num_heads = int(args.num_heads)
    device = args.device
    dtype = args.dtype

    print(f"Head Dim:  {head_dim}")
    print(f"Batch Size:  {batch_size}")
    print(f"Seq Len:     {seq_len}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Num Heads:   {num_heads}")
    print(f"Dtype:       {dtype}")
    print(f"Device:      {device}")

    import torch

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    _dtype = dtype_map[dtype]

    # (batch, heads, seq_len, head_dim)
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=_dtype
    )
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=_dtype
    )
    v = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=_dtype
    )

    mask = torch.ones(batch_size, seq_len, device=device).bool()

    try:
        y = forward(
            q,
            k,
            v,
            mask=mask,
            experiment_name=args.experiment_name,
        )
        print("Success! Output shape:", len(y))
    except Exception as e:
        print("Failed:", e)
