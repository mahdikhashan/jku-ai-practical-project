def forward(x, hidden_size, num_heads, mode="chunk"):
    from fla.layers import GatedLinearAttention

    gla = GatedLinearAttention(hidden_size, num_heads).to(
        device=device, dtype=dtype
    )
    return gla(x)


if __name__ == "__main__":
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
        "--device", type=str, default="cpu", help="Device to use (default: cpu)"
    )

    args = parser.parse_args()

    batch_size = int(args.batch_size)
    seq_len = int(args.seq_len)
    hidden_size = int(args.hidden_size)
    num_heads = int(args.num_heads)
    device = args.device
    dtype = args.dtype

    # ---------------------------------------------------------
    # Verification Print
    # ---------------------------------------------------------
    print(f"Batch Size:  {batch_size}")
    print(f"Seq Len:     {seq_len}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Num Heads:   {num_heads}")
    print(f"Dtype:       {dtype}")
    print(f"Device:      {device}")

    import torch

    x = torch.randn(
        batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16
    )

    try:
        y = forward(x, hidden_size, num_heads)
        print("Success! Output shape:", len(y))
    except Exception as e:
        print("Failed:", e)
