import triton  # type: ignore
import triton.language as tl # type: ignore #


@triton.jit
def gla_chunk_fwd_kernel(
    Q,
    K,
    V,
    G,
    Out,
    State_in,
    State_out,
    seq_len,
    d_model,
    chunk_size,
    stride_qb: tl.int64,
    stride_qh: tl.int64,
    stride_qs: tl.int64,
    stride_qd: tl.int64,
    stride_kb: tl.int64,
    stride_kh: tl.int64,
    stride_ks: tl.int64,
    stride_kd: tl.int64,
    stride_vb: tl.int64,
    stride_vh: tl.int64,
    stride_vs: tl.int64,
    stride_vd: tl.int64,
    stride_gb: tl.int64,
    stride_gh: tl.int64,
    stride_gs: tl.int64,
    stride_ob: tl.int64,
    stride_oh: tl.int64,
    stride_os: tl.int64,
    stride_od: tl.int64,
    stride_sb: tl.int64,
    stride_sh: tl.int64,
    stride_sd: tl.int64,
    BLOCK_D: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    pid_batch = tl.program_id(0).to(tl.int64)
    pid_head = tl.program_id(1).to(tl.int64)
    pid_chunk = tl.program_id(2).to(tl.int64)

    chunk_start = pid_chunk * BLOCK_CHUNK
    chunk_end = tl.minimum(chunk_start + BLOCK_CHUNK, seq_len)
    if (chunk_end - chunk_start) <= 0:
        return

    seq_offsets = chunk_start + tl.arange(0, BLOCK_CHUNK)
    d_offsets = tl.arange(0, BLOCK_D)
    seq_mask = seq_offsets < chunk_end
    d_mask = d_offsets < d_model

    q_off = pid_batch * stride_qb + pid_head * stride_qh
    k_off = pid_batch * stride_kb + pid_head * stride_kh
    v_off = pid_batch * stride_vb + pid_head * stride_vh
    g_off = pid_batch * stride_gb + pid_head * stride_gh

    Q_c = tl.load(
        Q + q_off + seq_offsets[:, None] * stride_qs + d_offsets[None, :] * stride_qd,
        mask=seq_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    K_c = tl.load(
        K + k_off + seq_offsets[:, None] * stride_ks + d_offsets[None, :] * stride_kd,
        mask=seq_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    V_c = tl.load(
        V + v_off + seq_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd,
        mask=seq_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    gates = tl.load(G + g_off + seq_offsets * stride_gs, mask=seq_mask, other=0.0).to(
        tl.float32
    )

    state_off = pid_batch * stride_sb + pid_head * stride_sh + pid_chunk * stride_sd
    prev_state = tl.load(State_in + state_off + d_offsets, mask=d_mask, other=0.0).to(
        tl.float32
    )

    scores = tl.dot(Q_c, tl.trans(K_c))
    causal_mask = seq_offsets[:, None] >= seq_offsets[None, :]
    scores = tl.where(
        causal_mask & seq_mask[:, None] & seq_mask[None, :], scores, float("-inf")
    )
    attn = tl.exp(scores - tl.max(scores, 1)[:, None])
    attn = attn / tl.sum(attn, 1)[:, None]

    out_local = tl.dot(attn * gates[:, None], V_c)
    cumulative_gates = tl.cumprod(gates)
    state_contrib = prev_state[None, :] * cumulative_gates[:, None]
    out_chunk = out_local + state_contrib

    last_mask = seq_offsets == (chunk_end - 1)
    new_state = tl.sum(tl.where(last_mask[:, None], out_chunk, 0.0), axis=0)

    state_out_off = (
        pid_batch * stride_sb + pid_head * stride_sh + (pid_chunk + 1) * stride_sd
    )
    tl.store(State_out + state_out_off + d_offsets, new_state, mask=d_mask)
    tl.store(
        Out
        + pid_batch * stride_ob
        + pid_head * stride_oh
        + seq_offsets[:, None] * stride_os
        + d_offsets[None, :] * stride_od,
        out_chunk,
        mask=seq_mask[:, None] & d_mask[None, :],
    )
