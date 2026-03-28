import math
import torch  # type: ignore


def fw_awa(q, k, v):
    B, H, L, D = q.shape
    mt = self.meta_tokens.expand(B, -1, -1, -1)
    out = torch.zeros_like(q)
    for i in range(L):
        s, e = max(0, i - self.window_size + 1), min(L, i + self.window_size)
        qi = q[:, :, i : i + 1, :]
        kw, vw = k[:, :, s:e, :], v[:, :, s:e, :]
        sk, st = torch.matmul(qi, kw.transpose(-2, -1)) / math.sqrt(D), torch.matmul(
            qi, mt.transpose(-2, -1)
        ) / math.sqrt(D)
        mv = torch.max(torch.cat([sk, st], dim=-1), dim=-1, keepdim=True)[0]
        ek, et = torch.exp(sk - mv), torch.exp(st - mv)
        out[:, :, i : i + 1, :] = torch.matmul(ek, vw) / (
            ek.sum(-1, True) + et.sum(-1, True) + 1e-8
        )
    return out
