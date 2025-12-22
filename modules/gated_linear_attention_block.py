import torch  # type: ignore


def fwd_gla(self, q, k, v, x):
    B, H, L, D = q.shape
    gamma = torch.sigmoid((self.W_gamma * x).sum(-1, keepdim=True))
    out = torch.zeros_like(q)
    for h in range(H):
        sn, sd = 0.0, 0.0
        for i in range(L):
            g = gamma[:, h, i : i + 1, :]
            qi, ki, vi = (
                q[:, h, i : i + 1, :],
                k[:, h, i : i + 1, :],
                v[:, h, i : i + 1, :],
            )
            qk = qi @ ki.transpose(-1, -2)
            sn = sn * g + qk @ vi
            sd = sd * g + qk
            out[:, h, i : i + 1, :] = sn / (sd + 1e-8)
    return out
