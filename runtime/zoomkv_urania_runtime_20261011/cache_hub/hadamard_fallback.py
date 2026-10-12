"""PyTorch FWHT fallback when fast_hadamard_transform is unavailable.

Matches Dao-AILab ``hadamard_transform(x, scale=)`` on the last dimension
(power-of-two). Used by ParisKV SRHT (normalize + rotate).
"""

import torch


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    n = int(x.shape[-1])
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"hadamard last dim must be a power of 2, got {n}")
    orig_shape = x.shape
    orig_dtype = x.dtype
    y = x.reshape(-1, n).float()
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.cat((a + b, a - b), dim=-1).reshape(-1, n)
        h *= 2
    if scale != 1.0:
        y = y * float(scale)
    return y.to(orig_dtype).reshape(orig_shape)
