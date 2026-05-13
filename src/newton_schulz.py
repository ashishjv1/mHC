"""
Newton-Schulz iteration for orthogonalization.

Two use cases with different coefficient sets and normalization:

1. **Muon optimizer** — gradient orthogonalization. Uses (3.4445, -4.7750, 2.0315)
   from Keller Jordan's Muon with Frobenius normalization. Equalizes singular
   values but NOT to 1.0. That's fine: Muon only needs the direction.

2. **mHC retraction** — project routing matrices onto O(n). Uses quintic polar
   decomposition coefficients (15/8, -10/8, 3/8) with spectral norm normalization.
   σ=1 is the stable fixed point, so SVs converge to 1.0.

Design note: the user specified (3.4445, -4.7750, 2.0315) for NS. Those are
the Muon coefficients (a+b+c ≈ 0.70, so σ=1 is NOT a fixed point). For Stiefel
retraction we need a+b+c = 1.0, hence (15/8, -10/8, 3/8).
"""
import torch


_MUON_COEFFS = (3.4445, -4.7750, 2.0315)

_POLAR_COEFFS = (15.0 / 8.0, -10.0 / 8.0, 3.0 / 8.0)


def newton_schulz_muon(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Muon-style orthogonalization for gradient updates.

    Normalizes by Frobenius norm. Equalizes SVs to ~0.87, not 1.0.
    """
    assert G.ndim == 2
    a, b, c = _MUON_COEFFS
    X = G.float()
    X = X / (X.norm() + eps)

    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B

    if transposed:
        X = X.T

    return X.to(G.dtype)


def newton_schulz_polar(G: torch.Tensor, steps: int = 7, eps: float = 1e-7) -> torch.Tensor:
    """
    Polar decomposition via Newton-Schulz with spectral norm normalization.

    After normalization, all SVs lie in (0, 1]. The quintic polynomial
    p(σ) = σ(15/8 - 10/8 σ² + 3/8 σ⁴) has σ=1 as its stable fixed point.
    """
    assert G.ndim == 2
    a, b, c = _POLAR_COEFFS
    X = G.float()

    # Spectral norm: ensures σ_max ≤ 1, critical for convergence to σ=1
    spectral = torch.linalg.norm(X, ord=2)
    X = X / (spectral + eps)

    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B

    if transposed:
        X = X.T

    return X.to(G.dtype)


def retract_to_stiefel(X: torch.Tensor, steps: int = 7) -> torch.Tensor:
    """Project X onto O(n) using Newton-Schulz polar decomposition."""
    return newton_schulz_polar(X, steps=steps)
