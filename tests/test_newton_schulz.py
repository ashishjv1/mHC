"""
Verify Newton-Schulz produces near-orthogonal matrices.

- retract_to_stiefel (polar coefficients): SVs within ~1e-3 of 1.0
- newton_schulz_muon (Muon coefficients): SVs equalized but not to 1.0
"""
import torch
import pytest
from src.newton_schulz import newton_schulz_muon, newton_schulz_polar, retract_to_stiefel


# --- Polar decomposition (retraction) tests ---

def test_retraction_4x4_random():
    """4×4 is the actual routing matrix size used in training."""
    torch.manual_seed(42)
    X = torch.randn(4, 4)
    X_orth = retract_to_stiefel(X, steps=7)
    svs = torch.linalg.svdvals(X_orth.float())
    assert torch.allclose(svs, torch.ones_like(svs), atol=1e-3), \
        f"SVs not near 1.0: min={svs.min():.6f}, max={svs.max():.6f}"


@pytest.mark.parametrize("size", [(4, 4), (8, 8), (16, 16), (32, 32)])
def test_retraction_near_identity(size):
    """Near-identity matrices (realistic for training) converge fast at any size."""
    torch.manual_seed(7)
    X = torch.eye(size[0]) + 0.1 * torch.randn(*size)
    X_orth = retract_to_stiefel(X, steps=7)
    svs = torch.linalg.svdvals(X_orth.float())
    assert torch.allclose(svs, torch.ones_like(svs), atol=1e-3), \
        f"SVs not near 1.0: min={svs.min():.6f}, max={svs.max():.6f}"


@pytest.mark.parametrize("size", [(8, 8), (16, 16), (64, 64)])
def test_retraction_random_more_iterations(size):
    """Arbitrary random matrices converge with enough iterations."""
    torch.manual_seed(42)
    X = torch.randn(*size)
    X_orth = retract_to_stiefel(X, steps=15)
    svs = torch.linalg.svdvals(X_orth.float())
    assert torch.allclose(svs, torch.ones_like(svs), atol=1e-3), \
        f"SVs not near 1.0: min={svs.min():.6f}, max={svs.max():.6f}"


@pytest.mark.parametrize("size", [(4, 4), (32, 32)])
def test_retraction_deterministic(size):
    torch.manual_seed(0)
    X = torch.randn(*size)
    Y1 = retract_to_stiefel(X, steps=7)
    Y2 = retract_to_stiefel(X, steps=7)
    assert torch.allclose(Y1, Y2, atol=1e-6)


def test_retraction_preserves_identity():
    X = torch.eye(4)
    X_orth = retract_to_stiefel(X, steps=7)
    assert torch.allclose(X_orth, torch.eye(4), atol=1e-3)


def test_retraction_rectangular():
    """NS handles non-square matrices (transposes internally)."""
    torch.manual_seed(3)
    X = torch.randn(8, 4)
    X_orth = retract_to_stiefel(X, steps=15)
    svs = torch.linalg.svdvals(X_orth.float())
    assert svs.shape[0] == 4
    assert torch.allclose(svs, torch.ones_like(svs), atol=1e-3)


# --- Muon-style orthogonalization tests ---

def test_muon_ns_equalizes_singular_values():
    """Muon NS should make SVs more uniform (lower spread), though not exactly 1.0."""
    torch.manual_seed(1)
    X = torch.randn(8, 8)
    svs_before = torch.linalg.svdvals(X)
    X_ns = newton_schulz_muon(X, steps=5)
    svs_after = torch.linalg.svdvals(X_ns.float())
    assert svs_after.std() < svs_before.std(), "NS should reduce SV spread"


def test_muon_ns_output_shape():
    X = torch.randn(32, 64)
    Y = newton_schulz_muon(X, steps=5)
    assert Y.shape == X.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
