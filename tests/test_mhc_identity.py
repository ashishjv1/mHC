"""
Sanity check: with A=B=identity for all HyperConnection modules,
the mHC model must produce the same output as the vanilla model
within numerical noise.

Strategy:
  1. Create vanilla and mHC configs with identical architecture
  2. Instantiate both models
  3. Copy all shared weights from vanilla → mHC
  4. Verify A, B are already identity (from init)
  5. Forward the same input, check outputs match
"""
import torch
import pytest
from configs.train_config import TrainConfig
from src.model import GPT


def make_small_config(use_mhc: bool) -> TrainConfig:
    return TrainConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_ff=256,
        vocab_size=256,
        context_len=32,
        dropout=0.0,
        bias=False,
        use_mhc=use_mhc,
        n_streams=4,
        compile=False,
    )


def test_mhc_identity_matches_vanilla():
    torch.manual_seed(42)

    vanilla_config = make_small_config(use_mhc=False)
    mhc_config = make_small_config(use_mhc=True)

    vanilla = GPT(vanilla_config)
    mhc = GPT(mhc_config)

    # Copy shared weights
    vanilla_sd = vanilla.state_dict()
    mhc_sd = mhc.state_dict()

    for key in vanilla_sd:
        assert key in mhc_sd, f"Key {key} missing in mHC model"
        mhc_sd[key] = vanilla_sd[key].clone()

    mhc.load_state_dict(mhc_sd)

    # Verify all A, B are identity
    for block in mhc.blocks:
        assert torch.allclose(block.hc_attn.A, torch.eye(4))
        assert torch.allclose(block.hc_attn.B, torch.eye(4))
        assert torch.allclose(block.hc_ffn.A, torch.eye(4))
        assert torch.allclose(block.hc_ffn.B, torch.eye(4))

    # Forward pass
    idx = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))

    vanilla.eval()
    mhc.eval()

    with torch.no_grad():
        v_logits, v_loss = vanilla(idx, targets)
        m_logits, m_loss = mhc(idx, targets)

    assert torch.allclose(v_logits, m_logits, atol=1e-5), \
        f"Logits differ: max delta = {(v_logits - m_logits).abs().max():.8f}"
    assert torch.allclose(v_loss, m_loss, atol=1e-5), \
        f"Loss differs: vanilla={v_loss:.6f}, mhc={m_loss:.6f}"


def test_mhc_streams_carry_information():
    """After a few gradient steps, non-zero streams should actually be used."""
    torch.manual_seed(0)
    config = make_small_config(use_mhc=True)
    model = GPT(config)

    # Perturb A, B away from identity
    with torch.no_grad():
        for block in model.blocks:
            block.hc_attn.A.add_(0.1 * torch.randn_like(block.hc_attn.A))
            block.hc_attn.B.add_(0.1 * torch.randn_like(block.hc_attn.B))
            block.hc_ffn.A.add_(0.1 * torch.randn_like(block.hc_ffn.A))
            block.hc_ffn.B.add_(0.1 * torch.randn_like(block.hc_ffn.B))

    # Vanilla model with same base weights should now give different output
    vanilla_config = make_small_config(use_mhc=False)
    vanilla = GPT(vanilla_config)

    vanilla_sd = vanilla.state_dict()
    mhc_sd = model.state_dict()
    for key in vanilla_sd:
        if key in mhc_sd:
            vanilla_sd[key] = mhc_sd[key].clone()
    vanilla.load_state_dict(vanilla_sd)

    idx = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        v_logits, _ = vanilla(idx)
        m_logits, _ = model(idx)

    assert not torch.allclose(v_logits, m_logits, atol=1e-3), \
        "Perturbed A/B should cause different outputs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
