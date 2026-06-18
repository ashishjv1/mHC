"""
Sanity check: with the effective routing matrices A=B=identity and static
slot-0 selection, the mHC model must produce the same output as the vanilla
model within numerical noise.

Since A/B go through Sinkhorn, identity is achieved by setting the *logits* to a
large-diagonal matrix (Sinkhorn(eye * c) -> identity as c grows).

Strategy:
  1. Create vanilla and mHC configs with identical architecture
  2. Instantiate both models, copy all shared weights vanilla -> mHC
  3. Force A_logits/B_logits to a large diagonal and verify effective A,B ~ I
  4. Forward the same input, check outputs match
"""
import torch
import pytest
from configs.train_config import TrainConfig
from src.model import GPT

# Sinkhorn(eye * 40) is identity to ~1e-11, so the routing is effectively exact.
_IDENTITY_LOGIT_SCALE = 40.0


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

    # Force effective routing matrices to identity via large-diagonal logits.
    eye = torch.eye(4) * _IDENTITY_LOGIT_SCALE
    with torch.no_grad():
        for block in mhc.blocks:
            for hc in (block.hc_attn, block.hc_ffn):
                hc.A_logits.copy_(eye)
                hc.B_logits.copy_(eye)

    # Verify all effective A, B are identity
    for block in mhc.blocks:
        for hc in (block.hc_attn, block.hc_ffn):
            assert torch.allclose(hc.mix_matrix(hc.A_logits), torch.eye(4), atol=1e-6)
            assert torch.allclose(hc.mix_matrix(hc.B_logits), torch.eye(4), atol=1e-6)

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

    # Perturb A, B logits away from identity
    with torch.no_grad():
        for block in model.blocks:
            for hc in (block.hc_attn, block.hc_ffn):
                hc.A_logits.add_(0.5 * torch.randn_like(hc.A_logits))
                hc.B_logits.add_(0.5 * torch.randn_like(hc.B_logits))

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
