from dataclasses import dataclass, field
from typing import List, Optional
import math


@dataclass
class TrainConfig:
    # --- Model ---
    n_layers: int = 24
    d_model: int = 1024
    n_heads: int = 16
    d_ff: int = 4096
    vocab_size: int = 50304  # GPT-2 50257 rounded up to nearest 64
    context_len: int = 2048
    dropout: float = 0.0
    bias: bool = False

    # --- mHC ---
    use_mhc: bool = False
    n_streams: int = 4
    # "constrained" Sinkhorn-projects A/B onto doubly-stochastic matrices;
    # "unconstrained" uses the raw parameter matrices (ablation).
    hc_mode: str = "constrained"
    # Stream selection: "static" | "learnable" | "per_token"
    hc_selection: str = "static"
    # Whether to apply the Sinkhorn A/B mixing matrices
    hc_mix: bool = True

    # --- Optimizer ---
    optimizer: str = "adamw"  # "adamw" or "muon"
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    muon_lr: float = 0.02
    muon_min_lr: float = 0.002
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    grad_clip: float = 1.0

    # --- Training ---
    batch_size: int = 32
    grad_accum_steps: int = 8
    max_steps: int = 19073  # ~10B tokens with batch_size=32, seq=2048, accum=8
    warmup_steps: int = 2000

    # --- Evaluation ---
    eval_interval: int = 250
    eval_steps: int = 20

    # --- Logging ---
    log_interval: int = 10
    sample_interval: int = 500
    sv_log_interval: int = 500

    # --- Data ---
    dataset: str = "openwebtext"
    data_dir: str = "data"

    # --- W&B ---
    wandb_project: str = "mhc-nanogpt"
    wandb_run_name: str = ""
    wandb_tags: List[str] = field(default_factory=list)
    wandb_group: str = "comparison"

    # --- System ---
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    seed: int = 42
    num_workers: int = 4

    # --- Checkpointing ---
    ckpt_dir: str = "checkpoints"
    save_interval: int = 2000

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.context_len * self.grad_accum_steps

    @property
    def total_tokens(self) -> int:
        return self.tokens_per_step * self.max_steps

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def estimate_params(self) -> int:
        emb = self.vocab_size * self.d_model
        pos = self.context_len * self.d_model
        per_layer = (
            3 * self.d_model * self.d_model  # QKV
            + self.d_model * self.d_model      # attn out
            + self.d_model * self.d_ff         # FFN up
            + self.d_ff * self.d_model         # FFN down
            + 4 * self.d_model                 # 2x LayerNorm
        )
        ln_f = 2 * self.d_model
        mhc = 0
        if self.use_mhc:
            mhc = self.n_layers * 2 * 2 * (self.n_streams ** 2)
        return emb + pos + self.n_layers * per_layer + ln_f + mhc


def baseline_adamw() -> TrainConfig:
    return TrainConfig(
        use_mhc=False,
        optimizer="adamw",
        lr=6e-4,
        min_lr=6e-5,
        wandb_run_name="baseline_adamw",
        wandb_tags=["baseline", "adamw"],
    )


def baseline_muon() -> TrainConfig:
    return TrainConfig(
        use_mhc=False,
        optimizer="muon",
        muon_lr=0.02,
        muon_min_lr=0.002,
        lr=3e-4,       # AdamW LR for non-matrix params
        min_lr=3e-5,
        wandb_run_name="baseline_muon",
        wandb_tags=["baseline", "muon"],
    )


def mhc_muon() -> TrainConfig:
    return TrainConfig(
        use_mhc=True,
        n_streams=4,
        hc_mode="constrained",
        optimizer="muon",
        muon_lr=0.02,
        muon_min_lr=0.002,
        lr=3e-4,
        min_lr=3e-5,
        wandb_run_name="mhc_muon",
        wandb_tags=["mhc", "muon", "constrained"],
    )


def mhc_muon_unconstrained() -> TrainConfig:
    """Ablation: hyper-connections without the manifold constraint."""
    return TrainConfig(
        use_mhc=True,
        n_streams=4,
        hc_mode="unconstrained",
        optimizer="muon",
        muon_lr=0.02,
        muon_min_lr=0.002,
        lr=3e-4,
        min_lr=3e-5,
        wandb_run_name="hc_muon_unconstrained",
        wandb_tags=["hc", "muon", "unconstrained", "ablation"],
    )


PRESETS = {
    "baseline_adamw": baseline_adamw,
    "baseline_muon": baseline_muon,
    "mhc_muon": mhc_muon,
    "hc_muon_unconstrained": mhc_muon_unconstrained,
}
