"""
Training loop with W&B integration.

Supports three configurations:
  1. Vanilla nanoGPT + AdamW
  2. Vanilla nanoGPT + Muon
  3. mHC nanoGPT + Muon (with Newton-Schulz retraction)
  4. (ablation) HC nanoGPT + Muon without retraction

Usage:
    python -m src.train --preset baseline_adamw
    python -m src.train --preset mhc_muon --compile
"""
import argparse
import math
import os
import time

import numpy as np
import tiktoken
import torch
import wandb

from configs.train_config import TrainConfig, PRESETS
from src.data import DataLoader, prepare_openwebtext, prepare_fineweb_edu
from src.model import GPT
from src.muon import Muon
from src.newton_schulz import retract_to_stiefel


SAMPLE_PROMPTS = [
    "The meaning of life is",
    "In a groundbreaking study, researchers discovered that",
    "Once upon a time, in a land far away,",
    "The key to effective machine learning is",
    "def fibonacci(n):\n",
]


def get_lr(step: int, config: TrainConfig, peak_lr: float, min_lr: float) -> float:
    if step < config.warmup_steps:
        return peak_lr * (step + 1) / config.warmup_steps
    if step >= config.max_steps:
        return min_lr
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (peak_lr - min_lr)


def build_optimizer(model: GPT, config: TrainConfig):
    if config.optimizer == "adamw":
        decay_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and p.ndim >= 2]
        nodecay_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and p.ndim < 2]
        groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            groups, lr=config.lr,
            betas=(config.beta1, config.beta2),
            fused=torch.cuda.is_available(),
        )
        return optimizer, None

    elif config.optimizer == "muon":
        muon_params, adamw_decay, adamw_nodecay = model.get_param_groups()
        muon_opt = Muon(
            [{"params": muon_params}],
            lr=config.muon_lr,
            momentum=config.muon_momentum,
            ns_steps=config.muon_ns_steps,
        )
        adamw_groups = [
            {"params": adamw_decay, "weight_decay": config.weight_decay},
            {"params": adamw_nodecay, "weight_decay": 0.0},
        ]
        adamw_opt = torch.optim.AdamW(
            adamw_groups, lr=config.lr,
            betas=(config.beta1, config.beta2),
            fused=torch.cuda.is_available(),
        )
        return muon_opt, adamw_opt

    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def set_lr(step: int, config: TrainConfig, optimizer, adamw_opt):
    if config.optimizer == "adamw":
        lr = get_lr(step, config, config.lr, config.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr
    else:
        muon_lr = get_lr(step, config, config.muon_lr, config.muon_min_lr)
        adamw_lr = get_lr(step, config, config.lr, config.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = muon_lr
        for g in adamw_opt.param_groups:
            g["lr"] = adamw_lr
        return muon_lr


def retract_routing_matrices(model: GPT, config: TrainConfig):
    """Apply Newton-Schulz retraction to all A, B routing matrices."""
    with torch.no_grad():
        for block in model.blocks:
            if hasattr(block, "hc_attn"):
                block.hc_attn.A.data = retract_to_stiefel(
                    block.hc_attn.A.data, steps=config.retraction_steps
                )
                block.hc_attn.B.data = retract_to_stiefel(
                    block.hc_attn.B.data, steps=config.retraction_steps
                )
                block.hc_ffn.A.data = retract_to_stiefel(
                    block.hc_ffn.A.data, steps=config.retraction_steps
                )
                block.hc_ffn.B.data = retract_to_stiefel(
                    block.hc_ffn.B.data, steps=config.retraction_steps
                )


@torch.no_grad()
def evaluate(model: GPT, val_loader: DataLoader, config: TrainConfig):
    model.eval()
    losses = []
    for _ in range(config.eval_steps):
        x, y = val_loader.get_batch()
        with torch.autocast(device_type="cuda", dtype=getattr(torch, config.dtype)):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)


@torch.no_grad()
def generate_samples(model: GPT, enc, device: str, dtype: str):
    model.eval()
    samples = []
    for prompt_text in SAMPLE_PROMPTS:
        tokens = enc.encode(prompt_text)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=getattr(torch, dtype)):
            out = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=40)
        text = enc.decode(out[0].tolist())
        samples.append({"prompt": prompt_text, "generation": text})
    model.train()
    return samples


def log_singular_values(model: GPT, step: int):
    """Log singular value stats of all routing matrices."""
    metrics = {}
    for i, block in enumerate(model.blocks):
        if not hasattr(block, "hc_attn"):
            continue
        for sub_name, hc in [("attn", block.hc_attn), ("ffn", block.hc_ffn)]:
            for mat_name, mat in [("A", hc.A), ("B", hc.B)]:
                svs = torch.linalg.svdvals(mat.data.float())
                prefix = f"sv/layer{i}_{sub_name}_{mat_name}"
                metrics[f"{prefix}/min"] = svs.min().item()
                metrics[f"{prefix}/max"] = svs.max().item()
                metrics[f"{prefix}/mean"] = svs.mean().item()
                metrics[f"{prefix}/std"] = svs.std().item()
    wandb.log(metrics, step=step)


def log_muon_spectrum(muon_opt: Muon, step: int):
    """Log spectrum stats of Muon's orthogonalized gradient updates."""
    stats = muon_opt.get_spectrum_stats()
    if not stats:
        return
    all_mins, all_maxs, all_means = [], [], []
    for s in stats.values():
        all_mins.append(s["min"])
        all_maxs.append(s["max"])
        all_means.append(s["mean"])
    wandb.log({
        "muon_spectrum/min": np.mean(all_mins),
        "muon_spectrum/max": np.mean(all_maxs),
        "muon_spectrum/mean": np.mean(all_means),
        "muon_spectrum/condition": np.mean(all_maxs) / (np.mean(all_mins) + 1e-8),
    }, step=step)


def train(config: TrainConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Data
    if config.dataset == "openwebtext":
        prepare_openwebtext(config.data_dir)
    elif config.dataset == "fineweb_edu":
        prepare_fineweb_edu(config.data_dir)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    train_loader = DataLoader(config.data_dir, "train", config.batch_size,
                              config.context_len, config.device)
    val_loader = DataLoader(config.data_dir, "val", config.batch_size,
                            config.context_len, config.device)

    # Model
    model = GPT(config).to(config.device)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Estimated from config: {config.estimate_params():,}")
    print(f"Tokens per step: {config.tokens_per_step:,}")
    print(f"Total tokens: {config.total_tokens:,}")

    if config.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Optimizer
    optimizer, adamw_opt = build_optimizer(
        model.module if hasattr(model, "module") else model, config
    )
    # Handle compiled model
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    # W&B
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        tags=config.wandb_tags,
        group=config.wandb_group,
        config={
            "n_layers": config.n_layers,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "d_ff": config.d_ff,
            "vocab_size": config.vocab_size,
            "context_len": config.context_len,
            "use_mhc": config.use_mhc,
            "n_streams": config.n_streams,
            "hc_mode": config.hc_mode,
            "optimizer": config.optimizer,
            "lr": config.lr,
            "muon_lr": config.muon_lr,
            "batch_size": config.batch_size,
            "grad_accum_steps": config.grad_accum_steps,
            "max_steps": config.max_steps,
            "total_tokens": config.total_tokens,
            "param_count": raw_model.count_parameters(),
            "seed": config.seed,
        },
    )

    enc = tiktoken.get_encoding("gpt2")
    os.makedirs(config.ckpt_dir, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(config.max_steps):
        step_t0 = time.time()

        # LR schedule
        lr = set_lr(step, config, optimizer, adamw_opt)

        # Gradient accumulation
        model.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(config.grad_accum_steps):
            x, y = train_loader.get_batch()
            with torch.autocast(device_type="cuda", dtype=getattr(torch, config.dtype)):
                _, loss = model(x, y)
            loss = loss / config.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), config.grad_clip
        )

        # Optimizer step
        optimizer.step()
        if adamw_opt is not None:
            adamw_opt.step()

        # mHC retraction
        if config.use_mhc and config.hc_mode == "constrained":
            retract_routing_matrices(raw_model, config)

        step_time = time.time() - step_t0
        tokens_per_sec = config.tokens_per_step / step_time

        # Logging
        if step % config.log_interval == 0:
            metrics = {
                "train/loss": accum_loss,
                "train/lr": lr,
                "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                "train/tokens_per_sec": tokens_per_sec,
                "train/step_time": step_time,
                "train/tokens_seen": (step + 1) * config.tokens_per_step,
            }
            wandb.log(metrics, step=step)

            if step % (config.log_interval * 10) == 0:
                elapsed = time.time() - t0
                print(
                    f"step {step:>6d} | loss {accum_loss:.4f} | "
                    f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                    f"tok/s {tokens_per_sec:,.0f} | elapsed {elapsed:.0f}s"
                )

        # Evaluation
        if step % config.eval_interval == 0 and step > 0:
            val_loss = evaluate(raw_model, val_loader, config)
            wandb.log({"val/loss": val_loss}, step=step)
            print(f"  val_loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(config.ckpt_dir, f"{config.wandb_run_name}_best.pt")
                torch.save({
                    "model": raw_model.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                    "config": config,
                }, ckpt_path)

        # Generate samples
        if step % config.sample_interval == 0 and step > 0:
            samples = generate_samples(raw_model, enc, config.device, config.dtype)
            table = wandb.Table(columns=["prompt", "generation", "step"])
            for s in samples:
                table.add_data(s["prompt"], s["generation"], step)
            wandb.log({"samples": table}, step=step)

        # Singular value logging (mHC)
        if step % config.sv_log_interval == 0 and config.use_mhc:
            log_singular_values(raw_model, step)

        # Muon spectrum logging
        if step % config.sv_log_interval == 0 and config.optimizer == "muon":
            log_muon_spectrum(optimizer, step)

        # Checkpointing
        if step % config.save_interval == 0 and step > 0:
            ckpt_path = os.path.join(config.ckpt_dir, f"{config.wandb_run_name}_step{step}.pt")
            torch.save({
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "adamw_opt": adamw_opt.state_dict() if adamw_opt else None,
                "step": step,
                "config": config,
            }, ckpt_path)

    # Final evaluation
    val_loss = evaluate(raw_model, val_loader, config)
    total_time = time.time() - t0

    wandb.log({"val/loss": val_loss}, step=config.max_steps)
    wandb.run.summary["final_val_loss"] = val_loss
    wandb.run.summary["total_time_s"] = total_time
    wandb.run.summary["param_count"] = raw_model.count_parameters()
    wandb.run.summary["total_tokens"] = config.total_tokens
    wandb.run.summary["avg_tokens_per_sec"] = config.total_tokens / total_time

    print(f"\nTraining complete.")
    print(f"  Final val loss: {val_loss:.4f}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Avg tokens/sec: {config.total_tokens / total_time:,.0f}")

    # Save final checkpoint
    ckpt_path = os.path.join(config.ckpt_dir, f"{config.wandb_run_name}_final.pt")
    torch.save({
        "model": raw_model.state_dict(),
        "step": config.max_steps,
        "val_loss": val_loss,
        "config": config,
    }, ckpt_path)

    wandb.finish()
    return val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, required=True, choices=list(PRESETS.keys()))
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--compile", action="store_true", default=None)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = PRESETS[args.preset]()

    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        config.grad_accum_steps = args.grad_accum_steps
    if args.compile is not None:
        config.compile = args.compile
    if args.no_compile:
        config.compile = False
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed

    train(config)


if __name__ == "__main__":
    main()
