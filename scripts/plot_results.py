"""
Generate comparison plots from W&B runs.

Produces:
  - results/loss_curves.png: train/val loss for all configs
  - results/singular_values.png: SV evolution for mHC run(s)
  - results/summary.md: final metrics table

Usage:
    python scripts/plot_results.py [--wandb_project mhc-nanogpt]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


RUN_STYLES = {
    "baseline_adamw": {"color": "#1f77b4", "label": "Vanilla + AdamW", "ls": "-"},
    "baseline_muon": {"color": "#ff7f0e", "label": "Vanilla + Muon", "ls": "-"},
    "mhc_muon": {"color": "#2ca02c", "label": "mHC + Muon", "ls": "-"},
    "hc_muon_unconstrained": {"color": "#d62728", "label": "HC (no constraint) + Muon", "ls": "--"},
}


def fetch_wandb_data(project: str):
    api = wandb.Api()
    runs = api.runs(project)
    data = {}
    for run in runs:
        name = run.name
        if name not in RUN_STYLES:
            continue
        history = run.scan_history(keys=["train/loss", "val/loss", "_step"])
        train_steps, train_losses = [], []
        val_steps, val_losses = [], []
        for row in history:
            step = row.get("_step", 0)
            if "train/loss" in row and row["train/loss"] is not None:
                train_steps.append(step)
                train_losses.append(row["train/loss"])
            if "val/loss" in row and row["val/loss"] is not None:
                val_steps.append(step)
                val_losses.append(row["val/loss"])

        sv_data = {}
        if "mhc" in name or "unconstrained" in name:
            sv_history = run.scan_history(
                keys=[k for k in run.history().columns if k.startswith("sv/")]
            )
            for row in sv_history:
                for k, v in row.items():
                    if k.startswith("sv/") and v is not None:
                        sv_data.setdefault(k, []).append(v)

        data[name] = {
            "train_steps": train_steps,
            "train_losses": train_losses,
            "val_steps": val_steps,
            "val_losses": val_losses,
            "sv_data": sv_data,
            "summary": run.summary._json_dict,
        }
    return data


def plot_loss_curves(data: dict, output_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, d in data.items():
        style = RUN_STYLES.get(name, {"color": "gray", "label": name, "ls": "-"})
        if d["train_steps"]:
            ax1.plot(d["train_steps"], d["train_losses"],
                     color=style["color"], ls=style["ls"],
                     label=style["label"], alpha=0.8, linewidth=0.8)
        if d["val_steps"]:
            ax2.plot(d["val_steps"], d["val_losses"],
                     color=style["color"], ls=style["ls"],
                     label=style["label"], linewidth=1.5)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Val Loss")
    ax2.set_title("Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_singular_values(data: dict, output_dir: str):
    mhc_runs = {k: v for k, v in data.items() if v["sv_data"]}
    if not mhc_runs:
        print("No SV data found, skipping SV plot.")
        return

    fig, axes = plt.subplots(1, len(mhc_runs), figsize=(7 * len(mhc_runs), 5),
                             squeeze=False)

    for idx, (name, d) in enumerate(mhc_runs.items()):
        ax = axes[0][idx]
        style = RUN_STYLES.get(name, {"color": "gray", "label": name})

        # Aggregate: plot mean of min/max across layers
        min_keys = [k for k in d["sv_data"] if k.endswith("/min")]
        max_keys = [k for k in d["sv_data"] if k.endswith("/max")]

        if min_keys and max_keys:
            n_points = min(len(d["sv_data"][min_keys[0]]), 200)
            mins = np.mean([d["sv_data"][k][:n_points] for k in min_keys], axis=0)
            maxs = np.mean([d["sv_data"][k][:n_points] for k in max_keys], axis=0)
            steps = np.arange(n_points)

            ax.fill_between(steps, mins, maxs, alpha=0.3, color=style["color"])
            ax.plot(steps, mins, color=style["color"], alpha=0.7, label="min σ (avg)")
            ax.plot(steps, maxs, color=style["color"], alpha=0.7, label="max σ (avg)")
            ax.axhline(y=1.0, color="black", ls="--", alpha=0.5, label="target σ=1")

        ax.set_xlabel("Log point")
        ax.set_ylabel("Singular value")
        ax.set_title(f"SV Evolution: {style.get('label', name)}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "singular_values.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def write_summary(data: dict, output_dir: str):
    lines = [
        "# Experiment Results\n",
        "| Config | Params | Final Val Loss | Tokens/sec | Wallclock (h) | Total Tokens |",
        "|--------|--------|----------------|------------|---------------|--------------|",
    ]

    for name in ["baseline_adamw", "baseline_muon", "mhc_muon", "hc_muon_unconstrained"]:
        if name not in data:
            continue
        s = data[name].get("summary", {})
        params = s.get("param_count", "?")
        val_loss = s.get("final_val_loss", "?")
        tok_sec = s.get("avg_tokens_per_sec", "?")
        total_time = s.get("total_time_s", "?")
        total_tokens = s.get("total_tokens", "?")

        hours = f"{total_time/3600:.1f}" if isinstance(total_time, (int, float)) else "?"
        tok_sec_str = f"{tok_sec:,.0f}" if isinstance(tok_sec, (int, float)) else "?"
        val_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else "?"
        params_str = f"{params:,}" if isinstance(params, int) else str(params)
        tok_str = f"{total_tokens/1e9:.1f}B" if isinstance(total_tokens, (int, float)) else "?"

        style = RUN_STYLES.get(name, {"label": name})
        lines.append(
            f"| {style['label']} | {params_str} | {val_str} | "
            f"{tok_sec_str} | {hours} | {tok_str} |"
        )

    lines.append("")
    lines.append("## Notes\n")
    lines.append("- All runs use identical seed, data order, and hyperparameters "
                 "(except optimizer/architecture).")
    lines.append("- Model: 24 layers, 1024 hidden, 16 heads, 4096 FFN, ~356M params.")
    lines.append("- Context length: 2048. Mixed precision: bf16.")
    lines.append("- mHC uses n=4 parallel residual streams with Newton-Schulz "
                 "retraction on A, B routing matrices.")
    lines.append("- Muon uses NS-orthogonalized momentum for weight matrices, "
                 "AdamW for embeddings/biases/norms.")

    path = os.path.join(output_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default="mhc-nanogpt")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not HAS_WANDB:
        print("wandb not installed. Cannot fetch data.")
        return

    print("Fetching data from W&B...")
    data = fetch_wandb_data(args.wandb_project)

    if not data:
        print("No matching runs found. Make sure runs are named: "
              "baseline_adamw, baseline_muon, mhc_muon")
        return

    plot_loss_curves(data, args.output_dir)
    plot_singular_values(data, args.output_dir)
    write_summary(data, args.output_dir)


if __name__ == "__main__":
    main()
