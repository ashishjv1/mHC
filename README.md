# mHC-nanoGPT

Manifold-Constrained Hyper-Connections (mHC) implemented on top of nanoGPT,
trained with the Muon optimizer and compared against vanilla baselines.

## Architecture

- **Model**: 24-layer transformer, 1024 hidden dim, 16 heads, 4096 FFN (~356M params)
- **Context**: 2048 tokens, GPT-2 tokenizer (50304 vocab, padded for GPU alignment)
- **Precision**: bf16 mixed precision with torch.compile

### Why 24 layers instead of 12–16?

With 1024 hidden dim, 16 layers only yields ~255M params — below the 350M target.
Rather than widening the model (which changes head_dim away from the efficient 64),
we increase depth to 24 layers. This also gives mHC more routing opportunities,
making the comparison fairer.

## Configurations

| Config | Architecture | Optimizer | Notes |
|--------|-------------|-----------|-------|
| `baseline_adamw` | Vanilla GPT | AdamW | Standard baseline |
| `baseline_muon` | Vanilla GPT | Muon + AdamW | Isolates optimizer contribution |
| `mhc_muon` | mHC GPT (n=4 streams) | Muon + AdamW | Full proposed combo |
| `hc_muon_unconstrained` | HC GPT (n=4, no retraction) | Muon + AdamW | Ablation |

All configs share: identical seed (42), data ordering, and hyperparameters
except where the configuration necessarily differs.

## Hyper-Connections (mHC)

The model maintains **n=4 parallel residual streams** as a single tensor
`(batch, seq, n_streams, d_model)` for memory efficiency.

Before each sublayer (attention or FFN):
- Routing matrix **A** ∈ R^{4×4} mixes streams → extracts sublayer input from stream 0
- Sublayer processes the single-stream input as usual

After each sublayer:
- Routing matrix **B** ∈ R^{4×4} distributes the output back across streams

**Manifold constraint**: After each optimizer step, A and B are projected onto
O(4) (the orthogonal group) via Newton-Schulz retraction. This prevents the
routing from collapsing or exploding the multi-stream representation.

**Initialization**: A = B = I (identity), so the model matches vanilla GPT at
init — stream 0 is the active residual, streams 1–3 are inert zeros.

## Newton-Schulz Orthogonalization

Used in two places:
1. **Muon optimizer**: orthogonalizes momentum of weight matrices (equalizes
   singular values of the update direction)
2. **mHC retraction**: projects A, B routing matrices back to O(n)

Implementation uses the quintic polynomial with coefficients (3.4445, −4.7750, 2.0315)
and 5 iterations. Input normalized by Frobenius norm. For retraction, a final
rescaling step maps the converged singular values from ~0.868 to ~1.0.

## Muon Optimizer

For 2D weight matrices: momentum is orthogonalized via Newton-Schulz before use
as the update direction. Learning rate ~0.02.

For non-matrix params (embeddings, biases, layer norms, routing matrices):
standard AdamW with learning rate ~3e-4.

## Token Budget

Target: ~10B tokens (chinchilla-optimal for ~356M params).

With `batch_size=32 × seq_len=2048 × grad_accum=8 = 524,288 tokens/step`:
- 19,073 steps → ~10B tokens
- Estimated wallclock on single A100: ~1.5–2h per run

If using Colab with T4, the notebook auto-scales to ~1B tokens with smaller batch.

## Usage

### A100 Cluster

```bash
# Install deps
pip install -r requirements.txt

# Set W&B key
export WANDB_API_KEY=your_key_here

# Run all experiments
bash scripts/run_all.sh

# Or run individually
python -m src.train --preset baseline_adamw --compile
python -m src.train --preset baseline_muon --compile
python -m src.train --preset mhc_muon --compile

# With ablation
bash scripts/run_all.sh --ablation
```

### Colab

Open `colab_runner.ipynb`, set your repo URL and W&B key, run all cells.

### Tests

```bash
python -m pytest tests/ -v
```

### Generate Plots

```bash
python scripts/plot_results.py
```

## Project Structure

```
├── configs/
│   └── train_config.py    # Dataclass configs + presets
├── src/
│   ├── model.py           # GPT model (vanilla + mHC)
│   ├── hyper_connections.py  # HyperConnection module
│   ├── newton_schulz.py   # NS iteration + Stiefel retraction
│   ├── muon.py            # Muon optimizer
│   ├── data.py            # Data prep + loading
│   └── train.py           # Training loop + W&B integration
├── tests/
│   ├── test_newton_schulz.py  # NS produces σ ≈ 1.0
│   └── test_mhc_identity.py  # A=B=I matches vanilla
├── scripts/
│   ├── run_all.sh         # Launch all experiments
│   └── plot_results.py    # Generate comparison plots
├── colab_runner.ipynb     # Thin Colab wrapper
└── results/               # Generated plots + summary
```

## W&B Integration

All runs log to project `mhc-nanogpt` with clear tags and grouping.

Tracked metrics:
- `train/loss`, `val/loss` (eval every 250 steps)
- `train/lr`, `train/grad_norm`, `train/tokens_per_sec`, `train/step_time`
- `sv/*` — singular value stats of A, B routing matrices (mHC, every 500 steps)
- `muon_spectrum/*` — spectrum of orthogonalized gradient updates (Muon, every 500 steps)
- `samples` — generation samples from fixed prompts (W&B table, every 500 steps)
