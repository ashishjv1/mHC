# mHC-nanoGPT

Reproducing key ideas from the DeepSeek V4 paper on top of nanoGPT:

- **Newton-Schulz orthogonalization** — compute $VW^T$ (polar decomposition) without the SVD, using only matmuls
- **Muon optimizer** — orthogonalize momentum via Newton-Schulz before each weight update
- **Manifold-Constrained Hyper-Connections (mHC)** — multi-stream residual with routing matrices retracted onto $O(n)$

## Install

```bash
pip install git+https://github.com/ashishjv1/mHC.git
```

## Blog Notebook

The notebook for [The Most Beautiful Trick in DeepSeek's V4 Paper, Part 1](notebooks/blog_part1_svd_and_muon.ipynb) demonstrates:

1. Newton-Schulz driving all singular values of a random matrix to 1.0 ($G \to VW^T$)
2. AdamW vs Muon training comparison on WikiText-2

Open it in Colab, set your W&B key, and run all cells.

## What's in the repo

```
src/
  newton_schulz.py   # NS iteration (polar + Muon coefficients)
  muon.py            # Muon optimizer
  hyper_connections.py  # mHC routing with A, B matrices
  model.py           # GPT (vanilla + mHC modes)
  train.py           # Training loop with W&B
  data.py            # OpenWebText / FineWeb-Edu prep
configs/
  train_config.py    # Presets: baseline_adamw, baseline_muon, mhc_muon
notebooks/
  blog_part1_svd_and_muon.ipynb
tests/
  test_newton_schulz.py   # NS produces σ ≈ 1.0
  test_mhc_identity.py    # A=B=I matches vanilla GPT
```

## Run experiments

```bash
python -m src.train --preset baseline_adamw --compile
python -m src.train --preset baseline_muon --compile
python -m src.train --preset mhc_muon --compile
```

## Tests

```bash
python -m pytest tests/ -v
```
