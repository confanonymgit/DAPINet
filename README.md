# DAPINet: Distribution-Aware Permutation-Invariant Network

DAPINet is an end-to-end meta-learning framework for selecting a suitable clustering algorithm for tabular data. It operates directly on raw n&times;m tables, enforces double permutation invariance over rows and columns, and uses distribution-aware column fingerprints to capture feature identity without positional indices. The model predicts expected performance across a pool of clustering algorithms and recommends the best candidate, achieving near-oracle quality with substantially lower compute than exhaustive search.

## Key Ideas
- **Double permutation invariance** across rows (objects) and columns (features).
- **Distribution-Aware Column Embeddings (DACE)** using quantile fingerprints to encode feature identity.
- **Set-of-sets architecture**: column interaction per row, followed by row interaction and attention pooling.
- **Predict-then-optimize** recommendation that approximates oracle performance while reducing compute.

## Highlights
- Trained on a large synthetic repository of diverse clustering topologies.
- Strong zero-shot generalization to real-world UCI benchmarks.
- Significant regret reduction compared to AutoML methods and CVIs.
- ~12&times; speedup over exhaustive hyperparameter search.

## Repository Contents
- [src/dapinet](src/dapinet): Core implementation.
- [scripts](scripts): Data generation, training, and evaluation utilities.
- [notebooks](notebooks): Analysis and plotting notebooks used in the paper.
- [models](models): Trained checkpoints and training histories.

## How It Works
1. **Dataset synthesis** generates diverse tabular datasets spanning convex, manifold, and density-based structures.
2. **Oracle labeling** uses Optuna to compute best achievable ARI per algorithm.
3. **DAPINet** predicts algorithm performance from raw data and recommends a candidate.

## Reproducibility
All experiments reported in the paper are derived from the scripts and notebooks in this repository.

## Project Startup
The recommended setup is with `uv` for dependency management and command execution. See [astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/).

### Sync dependencies
```
uv sync
```

### Run any script
```
uv run <command>
```

### Alternative (venv + pip)
If `uv` is not available, a standard virtual environment with pip also works.
```
python -m venv .venv
# activate the virtual environment
python -m pip install -e .
```

## Quick Run
### Dataset generation
Generate synthetic tabular datasets for training and evaluation.
```
uv run python scripts/01_generate_datasets.py
```
Key parameters:
- `--n-repeats`
- `--n-configs`

### Dataset clustering
Compute clustering results and oracle labels for generated datasets.
```
uv run python scripts/02_clustering_datasets.py
```
Key parameters:
- `--n-trials`
- `--optuna-jobs`
- `--timeout`

### Deep model training
Train DAPINet on the prepared dataset repository.
```
uv run python scripts/03_train_model.py
```
Key parameters:
- `--d-model`
- `--num-bins`

## Citation
This section will be updated upon paper acceptance.
