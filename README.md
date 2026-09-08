# GNN-Based BERT for Understanding Context from Music

Supervised Neural Network Project for CSE425.

## Overview
This repository implements a hybrid BERT + Graph Neural Network (GNN) system that models musical context by fusing relational chord/segment graph structures with contextual text token representations.

## Key Architectures
- Baseline 1 (Random Prior): Class prior probability distribution.
- Baseline 2 (Mel-CNN): 2D CNN on 128-bin log-mel spectrograms.
- Task 1 (DistilBERT): Multi-label tag classifier on text context embeddings.
- Task 2 (GraphSAGE): GNN processing segment-level chroma representations over temporal and chord cosine affinity graphs.
- Task 3 (Cross-Attention Fusion): Fuses GNN graph-level readout query vectors with token key/values from DistilBERT.
- Task 4 (Contrastive Dual-Encoder): Bidirectional InfoNCE alignment mapping audio structure and natural text to a shared metric space.

## Directory Structure
gnn-bert-music-context/
├── data/
│   └── processed/
│       └── graphs/          # >= 20 preprocessed sample .pt graphs
├── notebooks/
│   └── demo_context.ipynb   # End-to-end inference demo notebook
├── results/
│   ├── metrics.json         # Complete benchmark logs
│   ├── plots/               # F1 convergence and t-SNE latent manifold plots
│   └── retrieval_examples/  # Top-3 cross-modal retrieval JSON records
├── src/
│   ├── audio_features.py    # Log-mel and chroma segment extraction
│   ├── graph_builder.py     # Temporal and cosine graph topology builder
│   ├── bert_encoder.py      # DistilBERT classifier
│   ├── gnn_model.py         # GraphSAGE encoder
│   ├── fusion_model.py      # Concat & Cross-Attention fusion
│   ├── contrastive.py       # Dual-encoder InfoNCE contrastive model
│   ├── train.py             # Training loops
│   └── evaluate.py          # Metric calculations (F1, AUC-PR, R@K)
├── config.yaml              # Hyperparameter configuration
└── requirements.txt         # Package dependencies 

## Setup & Demo
```bash
pip install -r requirements.txt
python -m src.evaluate
Run notebooks/demo_context.ipynb to verify end-to-end inference on preprocessed audio graphs.
