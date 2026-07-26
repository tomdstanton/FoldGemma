# FoldGemma

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-active-success.svg)](#)

FoldGemma is an ultra-fast structural representation model based on the Gemma transformer architecture. It bridges the gap between protein language models and structural bioinformatics by rapidly translating raw amino acid sequences into structural 3Di tokens (Foldseek's structural alphabet). FoldGemma provides a pure PyTorch native pipeline for ultra-fast, high-throughput structural mapping.

## Background

Protein structure prediction is a computationally intensive task. While tools like AlphaFold offer extreme accuracy, they are too slow for large-scale metagenomic sweeps. FoldGemma takes a different approach: rather than predicting 3D coordinates, it translates 1D amino acid sequences into a 1D sequence of **3Di tokens**—the structural alphabet powering [Foldseek](https://github.com/steineggerlab/foldseek).

By formulating structure prediction as a Sequence-to-Sequence (Seq2Seq) problem and running inference on an ultra-fast PyTorch architecture, FoldGemma can "fold" millions of proteins in a fraction of the time, allowing you to instantly search metagenomes using Foldseek.

## Installation

You can install FoldGemma directly from PyPI using `uv` or `pip`:

```bash
uv pip install foldgemma
```

If you plan to train FoldGemma from scratch (e.g. using the massive AFDB dataset), install the optional training dependencies:

```bash
uv pip install foldgemma[train]
```

## CLI Usage

FoldGemma provides a robust CLI for inferencing, training, deploying, and data preparation.

### Inference

Convert a standard amino acid `.fasta` file into a 3Di `.fasta` file:

```bash
foldgemma infer -i proteins.fasta -o structures_3di.fasta --model-type foldgemma --weights ./model.safetensors
```

### Data Preparation (AFDB)

FoldGemma includes a native PyTorch IterableDataset pipeline that can stream and parse Foldcomp FCZ databases into TFRecords via background workers to maximize I/O throughput.

```bash
foldgemma prep --tsv ./afdb241_ss.tsv --fcz ./afdb241_fcz --out-dir ./tfrecords --num-workers 16
```

### Training

Start training on your generated TFRecords:

```bash
foldgemma train --data-dir ./tfrecords --epochs 10 --model-type foldgemma
```

### Deployment

Deploy a trained model straight to the Hugging Face Hub:

```bash
foldgemma deploy --repo-id <username>/foldgemma --model-path ./model.safetensors
```

## API Examples

FoldGemma exposes native PyTorch models (`FoldGemma` for fast classification mappings, and `FoldGemmaT5` for full generative decoder tasks).

### Basic Inference Example

```python
import torch
from foldgemma import FoldGemma
from foldgemma.config import FoldGemmaConfig, ModelType
from foldgemma.data.vocabulary import Protein3diVocabulary

# 1. Initialize configuration and model
config = FoldGemmaConfig(model_type=ModelType.FOLDGEMMA)
model = FoldGemma(config)

# (Optional) Load trained weights
# from safetensors.torch import load_file
# model.load_state_dict(load_file("model.safetensors"))

model.eval()

# 2. Your raw amino acid sequence as a byte-string
seq = b"MVLTIY"

# 3. Encode the byte-string using the vocabulary
vocab = Protein3diVocabulary()
input_ids = vocab.encode_bytes(seq)

# 4. Convert to a PyTorch tensor and add the batch dimension
input_tensor = torch.tensor([input_ids], dtype=torch.long)

# 5. Run the forward pass!
with torch.inference_mode():
    logits = model(input_tensor)
    
# 6. Extract the predicted 3Di tokens
out_ids = logits.argmax(dim=-1)[0].cpu().tolist()
predicted_3di = vocab.decode_bytes(out_ids)

print(predicted_3di)
```