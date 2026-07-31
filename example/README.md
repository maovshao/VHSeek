# VHSeek example

This directory contains the example sequences, precomputed embeddings, model
checkpoints, taxonomy resources, and expected outputs used by
[`pipeline.ipynb`](../pipeline.ipynb).

## Tested environment

The review workflow was tested with the following software environment:

- Ubuntu 22.04.4 LTS
- Python 3.12.13
- PyTorch 2.12.0+cu130
- CUDA 13.0
- NVIDIA driver 580.95.05

The documented pod configuration used an Intel Xeon Platinum 8481C CPU at
2.70 GHz (208 CPU cores), 1.8 TiB RAM, and eight NVIDIA H100 80GB HBM3 GPUs.
A GPU is recommended for efficient AIDO.DNA and ESM2 embedding generation,
but the required compute and memory depend on the number and length of the
input sequences.

## Installation

From the repository root:

```bash
conda create -y -n vhseek python=3.12
conda activate vhseek
bash requirements.sh
```

The typical installation time is approximately 20-30 minutes, depending on
network speed and hardware.

## Quick-start demo

1. Start Jupyter from the repository root.
2. Open [`pipeline.ipynb`](../pipeline.ipynb).
3. Run the cells in the `Quick Start` section in order.

The quick-start workflow uses the provided combined virus embeddings, VHSeek
checkpoint, label index, and taxonomy resources. It writes:

```text
example/result/quick_start/vhseek_probability
example/result/quick_start/vhseek_label_transfer
```

The verified Code Ocean quick-start run completed in approximately 12 minutes;
runtime varies with the available hardware.

## Custom sequence inputs

The `Custom Sequence Data` section of `pipeline.ipynb` demonstrates embedding
generation and prediction from user-provided sequences:

- The DNA branch accepts a viral genome FASTA file with one nucleotide
  sequence per viral genome.
- The protein branch accepts a protein FASTA file with one or more amino-acid
  sequences per virus. Each header should contain a protein identifier followed
  by the corresponding virus identifier.
- The full branch requires genome- and protein-derived embeddings with exactly
  matching virus identifiers.

The example input files are:

```text
example/data/sequence/virus_genome.fasta
example/data/sequence/virus_protein.fasta
```

## Code Ocean

A verified peer-review capsule with the preconfigured quick-start demo is
available at:

https://codeocean.com/capsule/0688927/tree
