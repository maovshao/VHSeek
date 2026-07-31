# VHSeek

This repository implements "VHSeek: Biology foundation model-based virus representations revealing virus host associations".

VHSeek combines `DNA and Protein Foundation Models` to build `Virus Embeddings`, and uses these embeddings to predict `Associations between Viruses and Hosts` for all types of viruses (more than phage) and all taxonomy levels of host (from Infraspecies to Phylum).

<div align=center><img src="example/figure/VHSeek.png" width="100%" height="100%" /></div>

## Quick links

* [Requirements](#requirements)
* [Data preparation](#data-preparation)
* [Reproduce all our experiments with only one file](#main)
* [Run VHSeek locally](#pipeline)

## Requirements

<span id="requirements"></span>

1. Create a virtual environment
`conda create -y -n vhseek python=3.12`
2. Activate the virtual environment
`conda activate vhseek`
3. Install the required packages in the virtual environment
`bash requirements.sh`

## Data preparation
<span id="data-preparation"></span>

Download the latest VHSeek data release from [Zenodo](https://doi.org/10.5281/zenodo.18718269). Extract `vhseek_data.zip` in the repository root so that the resulting `vhseek_data/` directory is available to `main.ipynb`.

## Reproduce all our experiments with only one file
<span id="main"></span>

- Reproduce all our experiments with good visualization by following the steps in [main.ipynb](main.ipynb)

**Notice: main.ipynb outputs are saved in** `vhseek_data/experiment/test_result/notebook/`.

## Run VHSeek locally
<span id="pipeline"></span>

- Run VHSeek locally by following the example in [pipeline.ipynb](pipeline.ipynb)
- See the [example documentation](example/README.md) for the tested environment, quick-start steps, expected outputs, and input requirements.

**Notice: the inputs and outputs of the example are saved in** `example/`.
