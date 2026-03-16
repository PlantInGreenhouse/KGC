# Connectivity-Aware Knowledge Graph Construction for Complex Sentences

Official implementation of the paper **"Connectivity-Aware Knowledge Graph Construction for Complex Sentences"**.

This repository provides the research code for a schema-aligned knowledge graph construction pipeline designed for complex sentences. The framework decomposes an input sentence into atomic facts, models inter-fact connectivity to alleviate decomposition-induced contextual fragmentation, extracts candidate triples, and normalizes open-vocabulary relation expressions to a predefined target schema. The current implementation supports experiments on **REBEL**, **Wiki-NRE**, and **WebNLG**.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/Task-Knowledge%20Graph%20Construction-green" />
  <img src="https://img.shields.io/badge/Status-Research%20Code-lightgrey" />
</p>

## Repository Overview

```text
.
├── datasets/              # input datasets
├── evaluate/              # evaluation scripts and references
├── few_shot_examples/     # dataset-specific few-shot examples
├── oie/                   # main implementation
├── outputs/               # final outputs
├── preprocess_outputs/    # intermediate results
├── prompt_templates/      # prompt templates
├── schemas/               # target relation schemas
├── environment.yml        # conda environment
├── run.py                 # python entry point
└── run.sh                 # main execution script
```

## Pipeline Overview

<p align="center">
  <img src="figs/pipeline.png" width="95%">
</p>


## Reproducibility

The full pipeline can be reproduced by creating the conda environment from `environment.yml` and running `run.sh`.

```bash
conda env create -f environment.yml
conda activate <env_name>
bash run.sh
```

The implementation is organized under `oie/`, with dataset files in `datasets/`, prompt templates in `prompt_templates/`, target schemas in `schemas/`, few-shot examples in `few_shot_examples/`, intermediate preprocessing results in `preprocess_outputs/`, final extracted and normalized triples in `outputs/`, and evaluation code in `evaluate/`.


```bibtex
@article{yourpaper2026,
  title={Connectivity-Aware Knowledge Graph Construction for Complex Sentences},
  author={},
  journal={Expert Systems with Applications},
  year={2026}
}
```