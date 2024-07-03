# OmniDrones

[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://omnidrones.readthedocs.io/en/latest/)

## How to generate md files from omnidrones

1. Activate conda environment including omnidrones

    ```bash
    conda activate sim
    ```

2. Generate md files

    ```bash
    python _generate_mds.py
    ```

## How to generate sphinx docs

1. Prepare python environment

    ```bash
    pip install -r requirements.txt
    ```

2. Build sphinx docs

    ```bash
    python -m sphinx -T source html
    ```

### Warnings (TODO)

1. [tips.rst](source/tutorials/tips.rst) isn't included in any toctree.
