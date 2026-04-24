# Kernel-based NPLM Test

A Python implementation of the New Physics Learning Machine (NPLM) test statistic using kernel methods via LogisticFalkon. 

## Overview

The NPLM test is a powerful nonparametric hypothesis test for comparing two distributions. This implementation uses:
- **LogisticFalkon**: A fast kernel method for classification (from the Falkon library)
- **Gaussian kernels**: For flexible distribution representation
- **Configurable hyperparameters**: Kernel bandwidth, regularization, and solver options

## Project Structure

```
kernel-nplm/
├── nplm/                          # Main package
│   ├── __init__.py
│   ├── logfalkon_nplm.py         # Core NPLM implementation
│   └── plotting.py               # Visualization utilities
├── data/                          # Data utilities
│   ├── __init__.py
│   ├── datasets.py               # Data handling (pooling samples)
│   ├── preprocessing.py          # Data preprocessing
│   └── synthetic.py              # Synthetic data generation
├── examples/                      # Example scripts
│   ├── __init__.py
│   └── 1DGaussian.py            # 1D Gaussian example
└── readme.md                      # This file
```

### Dependencies
- `numpy`: Numerical computations
- `scipy`: Distance calculations and utilities
- `torch`: GPU support (optional)
- `falkon`: Kernel methods library (LogisticFalkon)


### Example: 1D Gaussian Test

```bash
python -m examples/1DGaussian
```

This example:
- Generates reference samples from an exponential distribution
- Generates data samples from an exponetial distribution with a Gaussian signal in the tail
- Computes NPLM test statistics under null and alternative hypotheses
- Visualizes the resulting distributions


## References

Main reference: [Learning New Physics Efficiently with Nonparametric Methods](https://arxiv.org/abs/2204.02317)

For more information about Falkon, see: https://github.com/FalkonML/falkon

## Author

Marco Letizia