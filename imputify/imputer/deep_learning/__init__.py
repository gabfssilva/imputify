"""Deep learning imputers for tabular data.

This module provides three state-of-the-art neural network architectures
for imputing missing values in tabular datasets with mixed numerical and
categorical features:

- DAEImputer: Denoising AutoEncoder with swap noise
- VAEImputer: Variational AutoEncoder with probabilistic latent space
- GAINImputer: Generative Adversarial Imputation Nets

All imputers follow scikit-learn's API conventions and can be used in
pipelines, grid searches, and cross-validation.
"""
from .dae import DAEImputer
from .vae import VAEImputer
from .gain import GAINImputer

__all__ = ['DAEImputer', 'VAEImputer', 'GAINImputer']
