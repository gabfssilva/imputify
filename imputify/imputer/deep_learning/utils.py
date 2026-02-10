"""Utility functions for deep learning imputers.

This module provides specialized utilities for training deep learning
imputation models, including noise injection strategies and auxiliary
mechanisms for adversarial training.
"""
from __future__ import annotations

import torch

HINT_NO_INFO = 0.5
"""Value representing 'no information' in GAIN hint vectors (midpoint between 0=missing and 1=observed)."""


def apply_swap_noise(batch: torch.Tensor, noise_prob: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply swap noise: randomly exchange values between rows.

    Swap noise is a corruption strategy specifically designed for tabular data.
    Unlike Gaussian noise, swap noise preserves the marginal distribution of
    each feature, forcing the model to learn the joint distribution structure
    to distinguish corrupted from real values.

    For a feature x_j in row i, swap noise replaces it with x_j from a
    randomly selected row k with probability noise_prob. This is implemented
    efficiently using vectorized operations.

    Parameters
    ----------
    batch : torch.Tensor
        Input batch, shape (batch_size, num_features)
    noise_prob : float, default=0.15
        Probability of swapping each element

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - corrupted_batch: Batch with swap noise applied
        - noise_mask: Binary mask (1 where swap occurred, 0 otherwise)

    References
    ----------
    .. [1] Vincent et al., "Stacked Denoising Autoencoders: Learning Useful
           Representations in a Deep Network with a Local Denoising Criterion",
           JMLR 2010

    Examples
    --------
    >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> x_noisy, mask = apply_swap_noise(x, noise_prob=0.5)
    >>> # Some elements of x_noisy are swapped from other rows
    """
    batch_size, num_features = batch.shape

    perm_indices = torch.randperm(batch_size, device=batch.device)
    shuffled_batch = batch[perm_indices]

    swap_mask = torch.rand(batch_size, num_features, device=batch.device) < noise_prob

    corrupted = torch.where(swap_mask, shuffled_batch, batch)

    return corrupted, swap_mask.float()


def sample_hint_vector(mask: torch.Tensor, hint_rate: float = 0.9) -> torch.Tensor:
    """Sample hint vector for GAIN (Generative Adversarial Imputation Nets).

    The hint vector provides partial information about the missingness mask
    to the discriminator, solving an identifiability problem in adversarial
    imputation. Without hints, multiple imputation distributions could fool
    the discriminator equally well.

    Formula: H = B ⊙ M + 0.5 ⊙ (1 - B)
    where B ~ Bernoulli(hint_rate) and M is the observation mask.

    - If B[i,j] = 1: H[i,j] = M[i,j] (reveal true mask value)
    - If B[i,j] = 0: H[i,j] = 0.5 (no information)

    Parameters
    ----------
    mask : torch.Tensor
        Binary observation mask (1=observed, 0=missing), shape (batch_size, num_features)
    hint_rate : float, default=0.9
        Probability of revealing each mask element to discriminator

    Returns
    -------
    torch.Tensor
        Hint vector with values in {0, 0.5, 1}, same shape as mask

    References
    ----------
    .. [1] Yoon et al., "GAIN: Missing Data Imputation using Generative
           Adversarial Nets", ICML 2018

    Examples
    --------
    >>> mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    >>> hint = sample_hint_vector(mask, hint_rate=0.9)
    >>> # Most elements reveal mask, ~10% are 0.5
    """
    B = torch.bernoulli(torch.full_like(mask, hint_rate))
    H = B * mask + HINT_NO_INFO * (1 - B)

    return H
