"""Loss functions for heterogeneous tabular data imputation.

This module provides masked loss functions that only compute loss on observed
values, which is essential for training imputation models. The network should
not be penalized for failing to reconstruct placeholder values that were
initially filled in missing positions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

LOSS_EPSILON = 1e-8
"""Small value to prevent division by zero in masked loss reduction."""


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked Mean Squared Error loss for numerical features.

    Computes MSE only on positions where mask==1 (observed values).
    This prevents the model from being penalized for reconstructing
    placeholder values that were filled in for missing data.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values, shape (batch_size, num_features)
    target : torch.Tensor
        Target values, shape (batch_size, num_features)
    mask : torch.Tensor
        Binary mask (1=observed, 0=missing), shape (batch_size, num_features)

    Returns
    -------
    torch.Tensor
        Scalar loss value
    """
    squared_error = (pred - target) ** 2
    masked_error = squared_error * mask
    return masked_error.sum() / (mask.sum() + LOSS_EPSILON)


def masked_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked Cross Entropy loss for categorical features.

    Computes cross entropy only on positions where mask==1 (observed values).

    Parameters
    ----------
    logits : torch.Tensor
        Raw prediction logits, shape (batch_size, num_classes)
    target : torch.Tensor
        Target class indices, shape (batch_size,)
    mask : torch.Tensor
        Binary mask (1=observed, 0=missing), shape (batch_size,)

    Returns
    -------
    torch.Tensor
        Scalar loss value
    """
    ce_loss = F.cross_entropy(logits, target.long(), reduction='none')
    masked_loss = ce_loss * mask
    return masked_loss.sum() / (mask.sum() + LOSS_EPSILON)


def heterogeneous_reconstruction_loss(
    pred_num: torch.Tensor,
    target_num: torch.Tensor,
    mask_num: torch.Tensor,
    pred_cat: dict[str, torch.Tensor],
    target_cat: dict[str, torch.Tensor],
    mask_cat: dict[str, torch.Tensor],
    num_weight: float = 1.0,
    cat_weight: float = 1.0,
) -> torch.Tensor:
    """Combined reconstruction loss for numerical and categorical features.

    This is the primary loss function for heterogeneous tabular imputation.
    It combines MSE loss for continuous features and cross entropy loss
    for discrete features, with proper masking for both.

    Parameters
    ----------
    pred_num : torch.Tensor
        Predicted numerical values, shape (batch_size, num_numerical_features)
    target_num : torch.Tensor
        Target numerical values, shape (batch_size, num_numerical_features)
    mask_num : torch.Tensor
        Mask for numerical features, shape (batch_size, num_numerical_features)
    pred_cat : dict[str, torch.Tensor]
        Predicted categorical logits, one tensor per column
        Each tensor shape: (batch_size, num_classes_for_column)
    target_cat : dict[str, torch.Tensor]
        Target categorical indices, one tensor per column
        Each tensor shape: (batch_size,)
    mask_cat : dict[str, torch.Tensor]
        Masks for categorical features, one tensor per column
        Each tensor shape: (batch_size,)
    num_weight : float, default=1.0
        Weight for numerical loss
    cat_weight : float, default=1.0
        Weight for categorical loss

    Returns
    -------
    torch.Tensor
        Combined scalar loss value
    """
    if mask_num.sum() > 0:
        loss_num = masked_mse_loss(pred_num, target_num, mask_num)
    else:
        loss_num = torch.tensor(0.0, device=pred_num.device)

    loss_cat = torch.tensor(0.0, device=pred_num.device)
    num_cat_cols = len(pred_cat)

    if num_cat_cols > 0:
        for col in pred_cat:
            if col in target_cat and col in mask_cat:
                col_loss = masked_cross_entropy_loss(
                    pred_cat[col],
                    target_cat[col],
                    mask_cat[col],
                )
                loss_cat = loss_cat + col_loss

        loss_cat = loss_cat / max(num_cat_cols, 1)

    total_loss = num_weight * loss_num + cat_weight * loss_cat

    return total_loss


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence loss for Variational AutoEncoder.

    Computes the KL divergence between the learned latent distribution
    q(z|x) ~ N(mu, sigma^2) and the prior p(z) ~ N(0, I).

    Analytical formula:
    KL(q||p) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Parameters
    ----------
    mu : torch.Tensor
        Mean of latent distribution, shape (batch_size, latent_dim)
    logvar : torch.Tensor
        Log variance of latent distribution, shape (batch_size, latent_dim)

    Returns
    -------
    torch.Tensor
        Scalar KL divergence averaged over batch
    """
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl_div.mean()


def elbo_loss(
    pred_num: torch.Tensor,
    target_num: torch.Tensor,
    mask_num: torch.Tensor,
    pred_cat: dict[str, torch.Tensor],
    target_cat: dict[str, torch.Tensor],
    mask_cat: dict[str, torch.Tensor],
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
) -> torch.Tensor:
    """Evidence Lower Bound (ELBO) loss for VAE imputation.

    ELBO = Reconstruction Loss + KL_weight * KL Divergence

    Parameters
    ----------
    pred_num : torch.Tensor
        Predicted numerical values
    target_num : torch.Tensor
        Target numerical values
    mask_num : torch.Tensor
        Mask for numerical features
    pred_cat : dict[str, torch.Tensor]
        Predicted categorical logits
    target_cat : dict[str, torch.Tensor]
        Target categorical indices
    mask_cat : dict[str, torch.Tensor]
        Masks for categorical features
    mu : torch.Tensor
        Latent mean
    logvar : torch.Tensor
        Latent log variance
    kl_weight : float, default=1.0
        Weight for KL divergence term (beta in β-VAE)

    Returns
    -------
    torch.Tensor
        Scalar ELBO loss (to be minimized)
    """
    recon_loss = heterogeneous_reconstruction_loss(
        pred_num, target_num, mask_num,
        pred_cat, target_cat, mask_cat,
    )
    kl_loss = kl_divergence_loss(mu, logvar)

    return recon_loss + kl_weight * kl_loss
