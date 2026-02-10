"""Generative Adversarial Imputation Nets (GAIN) for tabular data."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin

from imputify.shared import seed_everything
from imputify.shared_types import Device
from .mixins import DeviceMixin, TabularDataMixin, SklearnMixin
from .preprocessing import TabularPreprocessor
from .networks import Generator, Discriminator
from .utils import sample_hint_vector

CLAMP_EPSILON = 1e-7
"""Clamping bound to prevent log(0) in adversarial loss."""


class GAINImputer(SklearnMixin, TabularDataMixin, DeviceMixin, BaseEstimator, TransformerMixin):
    """Generative Adversarial Imputation Nets for tabular data.

    GAIN frames imputation as a minimax game between Generator (fills missing
    values) and Discriminator (identifies observed vs imputed values).

    The generator minimizes:

        L_G = -E[(1-m) · log D(G(x,m,z))] + α · E[m · ||x - G(x,m,z)||²]

    The discriminator minimizes:

        L_D = -E[m · log D(x̂) + (1-m) · log(1 - D(x̂))]

    where m is the observation mask, z is random noise, and α weights
    the reconstruction term. A hint vector reveals ~90% of the true mask
    to the discriminator to solve identifiability.

    Reference: Yoon et al., 2018 - "GAIN: Missing Data Imputation using
    Generative Adversarial Nets"

    Parameters
    ----------
    hidden_dim : int, default=128
        Hidden layer dimension for both G and D.
    alpha : float, default=10.0
        Weight for reconstruction loss in generator.
    hint_rate : float, default=0.9
        Probability of revealing mask information to discriminator.
    g_lr : float, default=1e-4
        Learning rate for generator.
    d_lr : float, default=1e-4
        Learning rate for discriminator.
    batch_size : int, default=64
        Training batch size.
    epochs : int, default=100
        Number of training epochs.
    device : Device, default='auto'
        Device to use ('auto', 'cuda', 'mps', 'cpu').
    verbose : bool, default=True
        Print training progress.
    log_every : int, default=10
        Print training loss every N epochs (only if verbose=True).
    seed : int, optional
        Random seed for reproducibility.
    deterministic : bool, default=False
        Enable CUDA deterministic mode.

    Examples
    --------
    >>> from imputify import introduce_missing, GAINImputer
    >>> from sklearn.datasets import load_iris
    >>> import pandas as pd
    >>>
    >>> X = pd.DataFrame(load_iris().data, columns=['a','b','c','d'])
    >>> X_missing, mask = introduce_missing(X, proportion=0.3)
    >>>
    >>> imputer = GAINImputer(alpha=20, hint_rate=0.9, epochs=100)
    >>> imputer.fit(X_missing)
    >>> X_imputed = imputer.transform(X_missing)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        alpha: float = 10.0,
        hint_rate: float = 0.9,
        g_lr: float = 1e-4,
        d_lr: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 100,
        device: Device = 'auto',
        verbose: bool = True,
        log_every: int = 10,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.hint_rate = hint_rate
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.log_every = log_every
        self.seed = seed
        self.deterministic = deterministic

        self._fitted = False
        self._columns: list[str] = []
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []
        self._column_types: dict[str, str] = {}
        self._device = self._resolve_device(device)
        self._netG: nn.Module | None = None
        self._netD: nn.Module | None = None
        self._optG: torch.optim.Optimizer | None = None
        self._optD: torch.optim.Optimizer | None = None
        self._preprocessor: TabularPreprocessor | None = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any | None = None) -> GAINImputer:
        """Train the GAIN on data with missing values.

        Args:
            X: Input data containing NaN values.
            y: Ignored. Present for sklearn compatibility.

        Returns:
            The fitted imputer.
        """
        seed_everything(self.seed, self.deterministic)

        X_df = self._validate_fit_input(X)
        self._setup_columns(X_df)
        self._num_cols, self._cat_cols = self._split_columns(X_df)

        self._preprocessor = TabularPreprocessor()
        self._preprocessor.fit(X_df, self._num_cols, self._cat_cols)

        self._netG = Generator(
            num_features=self._preprocessor.num_features,
            embedding_info=self._preprocessor.embedding_info,
            hidden_dim=self.hidden_dim,
        ).to(self._device)

        self._netD = Discriminator(
            num_features=self._preprocessor.num_features,
            embedding_info=self._preprocessor.embedding_info,
            hidden_dim=self.hidden_dim,
        ).to(self._device)

        self._optG = torch.optim.Adam(self._netG.parameters(), lr=self.g_lr)
        self._optD = torch.optim.Adam(self._netD.parameters(), lr=self.d_lr)

        X_processed = self._preprocessor.transform(X_df)
        missing_mask = ~X_df.isnull()
        dataloader = self._create_dataloader(X_processed, missing_mask, self.batch_size)

        self._netG.train()
        self._netD.train()

        for epoch in range(self.epochs):
            epoch_loss_g = 0.0
            epoch_loss_d = 0.0
            num_batches = 0

            for batch in dataloader:
                loss_d, loss_g = self._train_step_adversarial(batch)
                epoch_loss_d += loss_d
                epoch_loss_g += loss_g
                num_batches += 1

            if self.verbose and (epoch + 1) % self.log_every == 0:
                avg_g = epoch_loss_g / max(num_batches, 1)
                avg_d = epoch_loss_d / max(num_batches, 1)
                print(f"Epoch {epoch + 1}/{self.epochs} - G Loss: {avg_g:.4f}, D Loss: {avg_d:.4f}")

        self._fitted = True
        return self

    def _train_step_adversarial(self, batch) -> tuple[float, float]:
        X_num, X_cat, mask_num, mask_cat = [t.to(self._device) for t in batch]

        X_cat_dict = {col: X_cat[:, i].long() for i, col in enumerate(self._cat_cols)}
        mask_cat_dict = {col: mask_cat[:, i] for i, col in enumerate(self._cat_cols)}

        Z_num = torch.randn_like(X_num)
        X_tilde_num = mask_num * X_num + (1 - mask_num) * Z_num

        H_num = sample_hint_vector(mask_num, self.hint_rate)
        H_cat_dict = {col: sample_hint_vector(mask_cat_dict[col], self.hint_rate) for col in self._cat_cols}

        self._optD.zero_grad()
        gen_num, gen_cat = self._netG(X_tilde_num, X_cat_dict, mask_num, mask_cat_dict, Z_num)

        X_hat_num = mask_num * X_num + (1 - mask_num) * gen_num
        X_hat_cat = {}
        for col in self._cat_cols:
            gen_indices = gen_cat[col].argmax(dim=1)
            X_hat_cat[col] = (mask_cat_dict[col] * X_cat_dict[col].float() +
                             (1 - mask_cat_dict[col]) * gen_indices.float()).long()

        X_hat_cat_detached = {col: t.detach() for col, t in X_hat_cat.items()}
        d_pred_num, d_pred_cat = self._netD(X_hat_num.detach(), X_hat_cat_detached, H_num, H_cat_dict)

        loss_d_num = F.binary_cross_entropy(d_pred_num, mask_num)
        loss_d_cat = 0.0
        for col in self._cat_cols:
            loss_d_cat += F.binary_cross_entropy(d_pred_cat[col], mask_cat_dict[col].unsqueeze(1))

        loss_d = loss_d_num + loss_d_cat / max(len(self._cat_cols), 1)
        loss_d.backward()
        self._optD.step()

        self._optG.zero_grad()
        gen_num, gen_cat = self._netG(X_tilde_num, X_cat_dict, mask_num, mask_cat_dict, Z_num)

        X_hat_num = mask_num * X_num + (1 - mask_num) * gen_num
        X_hat_cat = {}
        for col in self._cat_cols:
            gen_indices = gen_cat[col].argmax(dim=1)
            X_hat_cat[col] = (mask_cat_dict[col] * X_cat_dict[col].float() +
                             (1 - mask_cat_dict[col]) * gen_indices.float()).long()

        d_pred_num, d_pred_cat = self._netD(X_hat_num, X_hat_cat, H_num, H_cat_dict)

        d_pred_num_clamped = torch.clamp(d_pred_num, min=CLAMP_EPSILON, max=1 - CLAMP_EPSILON)
        loss_g_adv_num = -torch.mean((1 - mask_num) * torch.log(d_pred_num_clamped))
        loss_g_adv_cat = 0.0
        for col in self._cat_cols:
            d_pred_cat_clamped = torch.clamp(d_pred_cat[col], min=CLAMP_EPSILON, max=1 - CLAMP_EPSILON)
            loss_g_adv_cat += -torch.mean((1 - mask_cat_dict[col]).unsqueeze(1) * torch.log(d_pred_cat_clamped))
        loss_g_adv = loss_g_adv_num + loss_g_adv_cat / max(len(self._cat_cols), 1)

        loss_g_rec_num = torch.mean(mask_num * (X_num - gen_num) ** 2)
        loss_g_rec_cat = 0.0
        for col in self._cat_cols:
            ce_loss = F.cross_entropy(gen_cat[col], X_cat_dict[col], reduction='none')
            loss_g_rec_cat += torch.mean(mask_cat_dict[col] * ce_loss)
        loss_g_rec = loss_g_rec_num + loss_g_rec_cat / max(len(self._cat_cols), 1)

        loss_g = loss_g_adv + self.alpha * loss_g_rec
        loss_g.backward()
        self._optG.step()

        return loss_d.item(), loss_g.item()

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Impute missing values using the trained GAIN generator.

        Args:
            X: Data with missing values to impute.

        Returns:
            Data with missing positions filled. Same format as input.
        """
        X_df, return_numpy = self._validate_transform_input(X)

        if not X_df.isnull().any().any():
            return X if return_numpy else X_df

        missing_mask = X_df.isnull()
        observation_mask = ~missing_mask
        X_processed = self._preprocessor.transform(X_df)

        self._netG.eval()
        with torch.no_grad():
            X_num = torch.FloatTensor(X_processed['num']).to(self._device)
            X_cat = {col: torch.LongTensor(X_processed['cat'][col]).to(self._device) for col in self._cat_cols}

            mask_num = torch.FloatTensor(observation_mask[self._num_cols].values).to(self._device)
            mask_cat = {
                col: torch.FloatTensor(observation_mask[[col]].values.ravel()).to(self._device)
                for col in self._cat_cols
            }

            generator = torch.Generator(device=self._device)
            if self.seed is not None:
                generator.manual_seed(self.seed)
            Z_num = torch.randn(X_num.shape, generator=generator, device=self._device, dtype=X_num.dtype)

            gen_num, gen_cat = self._netG(X_num, X_cat, mask_num, mask_cat, Z_num)

            predictions = {
                'num': gen_num.cpu().numpy(),
                'cat': {col: logits.argmax(dim=1).cpu().numpy() for col, logits in gen_cat.items()}
            }

        X_reconstructed = self._preprocessor.inverse_transform(predictions)
        X_imputed = X_df.copy()
        X_imputed[missing_mask] = X_reconstructed[missing_mask]

        return X_imputed.values if return_numpy else X_imputed
