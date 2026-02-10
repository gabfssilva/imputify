"""Denoising AutoEncoder Imputer for tabular data."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin

from imputify.shared import seed_everything
from imputify.shared_types import Device
from .mixins import DeviceMixin, TabularDataMixin, SklearnMixin
from .preprocessing import TabularPreprocessor
from .networks import DAENetwork
from .losses import heterogeneous_reconstruction_loss
from .utils import apply_swap_noise


class DAEImputer(SklearnMixin, TabularDataMixin, DeviceMixin, BaseEstimator, TransformerMixin):
    """Denoising AutoEncoder for tabular data imputation.

    Uses swap noise corruption to learn robust representations of tabular data.
    The model reconstructs clean data from corrupted input, learning feature
    dependencies rather than simple marginal statistics.

    The training objective minimizes reconstruction loss only at observed
    positions, using swap noise as corruption:

        x̃ = swap_noise(x, p)
        L = Σ_observed ||x - f(x̃)||²

    where p is the noise probability and f is the autoencoder. This forces
    the network to learn inter-feature dependencies rather than identity
    mapping.

    Reference: Vincent et al., 2008 - "Extracting and composing robust
    features with denoising autoencoders"

    Parameters
    ----------
    hidden_dim : int, default=128
        Hidden layer dimension in encoder/decoder.
    latent_dim : int, default=64
        Latent representation dimension.
    noise_level : float, default=0.15
        Swap noise probability.
    dropout : float, default=0.1
        Dropout rate for regularization.
    batch_size : int, default=64
        Training batch size.
    epochs : int, default=100
        Number of training epochs.
    lr : float, default=1e-3
        Learning rate for Adam optimizer.
    device : Device, default='auto'
        Device to use ('auto', 'cuda', 'mps', 'cpu').
    verbose : bool, default=True
        Print training progress.
    log_every : int, default=10
        Print training loss every N epochs (only if verbose=True).
    seed : int, optional
        Random seed for reproducibility.
    deterministic : bool, default=False
        Enable CUDA deterministic mode (may reduce performance).

    Examples
    --------
    >>> from imputify import introduce_missing, DAEImputer
    >>> from sklearn.datasets import load_iris
    >>> import pandas as pd
    >>>
    >>> X = pd.DataFrame(load_iris().data, columns=['a','b','c','d'])
    >>> X_missing, mask = introduce_missing(X, proportion=0.3)
    >>>
    >>> imputer = DAEImputer(epochs=50, noise_level=0.2)
    >>> imputer.fit(X_missing)
    >>> X_imputed = imputer.transform(X_missing)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        noise_level: float = 0.15,
        dropout: float = 0.1,
        batch_size: int = 64,
        epochs: int = 100,
        lr: float = 1e-3,
        device: Device = 'auto',
        verbose: bool = True,
        log_every: int = 10,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.noise_level = noise_level
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
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
        self._model: DAENetwork | None = None
        self._preprocessor: TabularPreprocessor | None = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any | None = None) -> DAEImputer:
        """Train the DAE on data with missing values.

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

        self._model = DAENetwork(
            num_features=self._preprocessor.num_features,
            embedding_info=self._preprocessor.embedding_info,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(self._device)

        X_processed = self._preprocessor.transform(X_df)
        missing_mask = ~X_df.isnull()
        dataloader = self._create_dataloader(X_processed, missing_mask, self.batch_size)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self._model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch in dataloader:
                loss = self._train_step(batch, optimizer)
                epoch_loss += loss
                num_batches += 1

            if self.verbose and (epoch + 1) % self.log_every == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss / max(num_batches, 1):.4f}")

        self._fitted = True
        return self

    def _train_step(self, batch, optimizer) -> float:
        X_num, X_cat, mask_num, mask_cat = [t.to(self._device) for t in batch]

        X_cat_dict = {col: X_cat[:, i].long() for i, col in enumerate(self._cat_cols)}
        mask_cat_dict = {col: mask_cat[:, i] for i, col in enumerate(self._cat_cols)}

        X_num_noisy, _ = apply_swap_noise(X_num, self.noise_level)
        pred_num, pred_cat = self._model(X_num_noisy, X_cat_dict)

        loss = heterogeneous_reconstruction_loss(
            pred_num=pred_num, target_num=X_num, mask_num=mask_num,
            pred_cat=pred_cat, target_cat=X_cat_dict, mask_cat=mask_cat_dict,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Impute missing values using the trained DAE.

        Args:
            X: Data with missing values to impute.

        Returns:
            Data with missing positions filled. Same format as input.
        """
        X_df, return_numpy = self._validate_transform_input(X)

        if not X_df.isnull().any().any():
            return X if return_numpy else X_df

        missing_mask = X_df.isnull()
        X_processed = self._preprocessor.transform(X_df)

        self._model.eval()
        with torch.no_grad():
            X_num = torch.FloatTensor(X_processed['num']).to(self._device)
            X_cat = {col: torch.LongTensor(X_processed['cat'][col]).to(self._device) for col in self._cat_cols}

            pred_num, pred_cat = self._model(X_num, X_cat)

            predictions = {
                'num': pred_num.cpu().numpy(),
                'cat': {col: logits.argmax(dim=1).cpu().numpy() for col, logits in pred_cat.items()}
            }

        X_reconstructed = self._preprocessor.inverse_transform(predictions)
        X_imputed = X_df.copy()
        X_imputed[missing_mask] = X_reconstructed[missing_mask]

        return X_imputed.values if return_numpy else X_imputed
