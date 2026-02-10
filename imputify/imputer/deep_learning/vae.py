"""Variational AutoEncoder Imputer for tabular data."""
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
from .networks import VAENetwork
from .losses import elbo_loss


class VAEImputer(SklearnMixin, TabularDataMixin, DeviceMixin, BaseEstimator, TransformerMixin):
    """Variational AutoEncoder for tabular data imputation.

    VAE learns a probabilistic latent space instead of deterministic embeddings,
    enabling uncertainty quantification and sampling multiple imputations.

    The model maximizes the Evidence Lower Bound (ELBO):
    ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))

    Reference: Kingma & Welling, 2014 - "Auto-Encoding Variational Bayes"

    Parameters
    ----------
    hidden_dim : int, default=128
        Hidden layer dimension.
    latent_dim : int, default=32
        Latent space dimension.
    kl_weight : float, default=1.0
        Weight for KL divergence term (beta in beta-VAE).
    dropout : float, default=0.1
        Dropout rate.
    batch_size : int, default=64
        Training batch size.
    epochs : int, default=100
        Number of training epochs.
    lr : float, default=1e-3
        Learning rate.
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
    >>> from imputify import introduce_missing, VAEImputer
    >>> from sklearn.datasets import load_iris
    >>> import pandas as pd
    >>>
    >>> X = pd.DataFrame(load_iris().data, columns=['a','b','c','d'])
    >>> X_missing, mask = introduce_missing(X, proportion=0.3)
    >>>
    >>> imputer = VAEImputer(latent_dim=16, epochs=50)
    >>> imputer.fit(X_missing)
    >>> X_imputed = imputer.transform(X_missing)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        kl_weight: float = 1.0,
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
        self.kl_weight = kl_weight
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
        self._model: VAENetwork | None = None
        self._preprocessor: TabularPreprocessor | None = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any | None = None) -> VAEImputer:
        """Train the VAE on data with missing values.

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

        self._model = VAENetwork(
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

        pred_num, pred_cat, mu, logvar = self._model(X_num, X_cat_dict)

        loss = elbo_loss(
            pred_num=pred_num, target_num=X_num, mask_num=mask_num,
            pred_cat=pred_cat, target_cat=X_cat_dict, mask_cat=mask_cat_dict,
            mu=mu, logvar=logvar, kl_weight=self.kl_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Impute missing values using the trained VAE.

        Uses the deterministic mean (mu) rather than sampling for
        reproducible imputation.

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

            # Use mean (deterministic) rather than sampling for reproducible imputation
            mu, logvar = self._model.encode(X_num, X_cat)
            pred_num, pred_cat = self._model.decode(mu)

            predictions = {
                'num': pred_num.cpu().numpy(),
                'cat': {col: logits.argmax(dim=1).cpu().numpy() for col, logits in pred_cat.items()}
            }

        X_reconstructed = self._preprocessor.inverse_transform(predictions)
        X_imputed = X_df.copy()
        X_imputed[missing_mask] = X_reconstructed[missing_mask]

        return X_imputed.values if return_numpy else X_imputed
