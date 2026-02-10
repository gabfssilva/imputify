"""Neural network architectures for tabular data imputation."""
from __future__ import annotations

import torch
import torch.nn as nn


class DAENetwork(nn.Module):
    """Denoising AutoEncoder network for tabular imputation.

    Architecture:
    - Embeddings for categorical features
    - Encoder: input → hidden → latent
    - Decoder: latent → hidden → output
    - Separate output heads for numerical (linear) and categorical (softmax)

    Parameters
    ----------
    num_features : int
        Number of numerical features
    embedding_info : dict[str, tuple[int, int]]
        Categorical embedding specifications: {col: (vocab_size, embedding_dim)}
    hidden_dim : int, default=128
        Hidden layer dimension
    latent_dim : int, default=64
        Latent representation dimension
    dropout : float, default=0.1
        Dropout probability
    """

    def __init__(
        self,
        num_features: int,
        embedding_info: dict[str, tuple[int, int]],
        hidden_dim: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.cat_cols = list(embedding_info.keys())
        self.embedding_info = embedding_info

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, emb_dim)
            for col, (vocab_size, emb_dim) in embedding_info.items()
        })

        emb_total = sum(emb_dim for _, emb_dim in embedding_info.values())
        input_dim = num_features + emb_total

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.num_head = nn.Linear(input_dim, num_features)

        self.cat_heads = nn.ModuleDict({
            col: nn.Linear(input_dim, vocab_size)
            for col, (vocab_size, _) in embedding_info.items()
        })

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass through the autoencoder.

        Parameters
        ----------
        x_num : torch.Tensor
            Numerical features, shape (batch_size, num_features)
        x_cat : dict[str, torch.Tensor]
            Categorical features as indices, {col: tensor of shape (batch_size,)}

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            - pred_num: Reconstructed numerical features
            - pred_cat: Logits for categorical features (dict of tensors)
        """
        cat_embeds = [self.embeddings[col](x_cat[col]) for col in self.cat_cols]

        if cat_embeds:
            x_concat = torch.cat([x_num] + cat_embeds, dim=1)
        else:
            x_concat = x_num

        z = self.encoder(x_concat)
        h = self.decoder(z)

        pred_num = self.num_head(h)
        pred_cat = {col: self.cat_heads[col](h) for col in self.cat_cols}

        return pred_num, pred_cat


class VAENetwork(nn.Module):
    """Variational AutoEncoder network for tabular imputation.

    Extends DAE with probabilistic latent space:
    - Encoder outputs mu and logvar (parameters of latent distribution)
    - Reparameterization trick for differentiable sampling
    - Decoder reconstructs from sampled latent vector

    Parameters
    ----------
    num_features : int
        Number of numerical features
    embedding_info : dict[str, tuple[int, int]]
        Categorical embedding specifications
    hidden_dim : int, default=128
        Hidden layer dimension
    latent_dim : int, default=32
        Latent space dimension (typically smaller than DAE)
    dropout : float, default=0.1
        Dropout probability
    """

    def __init__(
        self,
        num_features: int,
        embedding_info: dict[str, tuple[int, int]],
        hidden_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.cat_cols = list(embedding_info.keys())
        self.latent_dim = latent_dim

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, emb_dim)
            for col, (vocab_size, emb_dim) in embedding_info.items()
        })

        emb_total = sum(emb_dim for _, emb_dim in embedding_info.values())
        input_dim = num_features + emb_total

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.num_head = nn.Linear(input_dim, num_features)
        self.cat_heads = nn.ModuleDict({
            col: nn.Linear(input_dim, vocab_size)
            for col, (vocab_size, _) in embedding_info.items()
        })

    def encode(
        self,
        x_num: torch.Tensor,
        x_cat: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Parameters
        ----------
        x_num : torch.Tensor
            Numerical features
        x_cat : dict[str, torch.Tensor]
            Categorical features

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - mu: Mean of latent distribution
            - logvar: Log variance of latent distribution
        """
        cat_embeds = [self.embeddings[col](x_cat[col]) for col in self.cat_cols]

        if cat_embeds:
            x_concat = torch.cat([x_num] + cat_embeds, dim=1)
        else:
            x_concat = x_num

        h = self.encoder(x_concat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample z = mu + sigma * epsilon.

        Parameters
        ----------
        mu : torch.Tensor
            Mean of latent distribution
        logvar : torch.Tensor
            Log variance of latent distribution

        Returns
        -------
        torch.Tensor
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode latent vector to reconstructed output.

        Parameters
        ----------
        z : torch.Tensor
            Latent vector

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            - pred_num: Reconstructed numerical features
            - pred_cat: Logits for categorical features
        """
        h = self.decoder(z)
        pred_num = self.num_head(h)
        pred_cat = {col: self.cat_heads[col](h) for col in self.cat_cols}
        return pred_num, pred_cat

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Parameters
        ----------
        x_num : torch.Tensor
            Numerical features
        x_cat : dict[str, torch.Tensor]
            Categorical features

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]
            - pred_num: Reconstructed numerical features
            - pred_cat: Logits for categorical features
            - mu: Latent mean (for KL loss)
            - logvar: Latent log variance (for KL loss)
        """
        mu, logvar = self.encode(x_num, x_cat)
        z = self.reparameterize(mu, logvar)
        pred_num, pred_cat = self.decode(z)
        return pred_num, pred_cat, mu, logvar


class Generator(nn.Module):
    """GAIN Generator network.

    Generates imputed values conditioned on:
    - X_tilde: data with missing values filled with random noise
    - M: binary mask (1=observed, 0=missing)
    - Z: additional random noise

    Parameters
    ----------
    num_features : int
        Number of numerical features
    embedding_info : dict[str, tuple[int, int]]
        Categorical embedding specifications
    hidden_dim : int, default=128
        Hidden layer dimension
    """

    def __init__(
        self,
        num_features: int,
        embedding_info: dict[str, tuple[int, int]],
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_features = num_features
        self.cat_cols = list(embedding_info.keys())

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, emb_dim)
            for col, (vocab_size, emb_dim) in embedding_info.items()
        })

        emb_total = sum(emb_dim for _, emb_dim in embedding_info.values())
        num_cat_cols = len(embedding_info)
        input_dim = 3 * num_features + num_cat_cols + emb_total

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features + emb_total),
        )

        self.num_head = nn.Linear(num_features + emb_total, num_features)
        self.cat_heads = nn.ModuleDict({
            col: nn.Linear(num_features + emb_total, vocab_size)
            for col, (vocab_size, _) in embedding_info.items()
        })

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: dict[str, torch.Tensor],
        mask_num: torch.Tensor,
        mask_cat: dict[str, torch.Tensor],
        z_num: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Generate imputed values.

        Parameters
        ----------
        x_num : torch.Tensor
            Numerical features (with noise in missing positions)
        x_cat : dict[str, torch.Tensor]
            Categorical features (with placeholders in missing positions)
        mask_num : torch.Tensor
            Binary mask for numerical features
        mask_cat : dict[str, torch.Tensor]
            Binary masks for categorical features
        z_num : torch.Tensor
            Random noise tensor

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Generated numerical and categorical values
        """
        cat_embeds = [self.embeddings[col](x_cat[col]) for col in self.cat_cols]
        mask_cat_concat = torch.stack([mask_cat[col] for col in self.cat_cols], dim=1) if self.cat_cols else torch.empty(x_num.shape[0], 0, device=x_num.device)

        if cat_embeds:
            inputs = torch.cat([x_num] + cat_embeds + [mask_num, mask_cat_concat, z_num], dim=1)
        else:
            inputs = torch.cat([x_num, mask_num, z_num], dim=1)

        h = self.network(inputs)
        gen_num = self.num_head(h)
        gen_cat = {col: self.cat_heads[col](h) for col in self.cat_cols}

        return gen_num, gen_cat


class Discriminator(nn.Module):
    """GAIN Discriminator network.

    Discriminates between observed and imputed values, conditioned on hint vector.

    Parameters
    ----------
    num_features : int
        Number of numerical features
    embedding_info : dict[str, tuple[int, int]]
        Categorical embedding specifications
    hidden_dim : int, default=128
        Hidden layer dimension
    """

    def __init__(
        self,
        num_features: int,
        embedding_info: dict[str, tuple[int, int]],
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_features = num_features
        self.cat_cols = list(embedding_info.keys())
        self.num_cat_cols = len(embedding_info)

        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_size, emb_dim)
            for col, (vocab_size, emb_dim) in embedding_info.items()
        })

        emb_total = sum(emb_dim for _, emb_dim in embedding_info.values())
        input_dim = 2 * num_features + self.num_cat_cols + emb_total

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features + self.num_cat_cols),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: dict[str, torch.Tensor],
        hint_num: torch.Tensor,
        hint_cat: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Discriminate between observed and imputed values.

        Parameters
        ----------
        x_num : torch.Tensor
            Numerical features (mix of observed and imputed)
        x_cat : dict[str, torch.Tensor]
            Categorical features (mix of observed and imputed)
        hint_num : torch.Tensor
            Hint vector for numerical features
        hint_cat : dict[str, torch.Tensor]
            Hint vectors for categorical features

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Predicted probabilities of being observed (0-1)
        """
        cat_embeds = [self.embeddings[col](x_cat[col]) for col in self.cat_cols]
        hint_cat_concat = torch.stack([hint_cat[col] for col in self.cat_cols], dim=1) if self.cat_cols else torch.empty(x_num.shape[0], 0, device=x_num.device)

        if cat_embeds:
            inputs = torch.cat([x_num] + cat_embeds + [hint_num, hint_cat_concat], dim=1)
        else:
            inputs = torch.cat([x_num, hint_num], dim=1)

        output = self.network(inputs)

        d_num = output[:, :self.num_features]
        d_cat = {}
        for i, col in enumerate(self.cat_cols):
            d_cat[col] = output[:, self.num_features + i].unsqueeze(1)

        return d_num, d_cat
