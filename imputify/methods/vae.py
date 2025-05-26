"""Variational Autoencoder-based imputation methods."""

from typing import Union, Optional, List, Callable, TypedDict, NotRequired, Tuple, Literal, Dict, Any
import warnings

import numpy as np
import pandas as pd

try:
    import keras
    from keras import random, layers, ops as K
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    warnings.warn(
        "Keras not available. VAEImputer will not work. "
        "Please install keras and a Keras backend (jax, pytorch or tensorflow) to use VAEImputer."
    )

from ..core.base import BaseImputer

type Layer = Tuple[float, Callable | str] | Tuple[float, Callable | str, float]

class NoiseConfig(TypedDict, total=False):
    """Configuration for noise mechanisms during training."""
    type: Literal['gaussian', 'dropout', 'masking', 'none']
    intensity: float  # Noise intensity/probability (0.0 to 1.0)
    schedule: Optional[Literal['constant', 'decay', 'increase']]  # How noise changes over epochs
    decay_rate: Optional[float]  # Decay/increase rate for scheduled noise


def build_layers(dim: int, layers_config: List[Layer]):
    """Build sequential layers with given configuration.
    
    Args:
        dim: Base dimension for layer sizing.
        layers_config: List of layer configurations, each containing
            (ratio, activation, dropout_rate).
    
    Yields:
        keras.layers: Sequential layers for the neural network.
    """
    for layer_config in layers_config:
        ratio = layer_config[0]
        activation = layer_config[1] if len(layer_config) > 1 else keras.activations.relu
        dropout = layer_config[2] if len(layer_config) > 2 else 0.0
        
        # Handle string activation names
        if isinstance(activation, str):
            activation = getattr(keras.activations, activation)
        
        yield layers.Dense(max(1, int(dim * ratio)), activation=activation)
        if dropout > 0:
            yield layers.Dropout(dropout)


class VAEModel(keras.Model):
    """Variational Autoencoder model for imputation."""
    
    def __init__(self, input_dim: int, layers_config: List[Layer], latent_dim: int = 16, beta: float = 1.0, noise_std: float = 0.0, stochastic_inference: float = 0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.noise_std = noise_std
        self.stochastic_inference = stochastic_inference
        
        # Initialize seed generator for JAX compatibility
        self.seed_generator = keras.random.SeedGenerator(seed=42)
        
        # Build encoder
        self.encoder_layers = list(build_layers(input_dim, layers_config))
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        
        # Build decoder (reverse order)
        reversed_config = list(reversed(layers_config))
        self.decoder_layers = list(build_layers(input_dim, reversed_config))
        self.decoder_output = layers.Dense(input_dim)
    
    def encode(self, inputs):
        """Encode inputs to latent representation."""
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var, training=True, stochastic_scale=0.0):
        """Reparameterization trick with controllable stochastic inference."""
        if training or stochastic_scale > 0:
            epsilon = random.normal(shape=K.shape(z_mean), seed=self.seed_generator)
            if training:
                return z_mean + K.exp(0.5 * z_log_var) * epsilon
            else:
                # Scale the stochasticity during inference
                return z_mean + K.exp(0.5 * z_log_var) * epsilon * stochastic_scale
        else:
            return z_mean
    
    def decode(self, z):
        """Decode latent representation to reconstruction."""
        x = z
        for layer in self.decoder_layers:
            x = layer(x)
        return self.decoder_output(x)
    
    def call(self, inputs, training=None):
        x, mask = inputs
        
        # Apply noise during training
        if training and self.noise_std > 0:
            noise = keras.random.normal(shape=K.shape(x), seed=self.seed_generator) * self.noise_std
            x_noisy = x + mask * noise  # Only add noise to observed values
        else:
            x_noisy = x
        
        # Encode
        z_mean, z_log_var = self.encode(x_noisy)
        # Only apply stochastic inference during inference (not training)
        stochastic_scale = 0.0 if training else self.stochastic_inference
        z = self.reparameterize(z_mean, z_log_var, training, stochastic_scale)
        
        # Decode
        reconstruction = self.decode(z)
        
        # For inference, blend deterministic and stochastic reconstructions based on mask
        if not training and self.stochastic_inference > 0:
            # Get deterministic reconstruction
            z_deterministic = self.reparameterize(z_mean, z_log_var, training=False, stochastic_scale=0.0)
            reconstruction_deterministic = self.decode(z_deterministic)
            
            # Use deterministic for observed (mask=1), stochastic for missing (mask=0)
            reconstruction = mask * reconstruction_deterministic + (1 - mask) * reconstruction
        
        # Compute losses (for both training and validation)
        # Masked reconstruction loss (only observed values)
        recon_loss = K.square(x - reconstruction)
        masked_recon_loss = K.sum(mask * recon_loss, axis=1)
        
        # KL divergence loss
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        
        # Total loss
        total_loss = masked_recon_loss + self.beta * kl_loss
        self.add_loss(K.mean(total_loss))
        
        return reconstruction


class VAEImputer:
    """Variational Autoencoder-based imputer for missing data.
    
    Parameters
    ----------
    layers_config : List[Layer]
        Layer configuration for encoder/decoder architecture.
    latent_dim : int, default=16
        Dimensionality of the VAE latent space.
    epochs : int, default=50
        Number of training epochs.
    batch_size : int, default=32
        Batch size for training.
    learning_rate : float, default=0.001
        Learning rate for optimizer.
    beta : float, default=1.0
        Weight for KL divergence term.
    early_stopping : bool, default=False
        Whether to use early stopping.
    patience : int, default=10
        Patience for early stopping.
    validation_split : float, default=0.1
        Fraction of data for validation.
    noise_std : float, default=0.0
        Standard deviation of Gaussian noise added during training.
    stochastic_inference : float, default=0.0
        Scale factor for stochastic sampling during inference (0.0=deterministic, 1.0=full stochastic).
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : int, default=0
        Training verbosity level.
    """
    
    def __init__(
        self,
        layers_config: List[Layer],
        latent_dim: int = 16,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        beta: float = 1.0,
        early_stopping: bool = True,
        patience: int = 10,
        validation_split: float = 0.2,
        noise_std: float = 0.2,
        stochastic_inference: float = 0.0,
        random_state: Optional[int] = None,
        verbose: int = 0,
        **kwargs
    ):
        if not KERAS_AVAILABLE:
            raise ImportError("Keras not available. Please install keras to use VAEImputer.")
            
        super().__init__(**kwargs)
        
        # Default architecture: 80% and 50% of input dimensions
        if not layers_config:
            layers_config = [
                (0.8, keras.activations.relu),
                (0.5, keras.activations.relu)
            ]
        
        self.layers_config = list(layers_config)
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_split = validation_split
        self.noise_std = noise_std
        self.stochastic_inference = stochastic_inference
        self.random_state = random_state
        self.verbose = verbose
        
        # Model components
        self.model_ = None
        self.input_dim_ = None
        self.feature_means_ = None
        
        # Normalization parameters
        self.feature_means_norm_ = None
        self.feature_stds_norm_ = None
        self.fitted_normalization_ = False
        
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame], fit_normalization: bool = False) -> tuple:
        """Prepare data for VAE training/inference."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        
        # Create mask: 1 for observed, 0 for missing
        mask = (~np.isnan(X)).astype(np.float32)
        
        # Fit normalization parameters if requested
        if fit_normalization:
            self._fit_normalization(X)
        
        # Fill missing values with means
        X_filled = X.copy()
        if self.feature_means_ is None:
            col_means = np.nanmean(X, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            self.feature_means_ = col_means
        else:
            col_means = self.feature_means_
        
        nan_mask = np.isnan(X_filled)
        X_filled[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        
        # Normalize the filled data
        if self.fitted_normalization_:
            X_filled_norm = self._normalize_data(X_filled)
        else:
            raise ValueError("Normalization not fitted.")
        
        return X_filled_norm, mask
    
    def _fit_normalization(self, X: np.ndarray) -> None:
        """Fit normalization parameters."""
        self.feature_means_norm_ = np.nanmean(X, axis=0)
        self.feature_stds_norm_ = np.nanstd(X, axis=0)
        
        # Handle constant features and missing columns
        self.feature_stds_norm_ = np.where(self.feature_stds_norm_ == 0, 1.0, self.feature_stds_norm_)
        self.feature_means_norm_ = np.where(np.isnan(self.feature_means_norm_), 0.0, self.feature_means_norm_)
        
        self.fitted_normalization_ = True
    
    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """Normalize data using fitted parameters."""
        return (X - self.feature_means_norm_) / self.feature_stds_norm_
    
    def _denormalize_data(self, X_norm: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale."""
        return X_norm * self.feature_stds_norm_ + self.feature_means_norm_
    
    def _build_model(self, input_dim: int) -> None:
        """Build the VAE model architecture."""
        self.model_ = VAEModel(
            input_dim=input_dim,
            layers_config=self.layers_config,
            latent_dim=self.latent_dim,
            beta=self.beta,
            noise_std=self.noise_std,
            stochastic_inference=self.stochastic_inference
        )
        
        # Compile model
        self.model_.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None, **kwargs) -> "VAEImputer":
        """Fit the VAE imputer."""
        if self.random_state is not None:
            keras.utils.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
        
        X_filled_norm, mask = self._prepare_data(X, fit_normalization=True)
        n_samples, n_features = X_filled_norm.shape
        self.input_dim_ = n_features
        
        # Build model
        self._build_model(n_features)
        
        # Prepare callbacks
        callbacks = []
        if self.early_stopping:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=self.patience, restore_best_weights=True
            ))
        
        # Train the VAE
        self.model_.fit(
            [X_filled_norm, mask],
            X_filled_norm,  # Dummy target
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split if self.early_stopping else 0.0,
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        self.fitted = True
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """Impute missing values using the trained VAE."""
        if not self.fitted:
            raise ValueError("Imputer must be fitted before transform.")
        
        # Store original format
        is_dataframe = isinstance(X, pd.DataFrame)
        original_columns = X.columns if is_dataframe else None
        original_index = X.index if is_dataframe else None
        X_original = X.copy() if is_dataframe else X.copy()
        
        # Prepare normalized data
        X_filled_norm, mask = self._prepare_data(X, fit_normalization=False)
        
        # Apply stochastic inference if enabled
        reconstruction_norm = self.model_.predict([X_filled_norm, mask], verbose=0)
            
        reconstruction = self._denormalize_data(reconstruction_norm)
        
        # Create output
        X_output = X_original.values.copy() if is_dataframe else X_original.copy()
        missing_mask = np.isnan(X_output)
        X_output[missing_mask] = reconstruction[missing_mask]
        
        # Return in original format
        if is_dataframe:
            return pd.DataFrame(X_output, columns=original_columns, index=original_index)
        return X_output

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """Fit the imputer and impute missing values.

        Args:
            X: Data with missing values to fit and impute.
            **kwargs: Additional parameters for fitting and transformation.

        Returns:
            Union[np.ndarray, pd.DataFrame]: Data with imputed values.
        """
        return self.fit(X, **kwargs).transform(X, **kwargs)

    def get_params(self) -> Dict[str, Any]:
        """Get the parameters of the imputer.

        Returns:
            Dict[str, Any]: Parameters of the imputer.
        """
        return {
            'layers_config': self.layers_config,
            'latent_dim': self.latent_dim,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'beta': self.beta,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'validation_split': self.validation_split,
            'noise_std': self.noise_std,
            'stochastic_inference': self.stochastic_inference,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params) -> "VAEImputer":
        """Set the parameters of the imputer.

        Args:
            **params: Parameters to set.

        Returns:
            self: The imputer with updated parameters.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self