"""Transformer-based imputation methods."""

from typing import Union, Optional, List, Dict, Any, TypedDict
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

try:
    import keras
    from keras import layers, ops
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    warnings.warn(
        "Keras not available. TransformerImputer will not work. "
        "Please install keras and a Keras backend (jax, pytorch or tensorflow) to use TransformerImputer."
    )

from ..core.base import BaseImputer


@dataclass
class DataSpec:
    """Data specification for TabularTransformer."""
    
    def __init__(self, X: Union[np.ndarray, pd.DataFrame]):
        """Initialize data specification from input data.
        
        Args:
            X: Input data to analyze
        """
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns.tolist()
            self._infer_schema(X)
        else:
            self.columns = [f"feature_{i}" for i in range(X.shape[1])]
            # For numpy arrays, treat all as numerical
            self.numerical_features = self.columns
            self.categorical_features = []
            self.categorical_vocab_sizes = {}
        
        # Create feature index mapping
        self.feature_index_map = {feat: i for i, feat in enumerate(self.columns)}
        
        # Create feature tasks based on inferred types
        self._create_feature_tasks()
    
    def _infer_schema(self, X: pd.DataFrame):
        """Automatically infer numerical and categorical features."""
        # Detect numerical features
        self.numerical_features = X.select_dtypes(include=['number']).columns.tolist()
        
        # Detect categorical features
        self.categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Calculate vocabulary sizes for categorical features
        self.categorical_vocab_sizes = {}
        for feat in self.categorical_features:
            # Count unique values excluding NaN
            vocab_size = X[feat].nunique(dropna=True)
            self.categorical_vocab_sizes[feat] = vocab_size
    
    def _create_feature_tasks(self):
        """Create feature tasks based on inferred types."""
        self.feature_tasks = {}
        
        # All numerical features are regression tasks
        for feat in self.numerical_features:
            self.feature_tasks[feat] = {"type": "regression"}
        
        # Categorical features depend on vocabulary size
        for feat in self.categorical_features:
            vocab_size = self.categorical_vocab_sizes[feat]
            if vocab_size == 1:
                # Constant feature - treat as regression
                self.feature_tasks[feat] = {"type": "regression"}
            elif vocab_size == 2:
                # Binary classification
                self.feature_tasks[feat] = {"type": "binary"}
            else:
                # Multi-class classification
                self.feature_tasks[feat] = {"type": "multiclass", "classes": vocab_size}


class TransformerBlock(layers.Layer):
    """Transformer block for tabular data."""
    
    def __init__(self, d_model, num_heads, ff_dim, dropout):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.drop1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization()

        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model),
        ])
        self.drop2 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization()

    def call(self, x, training=False):
        attn_out = self.att(x, x, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x, training=training)
        return self.norm2(x + self.drop2(ffn_out, training=training))


class TabularTransformer(keras.Model):
    """Tabular Transformer model for imputation."""
    
    def __init__(self, spec, d_model=32, num_heads=4, ff_dim=64, dropout=0.1, noise_std=0.0):
        super().__init__()
        self.spec = spec
        self.num_features = spec.numerical_features
        self.cat_features = spec.categorical_features
        self.cat_vocabs = spec.categorical_vocab_sizes
        self.d_model = d_model
        self.noise_std = noise_std
        
        # Initialize seed generator for JAX compatibility
        self.seed_generator = keras.random.SeedGenerator(seed=42)

        # Input projections
        self.num_proj = {
            feat: layers.Dense(d_model) for feat in self.num_features
        }
        self.cat_embed = {
            feat: layers.Embedding(input_dim=self.cat_vocabs[feat] + 1, output_dim=d_model)
            for feat in self.cat_features
        }

        # Positional embedding
        self.pos_embedding = layers.Embedding(
            input_dim=len(self.num_features) + len(self.cat_features),
            output_dim=d_model
        )

        self.block = TransformerBlock(d_model, num_heads, ff_dim, dropout)

        # Output heads
        self.heads = {}
        for feat, task in spec.feature_tasks.items():
            if task["type"] == "regression":
                self.heads[feat] = layers.Dense(1)
            elif task["type"] == "binary":
                self.heads[feat] = layers.Dense(1, activation="sigmoid")
            else:
                self.heads[feat] = layers.Dense(task["classes"], activation="softmax")

    def call(self, inputs, training=False):
        x, mask = inputs
        
        # Apply noise during training
        if training and self.noise_std > 0:
            noise = keras.random.normal(shape=ops.shape(x), seed=self.seed_generator) * self.noise_std
            x_noisy = x + mask * noise  # Only add noise to observed values
        else:
            x_noisy = x
        
        tokens = []

        # Project numerical features
        for feat in self.num_features:
            idx = self.spec.feature_index_map[feat]
            feat_x = ops.expand_dims(x_noisy[:, idx], -1)
            tokens.append(self.num_proj[feat](feat_x))

        # Embed categorical features
        for feat in self.cat_features:
            idx = self.spec.feature_index_map[feat]
            # Categorical features are already encoded as integers
            feat_x = ops.cast(x_noisy[:, idx], 'int32')
            feat_x = ops.expand_dims(feat_x, -1)
            embedded = self.cat_embed[feat](feat_x)
            tokens.append(ops.squeeze(embedded, axis=1))

        # Stack + position
        x_seq = ops.stack(tokens, axis=1)
        pos_ids = ops.arange(start=0, stop=x_seq.shape[1])
        x_seq = x_seq + self.pos_embedding(pos_ids)

        # Transformer block
        x_out = self.block(x_seq, training=training)

        # Output per feature
        out = {}
        all_feats = self.num_features + self.cat_features
        for i, feat in enumerate(all_feats):
            out[feat] = self.heads[feat](x_out[:, i, :])

        return out


class TransformerImputer:
    """Transformer-based imputer for missing data.
    
    Parameters
    ----------
    d_model : int, default=32
        Dimensionality of the transformer model.
    num_heads : int, default=4
        Number of attention heads.
    ff_dim : int, default=64
        Dimensionality of the feed-forward network.
    dropout : float, default=0.1
        Dropout rate.
    epochs : int, default=50
        Number of training epochs.
    batch_size : int, default=32
        Batch size for training.
    learning_rate : float, default=0.001
        Learning rate for optimizer.
    early_stopping : bool, default=False
        Whether to use early stopping.
    patience : int, default=10
        Patience for early stopping.
    validation_split : float, default=0.1
        Fraction of data for validation.
    noise_std : float, default=0.0
        Standard deviation of Gaussian noise added during training.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : int, default=0
        Training verbosity level.
    """
    
    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 4,
        ff_dim: int = 64,
        dropout: float = 0.0,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping: bool = True,
        patience: int = 10,
        validation_split: float = 0.2,
        noise_std: float = 0.2,
        random_state: Optional[int] = None,
        verbose: int = 0,
        **kwargs
    ):
        if not KERAS_AVAILABLE:
            raise ImportError("Keras not available. Please install keras to use TransformerImputer.")
            
        super().__init__(**kwargs)
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_split = validation_split
        self.noise_std = noise_std
        self.random_state = random_state
        self.verbose = verbose
        
        # Model components
        self.model_ = None
        self.spec_ = None
        self.feature_means_ = None
        
        # Sklearn preprocessing components
        self.scaler_ = StandardScaler()
        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.fitted_preprocessing_ = False
        
        if self.random_state is not None:
            keras.utils.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame], fit_preprocessing: bool = False) -> tuple:
        """Prepare data for transformer training/inference."""
        # Keep original format info
        is_dataframe = isinstance(X, pd.DataFrame)
        if is_dataframe:
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        # Create mask: 1 for observed, 0 for missing
        mask = (~X_df.isna()).astype(np.float32).values
        
        # Fit preprocessing if requested
        if fit_preprocessing:
            self._fit_preprocessing(X_df)
        
        # Apply preprocessing
        X_processed = self._preprocess_data(X_df)
        
        return X_processed, mask
    
    def _fit_preprocessing(self, X_df: pd.DataFrame) -> None:
        """Fit sklearn preprocessing components."""
        # Fill missing values temporarily for fitting
        X_temp = X_df.copy()
        
        # Fill numerical features with 0
        if self.spec_.numerical_features:
            X_temp[self.spec_.numerical_features] = X_temp[self.spec_.numerical_features].fillna(0)
            self.scaler_.fit(X_temp[self.spec_.numerical_features])
        
        # Fill categorical features with "missing" and fit encoder
        if self.spec_.categorical_features:
            X_temp[self.spec_.categorical_features] = X_temp[self.spec_.categorical_features].astype(str).fillna("missing")
            self.encoder_.fit(X_temp[self.spec_.categorical_features])
        
        self.fitted_preprocessing_ = True
    
    def _preprocess_data(self, X_df: pd.DataFrame) -> np.ndarray:
        """Preprocess data using fitted sklearn components."""
        if not self.fitted_preprocessing_:
            raise ValueError("Preprocessing not fitted.")
        
        X_processed = X_df.copy()
        
        # Handle numerical features
        if self.spec_.numerical_features:
            # Fill missing with 0 before scaling
            X_processed[self.spec_.numerical_features] = X_processed[self.spec_.numerical_features].fillna(0)
            X_processed[self.spec_.numerical_features] = self.scaler_.transform(X_processed[self.spec_.numerical_features])
        
        # Handle categorical features  
        if self.spec_.categorical_features:
            # Fill missing with "missing" before encoding
            X_processed[self.spec_.categorical_features] = X_processed[self.spec_.categorical_features].astype(str).fillna("missing")
            X_processed[self.spec_.categorical_features] = self.encoder_.transform(X_processed[self.spec_.categorical_features])
        
        return X_processed.values.astype(np.float32)
    
    def _inverse_transform_data(self, X_processed: np.ndarray) -> np.ndarray:
        """Inverse transform processed data back to original scale."""
        X_df = pd.DataFrame(X_processed, columns=self.spec_.columns)
        
        # Inverse transform numerical features
        if self.spec_.numerical_features:
            X_df[self.spec_.numerical_features] = self.scaler_.inverse_transform(X_df[self.spec_.numerical_features])
        
        # Categorical features will be handled separately in transform method
        return X_df.values
    
    def _build_model(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """Build the transformer model architecture."""
        self.spec_ = DataSpec(X)
        
        self.model_ = TabularTransformer(
            spec=self.spec_,
            d_model=self.d_model,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            noise_std=self.noise_std
        )
        
        # Create loss functions for each feature
        losses = {}
        for feat, task in self.spec_.feature_tasks.items():
            if task["type"] == "regression":
                losses[feat] = keras.losses.mean_squared_error
            elif task["type"] == "binary":
                losses[feat] = keras.losses.binary_crossentropy
            else:  # multiclass
                losses[feat] = keras.losses.sparse_categorical_crossentropy
        
        # Compile model with explicit loss functions
        self.model_.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=losses
        )
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None, **kwargs) -> "TransformerImputer":
        """Fit the transformer imputer."""
        # Build model and prepare data
        self._build_model(X)
        X_processed, mask = self._prepare_data(X, fit_preprocessing=True)
        
        # Create targets and sample weights based on feature types
        targets = self._create_targets(X_processed)
        sample_weights = self._create_sample_weights(mask)
        
        # Prepare callbacks
        callbacks = []
        if self.early_stopping:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=self.patience, restore_best_weights=True
            ))
        
        # Train the transformer
        validation_split = self.validation_split if self.early_stopping else 0.0
        
        self.model_.fit(
            (X_processed, mask),
            targets,
            sample_weight=sample_weights,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=self.verbose
        )
        
        self.fitted = True
        return self
    
    def _create_sample_weights(self, mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Create sample weights based on missing value mask."""
        sample_weights = {}
        
        for j, feat in enumerate(self.spec_.columns):
            # Use mask as sample weight: 1 for observed, 0 for missing
            sample_weights[feat] = mask[:, j]
        
        return sample_weights
    
    def _create_targets(self, X_processed: np.ndarray) -> Dict[str, np.ndarray]:
        """Create appropriate targets for each feature type."""
        targets = {}
        
        for j, feat in enumerate(self.spec_.columns):
            task = self.spec_.feature_tasks[feat]
            
            if task["type"] == "regression":
                # For regression, target is the processed value itself
                targets[feat] = X_processed[:, j:j+1]
            elif task["type"] == "binary":
                # For binary, target is the encoded categorical value (0 or 1)
                targets[feat] = X_processed[:, j:j+1]
            else:  # multiclass
                # For multiclass, target is the encoded categorical value (integer)
                targets[feat] = X_processed[:, j].astype(np.int32)
        
        return targets

    def transform(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """Impute missing values using the trained transformer."""
        if not self.fitted:
            raise ValueError("Imputer must be fitted before transform.")
        
        is_dataframe = isinstance(X, pd.DataFrame)

        if is_dataframe:
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.spec_.columns)
        
        original_missing_mask = X_df.isna()
        X_processed, mask = self._prepare_data(X_df, fit_preprocessing=False)
        predictions = self.model_.predict((X_processed, mask), verbose=0)
        X_output = X_df.copy()
        
        for i, feat in enumerate(self.spec_.columns):
            feat_missing = original_missing_mask[feat]
            
            if not feat_missing.any():
                continue  # No missing values for this feature
            
            task = self.spec_.feature_tasks[feat]
            
            if task["type"] == "regression":
                pred_vals = predictions[feat][:, 0]
                if feat in self.spec_.numerical_features:
                    feat_idx = self.spec_.numerical_features.index(feat)
                    scale = self.scaler_.scale_[feat_idx]
                    mean = self.scaler_.mean_[feat_idx]
                    pred_vals = pred_vals * scale + mean
                
                X_output.loc[feat_missing, feat] = pred_vals[feat_missing]

            elif task["type"] == "binary":
                pred_probs = predictions[feat][:, 0]
                pred_binary = (pred_probs > 0.5).astype(int)
                feat_idx = self.spec_.categorical_features.index(feat)
                pred_categories = self.encoder_.categories_[feat_idx][pred_binary[feat_missing]]
                
                # Convert to original dtype
                if hasattr(X_output[feat], 'dtype') and hasattr(X_output[feat].dtype, 'categories'):
                    original_dtype = X_output[feat].dtype.categories.dtype
                    pred_categories = pred_categories.astype(original_dtype)
                
                X_output.loc[feat_missing, feat] = pred_categories

            else:  # multiclass
                pred_probs = predictions[feat]
                pred_classes = np.argmax(pred_probs, axis=1)
                feat_idx = self.spec_.categorical_features.index(feat)
                pred_categories = self.encoder_.categories_[feat_idx][pred_classes[feat_missing]]
                
                # Convert to original dtype
                if hasattr(X_output[feat], 'dtype') and hasattr(X_output[feat].dtype, 'categories'):
                    original_dtype = X_output[feat].dtype.categories.dtype
                    pred_categories = pred_categories.astype(original_dtype)
                
                X_output.loc[feat_missing, feat] = pred_categories
        
        if is_dataframe:
            return X_output
        else:
            return X_output.values

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
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'validation_split': self.validation_split,
            'noise_std': self.noise_std,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params) -> "TransformerImputer":
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