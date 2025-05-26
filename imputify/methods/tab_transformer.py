import numpy as np
import pandas as pd
from typing import List, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from keras import Model, Input, layers, optimizers

class TabTransformerImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        embedding_dim: int = 32,
        transformer_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 64,
        dropout_rate: float = 0.1,
        epochs: int = 10,
        batch_size: int = 32
    ):
        """Initialize the TabTransformer imputer.
        
        Args:
            embedding_dim: Dimension of the embedding vectors for categorical features.
            transformer_layers: Number of transformer layers.
            num_heads: Number of attention heads in each transformer layer.
            ff_dim: Dimension of the feed-forward network.
            dropout_rate: Dropout rate for regularization.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
        """
        self.categorical_features = []
        self.numerical_features = []
        self.num_categories = {}
        self.embedding_dim = embedding_dim
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model: Optional[Model] = None
        self.feature_order: Optional[List[str]] = None

    def _build_model(self, cat_dims: List[int], num_features: int) -> Model:
        inputs = []
        
        # Handle categorical features
        cat_inputs = []
        if self.categorical_features:
            cat_inputs = [
                Input(shape=(1,), name=f"{feat}_input", dtype="int32")
                for feat in self.categorical_features
            ]
            inputs.extend(cat_inputs)

        # Handle numerical features  
        num_input = None
        if num_features > 0:
            num_input = Input(shape=(num_features,), name="num_input")
            inputs.append(num_input)

        # Build categorical processing pipeline
        if self.categorical_features:
            embeddings = []
            for inp, dim in zip(cat_inputs, cat_dims):
                emb = layers.Embedding(input_dim=dim, output_dim=self.embedding_dim)(inp)
                embeddings.append(emb)

            cat_stack = layers.Concatenate(axis=1)(embeddings)
            x = cat_stack

            # Transformer blocks
            for _ in range(self.transformer_layers):
                attn_output = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embedding_dim)(x, x)
                x = layers.Add()([x, attn_output])
                x = layers.LayerNormalization()(x)

                ff_output = layers.Dense(self.ff_dim, activation="relu")(x)
                ff_output = layers.Dense(self.embedding_dim)(ff_output)
                x = layers.Add()([x, ff_output])
                x = layers.LayerNormalization()(x)

            cat_output = layers.Flatten()(x)

        # Combine or use individual outputs
        if self.categorical_features and num_features > 0:
            # Both categorical and numerical
            combined = layers.Concatenate()([cat_output, num_input])
            x = layers.Dense(self.ff_dim, activation="relu")(combined)
        elif self.categorical_features:
            # Only categorical
            x = layers.Dense(self.ff_dim, activation="relu")(cat_output)
        else:
            # Only numerical
            x = layers.Dense(self.ff_dim, activation="relu")(num_input)
            
        x = layers.Dropout(self.dropout_rate)(x)
        output = layers.Dense(len(self.categorical_features) + num_features)(x)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=optimizers.Adam(), loss="mse")
        return model

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None) -> "TabTransformerImputer":
        """Fit the TabTransformer imputer on the data.
        
        Args:
            X: Training data with missing values.
            y: Target values (optional, not used in imputation).
            
        Returns:
            self: The fitted imputer.
        """
        df = pd.DataFrame(X)
        self.feature_order = list(df.columns)

        self._infer_feature_types(df)

        df_cat = df[self.categorical_features].fillna(-1).astype(int)
        df_num = df[self.numerical_features].fillna(0).astype(float)

        if not self.num_categories:
            for col in self.categorical_features:
                self.num_categories[col] = int(df_cat[col].max()) + 2  # +2 for unseen/missing

        cat_dims = [self.num_categories[feat] for feat in self.categorical_features]

        self.model = self._build_model(cat_dims=cat_dims, num_features=len(self.numerical_features))

        X_inputs = []
        if self.categorical_features:
            X_cat = [np.array(df_cat[col]) for col in self.categorical_features]
            X_inputs.extend(X_cat)
        if len(self.numerical_features) > 0:
            X_num = df_num.to_numpy()
            X_inputs.append(X_num)

        # Build target
        target_parts = []
        if self.categorical_features:
            target_cat = df[self.categorical_features].fillna(0).astype(float)
            target_parts.append(target_cat.to_numpy())
        if self.numerical_features:
            target_num = df[self.numerical_features].fillna(0).astype(float)
            target_parts.append(target_num.to_numpy())
        
        y_target = np.concatenate(target_parts, axis=1)

        self.model.fit(X_inputs, y_target, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Transform data by imputing missing values.
        
        Args:
            X: Data with missing values to impute.
            
        Returns:
            DataFrame with imputed values.
        """
        df = pd.DataFrame(X)
        df_cat = df[self.categorical_features].fillna(-1).astype(int)
        df_num = df[self.numerical_features].fillna(0).astype(float)

        X_inputs = []
        if self.categorical_features:
            X_cat = [np.array(df_cat[col]) for col in self.categorical_features]
            X_inputs.extend(X_cat)
        if len(self.numerical_features) > 0:
            X_num = df_num.to_numpy()
            X_inputs.append(X_num)

        preds = self.model.predict(X_inputs, batch_size=self.batch_size, verbose=1)
        pred_cat = preds[:, :len(self.categorical_features)]
        pred_num = preds[:, len(self.categorical_features):]

        df_out = df.copy()

        for i, col in enumerate(self.categorical_features):
            mask = df_out[col].isna()
            df_out.loc[mask, col] = np.round(pred_cat[mask, i]).astype(int)

        for i, col in enumerate(self.numerical_features):
            mask = df_out[col].isna()
            df_out.loc[mask, col] = pred_num[mask, i]

        return df_out

    def _infer_feature_types(self, df: pd.DataFrame) -> None:
        inferred_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
        inferred_cat += [
            col for col in df.select_dtypes(include=["int"]).columns
            if df[col].nunique() < 20
        ]
        self.categorical_features = list(set(inferred_cat))

        self.numerical_features = [
             col for col in df.columns
            if col not in self.categorical_features
               and pd.api.types.is_numeric_dtype(df[col])
        ]
