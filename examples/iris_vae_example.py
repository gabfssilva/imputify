import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from imputify.datasets import load_iris
from imputify.methods import VAEImputer

np.random.seed(42)

X_missing, X_complete, y = load_iris(
    amputation={
        'prop': 0.5,
        'seed': 42,
        'patterns': [{
            'incomplete_vars': [0, 1, 2, 3],
            'mechanism': 'MCAR',
        }]
    }
)

print("Original dataset shape:", X_complete.shape)
print("Missing data percentage:", (X_missing.isna().sum().sum() / X_missing.size) * 100, "%")
print("Missing values per feature:")
print(X_missing.isna().sum())

vae_imputer = VAEImputer(
    layers_config=[
        (2, 'leaky_relu'),
        (1, 'leaky_relu'),
        (1, 'leaky_relu'),
    ],
    latent_dim=2,
    beta=1.0,
    stochastic_inference=0.0,
    epochs=50,
    batch_size=8,
    learning_rate=0.001,
    verbose=1,
    random_state=42,
    noise_std=0.2,
    validation_split=0.2,
    early_stopping=True,
)

print("\nTraining VAE imputer...")
X_imputed = vae_imputer.fit_transform(X_missing)

df_imputed = pd.DataFrame(X_imputed, columns=X_complete.columns)
target_names = ['setosa', 'versicolor', 'virginica']
species = [target_names[i] for i in y]

fig = plt.figure(figsize=(16, 6))

# Plot 1: Original
ax1 = fig.add_subplot(121, projection='3d')
colors = ['red', 'green', 'blue']
target_names = ['setosa', 'versicolor', 'virginica']
for i, species_name in enumerate(target_names):
    mask = y == i
    ax1.scatter(X_complete.loc[mask, 'sepal_length'], 
               X_complete.loc[mask, 'petal_length'], 
               X_complete.loc[mask, 'petal_width'],
               c=colors[i], label=species_name, alpha=0.7, s=50)

ax1.set_xlabel('Sepal Length (cm)')
ax1.set_ylabel('Petal Length (cm)')
ax1.set_zlabel('Petal Width (cm)')
ax1.set_title('Original Complete Data')
ax1.legend()

# Plot 2: Imputed
ax2 = fig.add_subplot(122, projection='3d')
for i, species_name in enumerate(target_names):
    mask = y == i
    ax2.scatter(df_imputed.loc[mask, 'sepal_length'], 
               df_imputed.loc[mask, 'petal_length'], 
               df_imputed.loc[mask, 'petal_width'],
               c=colors[i], label=species_name, alpha=0.7, s=50)

ax2.set_xlabel('Sepal Length (cm)')
ax2.set_ylabel('Petal Length (cm)')
ax2.set_zlabel('Petal Width (cm)')
ax2.set_title('VAE Imputed Data')
ax2.legend()

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("PETAL COMPARISON SUMMARY")
print("="*60)

print("\nPetal Length - Original vs Imputed:")
print(f"Original mean: {X_complete['petal_length'].mean():.3f} ± {X_complete['petal_length'].std():.3f}")
print(f"Imputed mean:  {df_imputed['petal_length'].mean():.3f} ± {df_imputed['petal_length'].std():.3f}")

print("\nPetal Width - Original vs Imputed:")
print(f"Original mean: {X_complete['petal_width'].mean():.3f} ± {X_complete['petal_width'].std():.3f}")
print(f"Imputed mean:  {df_imputed['petal_width'].mean():.3f} ± {df_imputed['petal_width'].std():.3f}")

original_corr = X_complete[['petal_length', 'petal_width']].corr().iloc[0, 1]
imputed_corr = df_imputed[['petal_length', 'petal_width']].corr().iloc[0, 1]

print(f"\nPetal Length-Width Correlation:")
print(f"Original: {original_corr:.3f}")
print(f"Imputed:  {imputed_corr:.3f}")

mask = X_missing.isna()
reconstruction_errors = []

for col in ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']:
    col_mask = ~mask[col]
    if col_mask.sum() > 0:
        error = np.mean((X_complete.loc[col_mask, col] - df_imputed.loc[col_mask, col]) ** 2)
        reconstruction_errors.append(error)
        print(f"\nReconstruction MSE for {col}: {error:.6f}")

print(f"\nOverall petal reconstruction MSE: {np.mean(reconstruction_errors):.6f}")