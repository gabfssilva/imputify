# Imputify

A Python library for **evaluating and performing missing data imputation**. It measures imputation quality across three dimensions: reconstruction (how close are imputed values to the truth?), distribution preservation (are statistical properties maintained?), and predictive utility (can downstream models still perform well?).

The library is fully compatible with scikit-learn's `fit`/`transform` API and provides ready-to-use imputers: KNN, statistical baselines, autoencoders (DAE, VAE), GAIN, and a decoder-only LLM fine-tuned for tabular imputation.

> This library is part of my master's research proposal, so apart from scikit-learn compatibility, expect breaking changes. The API will stabilize as the research progresses.

---

## Why missingness matters

Not all missing data is created equal. The *mechanism* behind missingness determines which imputation methods will work. **MCAR** (Missing Completely at Random) is the easy case, values disappear randomly with no pattern, like a sensor failing at random times. **MAR** (Missing at Random) is trickier, missingness depends on *other* observed values, like high earners being more likely to skip income questions. **MNAR** (Missing Not at Random) is the hardest, missingness depends on the *missing value itself*, like very sick patients being unable to complete health surveys.

Most imputation methods assume MCAR or MAR. MNAR breaks these assumptions because the data you're trying to recover is exactly what's causing it to be missing. This is where I think LLMs might help, they can learn complex conditional distributions from the observed data and extrapolate patterns that simpler methods miss.

## Evaluation

A good imputation isn't just "close to the true value". Imputify measures quality from three complementary perspectives:

**Reconstruction**, point-wise accuracy, with MAE, RMSE, NRMSE for numerical features and accuracy for categorical features.

**Distribution**, statistical properties, as Wasserstein distance, KS statistic, KL divergence (how much distributions shifted), as well as Correlation shift (did we break relationships between variables?)

**Predictive utility**, downstream impact, by training a model on original vs imputed data and compare the performance gap. 

> Predictive metrics:
> 
> **Classification**: accuracy, precision, recall, F1 
> 
> **Regression**: R², MAE, RMSE

The overall score combines these into a single number in [0, 1]. Reconstruction and distribution are normalized as `1 / (1 + error)`, predictive as `1 - |Δmetrics|`. The final score is the mean of all three.

## Imputers

| Imputer | Category | Description |
|---------|----------|-------------|
| `StatisticalImputer` | Baseline | Mean/median for numerical, mode for categorical |
| `KNNImputer` | Baseline | k-nearest neighbors |
| `MICEImputer` | Baseline | Multiple Imputation by Chained Equations |
| `MissForestImputer` | Baseline | Random Forest-based iterative imputation |
| `XGBoostImputer` | Baseline | XGBoost-based iterative imputation |
| `DAEImputer` | Deep Learning | Denoising AutoEncoder with swap noise |
| `VAEImputer` | Deep Learning | Variational AutoEncoder (probabilistic latent space) |
| `GAINImputer` | Deep Learning | Generative Adversarial Imputation Nets |
| `DecoderOnlyImputer` | LLM | Fine-tuned decoder-only transformer via structured JSON serialization |

## Example

```python
from imputify.imputer import DAEImputer
from imputify.missing import introduce_missing, PatternConfig
from imputify.metrics import evaluate

# Create realistic missing data (MNAR pattern)
pattern = PatternConfig(incomplete_vars=['income'], mechanism='MNAR')
X_missing, mask = introduce_missing(X, proportion=0.3, patterns=[pattern])

# Impute
imputer = DAEImputer(hidden_dim=128, epochs=100)
X_imputed = imputer.fit_transform(X_missing)

# Evaluate across all dimensions
results = evaluate(X_original, X_imputed, mask, y=y)
print(f"Overall score: {results.overall_score:.3f}")
```

## Installation

If you don't have `uv` installed, do yourself a favor and:

```bash
# Linux & macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS (Homebrew)
brew install uv

# Windows
# Well, check their installation page: https://docs.astral.sh/uv/getting-started/installation/
```

Once installed, simply clone the repo and run `uv sync` to install dependencies:

```bash
git clone https://github.com/gabfssilva/imputify
cd imputify
uv sync
```

Requires Python 3.12+. 

Open the project using your favorite IDE and that's it.

## License

MIT
