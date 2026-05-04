# 🏠 House Prices — Advanced Regression

Predicting house sale prices using the [Kaggle House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

This project started as an introduction to EDA and data cleaning, then naturally evolved into a full ML pipeline with hyperparameter tuning and model ensembling.

> Developed as a learning project with assistance from Claude and Gemini AI.

## 📊 Dataset

- **Source:** Kaggle — House Prices: Advanced Regression Techniques
- **Train:** 1460 houses, 80 features
- **Test:** 1459 houses
- **Target:** `SalePrice` (USD)

> Data files are not included. Download `train.csv` and `test.csv` from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place them in the project root.

## 🎯 Results

| Version | Approach | Kaggle Score |
|---------|----------|-------------|
| v1 — Baseline | Single GradientBoosting, LabelEncoder | 0.13695 |
| v2 — Model comparison | 8 models, best selected automatically | 0.12916 |
| v3 — Optuna + Ensemble | CatBoost (Optuna) + XGBoost + Ridge | 0.12497 |
| **v4 — Advanced** | Ordinal encoding + skewness correction + improved ensemble | **0.12281** |

## 📁 Structure

```
house-prices/
├── main.py                  # baseline: 8 models comparison
├── main_advanced.py         # best result: Optuna + ensemble
├── submission.csv           # baseline predictions
├── submission_advanced.csv  # best predictions
├── requirements.txt
├── README.md
└── README UA.md
```

> Data files (`train.csv`, `test.csv`) are not included per Kaggle terms of use.

## 🔧 Pipeline (main_advanced.py)

**1. Data Cleaning**
- Quality/feature-absence columns → `'None'` or `0`
- `LotFrontage` → neighborhood median
- Remaining → mode / median

**2. Ordinal Encoding**
Quality columns have a natural order — mapped to numbers instead of arbitrary codes:
```python
{'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
```
Applied to: `ExterQual`, `BsmtQual`, `KitchenQual`, `FireplaceQu`, etc.

**3. Feature Engineering**

| Feature | Formula |
|---------|---------|
| `TotalSF` | TotalBsmtSF + 1stFlrSF + 2ndFlrSF |
| `TotalBath` | FullBath + HalfBath×0.5 + BsmtFullBath + BsmtHalfBath×0.5 |
| `Age` | YrSold − YearBuilt |
| `IsNew` | 1 if sold in year of construction |

**4. Skewness Correction**
Features with `|skew| > 0.75` transformed with `log1p` — reduces the effect of extreme values on linear models.

**5. Encoding & Scaling**
- Remaining categorical → `pd.get_dummies` (One-Hot)
- `RobustScaler` for Ridge — resistant to outliers

**6. Hyperparameter Tuning**
`Optuna` with 55 trials optimizes CatBoost parameters:
- `learning_rate`: 0.005–0.03
- `depth`: 4–6
- `l2_leaf_reg`: 1–10

**7. Ensemble (Blending)**
```
Final = 0.4 × CatBoost + 0.2 × XGBoost + 0.4 × Ridge
```
Each model captures different patterns — CatBoost handles non-linearity, Ridge provides stable linear baseline.

## ⚙️ How to Run

```bash
pip install -r requirements.txt

# Model comparison
python main.py

# Best result
python main_advanced.py
```

## 🛠 Tech Stack

- **pandas, numpy** — data processing
- **scikit-learn** — Ridge, RobustScaler, cross-validation
- **XGBoost / CatBoost** — gradient boosting
- **Optuna** — hyperparameter optimization
- **scipy** — skewness calculation

## 💡 Key Findings

**Train and test must be processed together.** Concatenating both datasets before encoding ensures identical feature columns — a common source of bugs when done separately.

**Ordinal encoding beats Label Encoding for quality features.** Mapping `Ex > Gd > TA > Fa > Po` to `5-4-3-2-1` gives the model meaningful numeric relationships.

**Adding models to an ensemble doesn't always help.** Lasso and ElasticNet were tested but made results worse — the existing CatBoost + XGBoost + Ridge combination already covered different error patterns.

**Optuna > manual tuning.** 55 automated trials consistently outperformed hand-picked parameters, especially for `l2_leaf_reg`.
