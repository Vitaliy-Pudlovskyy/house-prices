import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ============================================================
# 1 — ЗАВАНТАЖЕННЯ
# ============================================================
df = pd.read_csv('train.csv')
print(f"Завантажено: {df.shape}")

# ============================================================
# 2 — ВИДАЛЕННЯ ВИКИДІВ
# ============================================================
df = df[(df['GrLivArea'] < 4000) | (df['SalePrice'] > 200000)]
print(f"Після видалення викидів: {df.shape}")

# ============================================================
# 3 — ЗАПОВНЕННЯ ПРОПУЩЕНИХ ЗНАЧЕНЬ
# ============================================================

# NaN = "немає цього" → категоріальні
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence',
             'FireplaceQu', 'GarageType', 'GarageFinish',
             'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
for col in none_cols:
    df[col] = df[col].fillna('None')

# NaN = "немає цього" → числові
zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars',
             'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
             'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in zero_cols:
    df[col] = df[col].fillna(0)

# NaN = дані відсутні → заповнюємо статистикою
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

print(f"Пропущених значень: {df.isnull().sum().sum()} ✓")

# ============================================================
# 4 — FEATURE ENGINEERING
# ============================================================
df['Age']         = 2010 - df['YearBuilt']
df['TotalSF']     = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['TotalBath']   = (df['FullBath'] + df['HalfBath'] * 0.5 +
                     df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5)
df['HasGarage']    = (df['GarageArea'] > 0).astype(int)
df['HasPool']      = (df['PoolArea'] > 0).astype(int)
df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

# ============================================================
# 5 — КОДУВАННЯ І ФІНАЛЬНА ПІДГОТОВКА
# ============================================================
df = df.drop(['Id', 'IsRemodeled'], axis=1, errors='ignore')

cat_cols = df.select_dtypes(include='str').columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print(f"Фінальний датасет: {df.shape}")

# ============================================================
# 6 — ПІДГОТОВКА ДО МОДЕЛІ
# ============================================================
X = df.drop('SalePrice', axis=1)
y = np.log(df['SalePrice'])  # логарифм — краща метрика для цін

print(f"X: {X.shape} | y: {y.shape}\n")

# ============================================================
# 7 — ПОРІВНЯННЯ МОДЕЛЕЙ
# ============================================================
models = {
    'Linear Regression': LinearRegression(),
    'Ridge':             Ridge(),
    'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                    max_depth=4, random_state=42),
    'XGBoost':           XGBRegressor(n_estimators=300, learning_rate=0.05,
                                      max_depth=4, random_state=42, verbosity=0),
    'LightGBM':          LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                       num_leaves=31, random_state=42, verbose=-1),
    'CatBoost':          CatBoostRegressor(iterations=1200, learning_rate=0.05,
                                           depth=4, random_seed=42, verbose=False),
    'Neural Network':    Pipeline([
                             ('scaler', StandardScaler()),
                             ('mlp',    MLPRegressor(hidden_layer_sizes=(16, 8, 4),
                                                     max_iter=1000, random_state=42,
                                                     early_stopping=True))
                         ]),
}

print("=" * 45)
print("  ПОРІВНЯННЯ МОДЕЛЕЙ (5-fold CV)")
print("=" * 45)

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    rmse = -scores.mean()
    results[name] = rmse
    print(f"  {name:25s} RMSE: {rmse:.4f}")

best_name = min(results, key=results.get)
print(f"\n  Найкраща модель: {best_name} → {results[best_name]:.4f}")
print("=" * 45)

# ============================================================
# 8 — SUBMISSION З НАЙКРАЩОЮ МОДЕЛЛЮ
# ============================================================
test = pd.read_csv('test.csv')
test_ids = test['Id']
test = test.drop('Id', axis=1)

for col in none_cols:
    if col in test.columns:
        test[col] = test[col].fillna('None')

for col in zero_cols:
    if col in test.columns:
        test[col] = test[col].fillna(0)

test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)
test['Electrical'] = test['Electrical'].fillna(test['Electrical'].mode()[0])
test = test.fillna(test.median(numeric_only=True))

test['Age']          = 2010 - test['YearBuilt']
test['TotalSF']      = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
test['TotalBath']    = (test['FullBath'] + test['HalfBath'] * 0.5 +
                        test['BsmtFullBath'] + test['BsmtHalfBath'] * 0.5)
test['HasGarage']    = (test['GarageArea'] > 0).astype(int)
test['HasPool']      = (test['PoolArea'] > 0).astype(int)
test['HasFireplace'] = (test['Fireplaces'] > 0).astype(int)

cat_cols_test = test.select_dtypes(include='str').columns
for col in cat_cols_test:
    test[col] = le.fit_transform(test[col].astype(str))

best_model = models[best_name]
best_model.fit(X, y)
preds = np.exp(best_model.predict(test))

submission = pd.DataFrame({'Id': test_ids, 'SalePrice': preds})
submission.to_csv('submission.csv', index=False)
print(f"\nSubmission готовий з моделлю: {best_name}")
print(submission.head())