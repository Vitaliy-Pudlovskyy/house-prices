import pandas as pd
import numpy as np
import optuna
from sklearn.linear_model import Ridge, LassoCV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from scipy.stats import skew

# 1 — ЗАВАНТАЖЕННЯ
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
test_ids = df_test['Id']

# 2 — ВИДАЛЕННЯ ВИКИДІВ
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)

all_data = pd.concat((df_train.drop(['SalePrice', 'Id'], axis=1),
                      df_test.drop(['Id'], axis=1))).reset_index(drop=True)

# 3 — ЗАПОВНЕННЯ ПРОПУСКІВ
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
             'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
for col in none_cols:
    all_data[col] = all_data[col].fillna('None')

zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
             'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in zero_cols:
    all_data[col] = all_data[col].fillna(0)

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.fillna(all_data.median(numeric_only=True))
all_data = all_data.fillna(all_data.mode().iloc[0])

# 4 — ORDINAL ENCODING (Важливо для Ridge)
ord_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
ord_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
for col in ord_cols:
    all_data[col] = all_data[col].map(ord_map).astype(float)

# 5 — FEATURE ENGINEERING
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath'])
all_data['Age'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['IsNew'] = (all_data['YearBuilt'] == all_data['YrSold']).astype(int)

# 6 — SKEWNESS & LOG TRANSFORMATION
numeric_feats = all_data.select_dtypes(include=[np.number]).columns
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.75]

for feat in high_skew.index:
    all_data[feat] = np.log1p(all_data[feat])

# One-Hot Encoding
all_data = pd.get_dummies(all_data)

# Розподіл на Train/Test
X = all_data[:len(df_train)]
test = all_data[len(df_train):]
y = np.log1p(df_train['SalePrice'])

# Масштабування (RobustScaler стійкий до викидів)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

# 7 — OPTUNA ДЛЯ CATBOOST
def objective(trial):
    params = {
        "iterations": 1500,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03),
        "depth": trial.suggest_int("depth", 4, 6),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_seed": 42,
        "verbose": False
    }
    model = CatBoostRegressor(**params)
    # Використовуємо X (не Scaled), бо CatBoost сам добре працює з сирими даними
    score = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    return -score.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=55) # Рекомендую 50+

# 8 — ТРЕНУВАННЯ ТА АНСАМБЛЬ
print("Фінальне тренування...")
best_cat = CatBoostRegressor(**study.best_params, verbose=False)
best_xgb = XGBRegressor(n_estimators=3000, learning_rate=0.01, max_depth=4, subsample=0.7, colsample_bytree=0.7)
best_ridge = Ridge(alpha=15)

best_cat.fit(X, y)
best_xgb.fit(X, y)
best_ridge.fit(X_scaled, y) # Для Ridge використовуємо Scaled дані
# Прогнози
preds_cat = np.expm1(best_cat.predict(test))
preds_xgb = np.expm1(best_xgb.predict(test))
preds_ridge = np.expm1(best_ridge.predict(test_scaled))

# Blending з акцентом на CatBoost та Ridge
final_preds = (0.4 * preds_cat) + (0.2 * preds_xgb) + (0.4 * preds_ridge)

# 9 — ЗБЕРЕЖЕННЯ
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': final_preds})
submission.to_csv('submission_advanced.csv', index=False)
print("Файл 'submission_advanced.csv' готовий!")