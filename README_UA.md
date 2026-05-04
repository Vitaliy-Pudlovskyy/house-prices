# 🏠 House Prices — Регресія Цін на Нерухомість

Передбачення цін продажу будинків на основі [датасету Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

Проект починався як знайомство з EDA і очисткою даних, і природньо переріс у повноцінний ML pipeline з тюнінгом гіперпараметрів та ансамблюванням моделей.

> Розроблений як навчальний проект з використанням Claude та Gemini AI.

## 📊 Дані

- **Джерело:** Kaggle — House Prices: Advanced Regression Techniques
- **Train:** 1460 будинків, 80 ознак
- **Test:** 1459 будинків
- **Ціль:** `SalePrice` (USD)

> Файли даних не включені. Завантаж `train.csv` і `test.csv` з [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) і поклади в корінь проекту.

## 🎯 Результати

| Версія | Підхід | Kaggle Score |
|--------|--------|-------------|
| v1 — Baseline | Одна модель GradientBoosting, LabelEncoder | 0.13695 |
| v2 — Порівняння | 8 моделей, найкраща вибирається автоматично | 0.12916 |
| v3 — Optuna + Ансамбль | CatBoost (Optuna) + XGBoost + Ridge | 0.12497 |
| **v4 — Покращений** | Ordinal encoding + корекція асиметрії + кращий ансамбль | **0.12281** |

## 📁 Структура

```
house-prices/
├── main.py                  # базовий: порівняння 8 моделей
├── main_advanced.py         # найкращий результат: Optuna + ансамбль
├── submission.csv           # базові передбачення
├── submission_advanced.csv  # найкращі передбачення
├── requirements.txt
├── README.md
└── README UA.md
```

> Файли даних (`train.csv`, `test.csv`) не включені згідно умов використання Kaggle.

## 🔧 Pipeline (main_advanced.py)

**1. Очистка даних**
- Колонки відсутності ознаки → `'None'` або `0`
- `LotFrontage` → медіана по району
- Решта → мода / медіана

**2. Ordinal Encoding**
Якісні колонки мають природній порядок — маппінг на числа замість довільних кодів:
```python
{'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
```
Застосовано до: `ExterQual`, `BsmtQual`, `KitchenQual`, `FireplaceQu` і т.д.

**3. Feature Engineering**

| Ознака | Формула |
|--------|---------|
| `TotalSF` | TotalBsmtSF + 1stFlrSF + 2ndFlrSF |
| `TotalBath` | FullBath + HalfBath×0.5 + BsmtFullBath + BsmtHalfBath×0.5 |
| `Age` | YrSold − YearBuilt |
| `IsNew` | 1 якщо продано у рік побудови |

**4. Корекція асиметрії**
Ознаки з `|skew| > 0.75` трансформовані через `log1p` — зменшує вплив екстремальних значень на лінійні моделі.

**5. Кодування і масштабування**
- Решта категоріальних → `pd.get_dummies` (One-Hot)
- `RobustScaler` для Ridge — стійкий до викидів

**6. Підбір гіперпараметрів**
`Optuna` з 55 спробами оптимізує параметри CatBoost:
- `learning_rate`: 0.005–0.03
- `depth`: 4–6
- `l2_leaf_reg`: 1–10

**7. Ансамбль (Blending)**
```
Фінал = 0.4 × CatBoost + 0.2 × XGBoost + 0.4 × Ridge
```
Кожна модель вловлює різні патерни — CatBoost обробляє нелінійність, Ridge дає стабільний лінійний базис.

## ⚙️ Запуск

```bash
pip install -r requirements.txt

# Порівняння моделей
python main.py

# Найкращий результат
python main_advanced.py
```

## 🛠 Технології

- **pandas, numpy** — обробка даних
- **scikit-learn** — Ridge, RobustScaler, крос-валідація
- **XGBoost / CatBoost** — градієнтний бустинг
- **Optuna** — автоматичний підбір гіперпараметрів
- **scipy** — обчислення асиметрії

## 💡 Головні висновки

**Train і test треба обробляти разом.** Об'єднання датасетів перед кодуванням гарантує однакові колонки — часта причина помилок коли обробляють окремо.

**Ordinal encoding краще за Label Encoding для якісних ознак.** Маппінг `Ex > Gd > TA > Fa > Po` на `5-4-3-2-1` дає моделі значущі числові відношення.

**Більше моделей в ансамблі не завжди краще.** Lasso і ElasticNet тестувались але погіршили результат — комбінація CatBoost + XGBoost + Ridge вже покривала різні патерни помилок.

**Optuna краще за ручний підбір.** 55 автоматичних спроб стабільно перевершують ручно підібрані параметри, особливо для `l2_leaf_reg`.
