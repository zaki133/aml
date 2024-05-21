import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import StackingRegressor

# Load the cleaned dataset
df = pd.read_csv('house_prices_cleaned_v2.csv')

# Separate features and target
X = df.drop(columns=['SalePrice', 'Id'])
y = df['SalePrice']

# Identify categorical features
cat_features = [X.columns.get_loc(col) for col in X.select_dtypes(include='object').columns]

# Convert to numpy arrays
X = X.values
y = y.values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the best models
cat_model = CatBoostRegressor(iterations=1500, learning_rate=0.01, depth=6, cat_features=cat_features, random_seed=42, verbose=0)
bagging_model = BaggingRegressor(estimator=cat_model, n_estimators=10, random_state=42)
lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, num_leaves=63, random_state=42)

# Train individual models
cat_model.fit(X_train, y_train)
bagging_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

# Define the stacking model with different meta-models
stacking_model_ridge = StackingRegressor(
    estimators=[
        ('cat', cat_model),
        ('bagging', bagging_model),
        ('lgb', lgb_model)
    ],
    final_estimator=RidgeCV()
)
stacking_model_lasso = StackingRegressor(
    estimators=[
        ('cat', cat_model),
        ('bagging', bagging_model),
        ('lgb', lgb_model)
    ],
    final_estimator=LassoCV()
)
stacking_model_elasticnet = StackingRegressor(
    estimators=[
        ('cat', cat_model),
        ('bagging', bagging_model),
        ('lgb', lgb_model)
    ],
    final_estimator=ElasticNetCV()
)

# Train and evaluate stacking models
for name, model in [("ElasticNet", stacking_model_elasticnet)]:
    model.fit(X_train, y_train)
    stacking_pred = model.predict(X_val)
    val_mape = mean_absolute_percentage_error(y_val, stacking_pred) * 100
    print(f'Validation MAPE with {name} Stacking: {val_mape:.4f}%')
