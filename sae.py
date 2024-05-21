import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import StackingRegressor

# Load the cleaned dataset
df = pd.read_csv('house_prices_cleaned_v2.csv')

# Separate features and target
X = df.drop(columns=['SalePrice', 'Id'])
y = df['SalePrice']

# Identify categorical and numerical features
cat_features = [X.columns.get_loc(col) for col in X.select_dtypes(include='object').columns]
num_features = X.select_dtypes(include=[np.number]).columns

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the numerical features for models that need it
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])
X_val_scaled[num_features] = scaler.transform(X_val[num_features])

# Initialize the best CatBoost model
cat_model = CatBoostRegressor(iterations=1500, learning_rate=0.01, depth=6, cat_features=cat_features, random_seed=42, verbose=0)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_val)

# Initialize the best Bagging model with CatBoost as the base estimator
bagging_model = BaggingRegressor(base_estimator=cat_model, n_estimators=10, random_state=42)
bagging_model.fit(X_train, y_train)
bagging_pred = bagging_model.predict(X_val)

# Initialize the best LightGBM model (no scaling needed for LightGBM)
lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05,early_stopping_round=10, max_depth=6, num_leaves=63, random_state=42)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
lgb_pred = lgb_model.predict(X_val)

# Initialize a Linear Model for stacking (needs scaling)
stacking_model = StackingRegressor(
    estimators=[
        ('cat', CatBoostRegressor(iterations=1500, learning_rate=0.01, depth=6, cat_features=cat_features, random_seed=42, verbose=0)),
        ('lgb', lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, num_leaves=63, random_state=42))
    ],
    final_estimator=RidgeCV()
)
stacking_model.fit(X_train_scaled, y_train)
stacking_pred = stacking_model.predict(X_val_scaled)

# Simple averaging of the predictions
final_pred = (cat_pred + bagging_pred + lgb_pred + stacking_pred) / 4

# Calculate MAPE
val_mape = mean_absolute_percentage_error(y_val, final_pred) * 100
print(f'Validation MAPE with Averaging Ensemble: {val_mape:.4f}%')
