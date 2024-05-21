import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Load the cleaned dataset
df = pd.read_csv('house_prices_cleaned_v2.csv')

# Separate features and target
X = df.drop(columns=['SalePrice', 'Id'])
y = df['SalePrice']

# Identify categorical features
cat_features = X.select_dtypes(include='object').columns
num_features = X.select_dtypes(include=[np.number]).columns

# Encode categorical features
for col in cat_features:
    X[col] = LabelEncoder().fit_transform(X[col])

# Standardize numerical features
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# Convert to numpy arrays
X = X.values
y = y.values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
cat_model = CatBoostRegressor(random_seed=42, verbose=0)
lgb_model = LGBMRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42)

# Train models
cat_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Predict on validation set
cat_pred = cat_model.predict(X_val)
lgb_pred = lgb_model.predict(X_val)
xgb_pred = xgb_model.predict(X_val)

# Blending predictions (you can try different weights)
final_pred = (cat_pred + lgb_pred + xgb_pred) / 3

# Calculate MAPE
val_mape = mean_absolute_percentage_error(y_val, final_pred) * 100
print(f'Validation MAPE with Blending: {val_mape:.4f}%')
