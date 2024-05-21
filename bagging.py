import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Load the cleaned dataset
df = pd.read_csv('house_prices_poly.csv')

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

# Initialize the CatBoost model
cat_model = CatBoostRegressor(cat_features=cat_features, random_seed=42, verbose=0)

# Initialize the Bagging Regressor with CatBoost as the base estimator
bagging_model = BaggingRegressor(base_estimator=cat_model, n_estimators=10, random_state=42)

# Train the Bagging model
bagging_model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = bagging_model.predict(X_val)

# Calculate MAPE
val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
print(f'Validation MAPE with Bagging: {val_mape:.4f}%')
