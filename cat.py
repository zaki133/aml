import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Load the cleaned dataset
df = pd.read_csv('house_prices_cleaned.csv')

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
cat_model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, random_seed=42, verbose=100)

# Train the model
cat_model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_val, y_val), early_stopping_rounds=10)

# Predict on the validation set
y_val_pred = cat_model.predict(X_val)

# Calculate MAPE
val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100

print(f'Validation MAPE: {val_mape:.4f}%')
