import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Load the cleaned dataset
df = pd.read_csv('house_prices_cleaned.csv')

# Separate features and target
X = df.drop(columns=['SalePrice', 'Id'])
y = df['SalePrice']

# Ensure all columns are numeric
for col in X.select_dtypes(include='object').columns:
    if len(X[col].unique()) <= 10:  # Label encode for columns with fewer unique values
        X[col] = LabelEncoder().fit_transform(X[col])
    else:  # One-hot encode for columns with more unique values
        X = pd.get_dummies(X, columns=[col], drop_first=True)

# Convert to numpy arrays
X = X.values
y = y.values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Initialize the Gradient Boosting model
gbm_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)

# Train the model
gbm_model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = gbm_model.predict(X_val)

# Calculate MAPE
val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100

print(f'Validation MAPE: {val_mape:.4f}%')
