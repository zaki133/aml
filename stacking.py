import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error

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

# Initialize the base models
cat_model = CatBoostRegressor(cat_features=cat_features, random_seed=42, verbose=0)
gbr_model = GradientBoostingRegressor(random_state=42)

# Define the stacking model
stacking_model = StackingRegressor(
    estimators=[
        ('cat', cat_model),
        ('gbr', gbr_model)
    ],
    final_estimator=RidgeCV()
)

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = stacking_model.predict(X_val)

# Calculate MAPE
val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
print(f'Validation MAPE with Stacking: {val_mape:.4f}%')
