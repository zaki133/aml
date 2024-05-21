import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from sklearn.metrics import make_scorer, mean_absolute_percentage_error

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

# Define the parameter grid
param_grid = {
    'iterations': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8]
}

# Initialize the CatBoost model
cat_model = CatBoostRegressor(cat_features=cat_features, random_seed=42, verbose=0)

# Define the scoring metric
scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=cat_model, param_grid=param_grid, scoring=scorer, cv=3, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=10)

# Get the best parameters
best_params = grid_search.best_params_
print(f'Best parameters found: {best_params}')

# Predict on the validation set using the best model
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val)

# Calculate MAPE
val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
print(f'Validation MAPE: {val_mape:.4f}%')
