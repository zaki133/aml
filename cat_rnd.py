import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import uniform, randint

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

# Define the parameter grid
param_dist = {
    'iterations': randint(1000, 2000),
    'depth': randint(4, 10),
    'learning_rate': uniform(0.01, 0.1),
    'l2_leaf_reg': uniform(1, 10),
    'bagging_temperature': uniform(0, 1),
    'border_count': randint(32, 100)
}

# Initialize the CatBoost model
cat_model = CatBoostRegressor(cat_features=cat_features, random_seed=42, verbose=0)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=cat_model, param_distributions=param_dist, n_iter=100, scoring='neg_mean_absolute_percentage_error', cv=3, n_jobs=-1, random_state=42)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best parameters
best_params = random_search.best_params_
print(f'Best parameters found: {best_params}')

# Predict on the validation set using the best model
best_model = random_search.best_estimator_
y_val_pred = best_model.predict(X_val)

# Calculate MAPE
val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
print(f'Validation MAPE: {val_mape:.4f}%')
