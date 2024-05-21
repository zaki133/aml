from bayes_opt import BayesianOptimization
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd

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

# Define the function to optimize
def catboost_hyperparam(depth, iterations, learning_rate, l2_leaf_reg, bagging_temperature):
    params = {
        'depth': int(depth),
        'iterations': int(iterations),
        'learning_rate': learning_rate,
        'l2_leaf_reg': l2_leaf_reg,
        'bagging_temperature': bagging_temperature,
        'cat_features': cat_features,
        'random_seed': 42,
        'verbose': 0
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    return -mean_absolute_percentage_error(y_val, y_val_pred)  # Maximize negative MAPE

# Define the parameter space
pbounds = {
    'depth': (4, 10),
    'iterations': (1000, 2000),
    'learning_rate': (0.01, 0.1),
    'l2_leaf_reg': (1, 10),
    'bagging_temperature': (0, 1)
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(f=catboost_hyperparam, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=50)

# Get the best parameters
best_params = optimizer.max['params']
best_params['depth'] = int(best_params['depth'])
best_params['iterations'] = int(best_params['iterations'])
print(f'Best parameters found: {best_params}')

# Train the final model with the best parameters
best_model = CatBoostRegressor(**best_params, cat_features=cat_features, random_seed=42, verbose=0)
best_model.fit(X_train, y_train)
y_val_pred = best_model.predict(X_val)
val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
print(f'Validation MAPE: {val_mape:.4f}%')
