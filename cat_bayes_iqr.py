import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error
from catboost import CatBoostRegressor
from bayes_opt import BayesianOptimization

# Load the cleaned dataset
df = pd.read_csv('house_prices_cleaned_v5_unscaled.csv')

# Separate features and target
X = df.drop(columns=['SalePrice', 'Id'])
y = df['SalePrice']

# Identify categorical features
cat_features = [X.columns.get_loc(col) for col in X.select_dtypes(include='object').columns]

# Log transformation on skewed numerical features in the entire dataset except SalePrice
numerical_features = X.select_dtypes(include=[np.number])
skewed_cols = numerical_features.apply(lambda x: x.skew()).sort_values(ascending=False)
skewness_threshold = 0.75
high_skewness = skewed_cols[skewed_cols > skewness_threshold]

for col in high_skewness.index:
    X[col] = np.log1p(X[col])

# Apply IQR method to remove outliers from the entire dataset for numerical features only
numerical_cols = X.select_dtypes(include=[np.number])
Q1 = numerical_cols.quantile(0.25)
Q3 = numerical_cols.quantile(0.75)
IQR = Q3 - Q1

# Removing outliers from the numerical columns
outlier_condition = ~((numerical_cols < (Q1 - 1.5 * IQR)) | (numerical_cols > (Q3 + 1.5 * IQR))).any(axis=1)
X = X[outlier_condition]
y = y[outlier_condition]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log-transform the target variable in the training set
y_train_log = np.log1p(y_train)

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
    model.fit(X_train, y_train_log)
    y_val_pred_log = model.predict(X_test)
    y_val_pred = np.expm1(y_val_pred_log)
    return -mean_absolute_percentage_error(y_test, y_val_pred)  # Maximize negative MAPE

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
optimizer.maximize(init_points=10, n_iter=50)

# Get the best parameters
best_params = optimizer.max['params']
best_params['depth'] = int(best_params['depth'])
best_params['iterations'] = int(best_params['iterations'])
print(f'Best parameters found: {best_params}')

# Train the final model with the best parameters
best_model = CatBoostRegressor(**best_params, cat_features=cat_features, random_seed=42, verbose=0)
best_model.fit(X_train, y_train_log)

# Predict on the test dataset
y_test_pred_log = best_model.predict(X_test)

# Reverse the log transformation of predictions
y_test_pred = np.expm1(y_test_pred_log)

# Evaluate the model on the original scale of the test data
test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
print(f'Test MAPE: {test_mape:.4f}%')
