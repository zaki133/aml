import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_percentage_error
from catboost import CatBoostRegressor
from sklearn.neighbors import LocalOutlierFactor

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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Local Outlier Factor (LOF) to remove outliers from the training set
lof = LocalOutlierFactor(contamination=0.1, n_neighbors=20)
outliers = lof.fit_predict(X_train.select_dtypes(include=[np.number]))

# Keep only inliers in the training data
X_train = X_train[outliers == 1]
y_train = y_train[outliers == 1]

# Define the best parameters found
best_params = {
    'bagging_temperature': 0.22606655286083255,
    'depth': 5,
    'iterations': 1087,
    'l2_leaf_reg': 9.104770789148295,
    'learning_rate': 0.07019277755404353
}

# Train the model using 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model = CatBoostRegressor(
        **best_params,
        cat_features=cat_features,
        random_seed=42,
        verbose=0
    )
    
    model.fit(X_train_fold, y_train_fold)
    y_val_pred = model.predict(X_val_fold)
    fold_mape = mean_absolute_percentage_error(y_val_fold, y_val_pred) * 100
    cv_scores.append(fold_mape)

# Calculate the mean and standard deviation of the MAPE scores
cv_mape = np.mean(cv_scores)
cv_mape_std = np.std(cv_scores)
print(f'Cross-validated MAPE: {cv_mape:.4f}% ± {cv_mape_std:.4f}%')

# Train the final model on the entire training data
best_model = CatBoostRegressor(**best_params, cat_features=cat_features, random_seed=42, verbose=0)
best_model.fit(X_train, y_train)

# Predict on the test dataset
y_test_pred = best_model.predict(X_test)

# Evaluate the model on the original scale of the test data
test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
print(f'Test MAPE: {test_mape:.4f}%')
