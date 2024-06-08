import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error
from catboost import CatBoostRegressor
from sklearn.cluster import DBSCAN
# Load the cleaned dataset
df = pd.read_csv('house_prices_cleaned_v9_unscaled.csv')


# Separate features and target
X = df.drop(columns=['SalePrice', 'Id'])
y = df['SalePrice']

# Identify categorical features
cat_features = [X.columns.get_loc(col) for col in X.select_dtypes(include='object').columns]

# # Log transformation on skewed numerical features in the entire dataset except SalePrice
# numerical_features = X.select_dtypes(include=[np.number])
# skewed_cols = numerical_features.apply(lambda x: x.skew()).sort_values(ascending=False)
# skewness_threshold = 0.75
# high_skewness = skewed_cols[skewed_cols > skewness_threshold]

# for col in high_skewness.index:
#     X[col] = np.log1p(X[col])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log-transform the target variable in the training set
y_train_log = np.log1p(y_train)

# Best parameters found from Bayesian Optimization
best_params = {
    'bagging_temperature': 0.7135283954371104,
    'depth': 3,
    'iterations': 1091,
    'l2_leaf_reg': 6.754156833807452,
    'learning_rate': 0.09953416832883896
}

# Train the model using cross-validation
model = CatBoostRegressor(
    **best_params,
    cat_features=cat_features,
    random_seed=42,
    verbose=0
)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train_log, cv=kf, scoring='neg_mean_absolute_percentage_error')

# Calculate the mean and standard deviation of the MAPE scores
cv_mape = -cv_scores.mean() * 100
cv_mape_std = cv_scores.std() * 100
print(f'Cross-validated MAPE: {cv_mape:.4f}% ± {cv_mape_std:.4f}%')

# Train the final model on the entire training data
model.fit(X_train, y_train_log)

# Predict on the test dataset
y_test_pred_log = model.predict(X_test)

# Reverse the log transformation of predictions
y_test_pred = np.expm1(y_test_pred_log)

# Evaluate the model on the original scale of the test data
test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
print(f'Test MAPE: {test_mape:.4f}%')