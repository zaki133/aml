import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_percentage_error
from catboost import CatBoostRegressor
from sklearn.preprocessing import RobustScaler

# Load the cleaned dataset
df = pd.read_csv('house_prices_cleaned_v5_unscaled.csv')

# Separate features and target
X = df.drop(columns=['SalePrice', 'Id'])
y = df['SalePrice']

# Identify categorical features
cat_features = [X.columns.get_loc(col) for col in X.select_dtypes(include='object').columns]

# Apply robust scaling to numerical features
numerical_features = X.select_dtypes(include=[np.number])
scaler = RobustScaler()
X[numerical_features.columns] = scaler.fit_transform(numerical_features)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply robust scaling to the target variable in the training set
y_scaler = RobustScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()

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
    y_train_fold, y_val_fold = y_train_scaled[train_index], y_train_scaled[val_index]
    
    model = CatBoostRegressor(
        **best_params,
        cat_features=cat_features,
        random_seed=42,
        verbose=0
    )
    
    model.fit(X_train_fold, y_train_fold)
    y_val_pred_scaled = model.predict(X_val_fold)
    y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
    fold_mape = mean_absolute_percentage_error(y_train.iloc[val_index], y_val_pred) * 100
    cv_scores.append(fold_mape)

# Calculate the mean and standard deviation of the MAPE scores
cv_mape = np.mean(cv_scores)
cv_mape_std = np.std(cv_scores)
print(f'Cross-validated MAPE: {cv_mape:.4f}% ± {cv_mape_std:.4f}%')

# Train the final model on the entire training data
best_model = CatBoostRegressor(**best_params, cat_features=cat_features, random_seed=42, verbose=0)
best_model.fit(X_train, y_train_scaled)

# Predict on the test dataset
y_test_pred_scaled = best_model.predict(X_test)

# Reverse the robust scaling of predictions
y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

# Evaluate the model on the original scale of the test data
test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
print(f'Test MAPE: {test_mape:.4f}%')
