import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Load the cleaned dataset
df = pd.read_csv('house_prices_cleaned_v2.csv')

# Separate features and target
X = df.drop(columns=['SalePrice', 'Id'])
y = df['SalePrice']

# Identify categorical and numerical features
cat_features = X.select_dtypes(include='object').columns
num_features = X.select_dtypes(include=[np.number]).columns

# Encode categorical features
for col in cat_features:
    X[col] = LabelEncoder().fit_transform(X[col])

# Standardize numerical features
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# Apply PCA
pca = PCA(n_components=0.95, random_state=42)  # Keep 95% of the variance
X_pca = pca.fit_transform(X[num_features])

# Combine PCA features with categorical features
X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
X_combined = pd.concat([X_pca_df, X[cat_features].reset_index(drop=True)], axis=1)

# Convert to numpy arrays
X_combined = X_combined.values
y = y.values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Initialize the CatBoost model without specifying cat_features
cat_model = CatBoostRegressor(random_seed=42, verbose=0)

# Initialize the Bagging Regressor with CatBoost as the base estimator
bagging_model = BaggingRegressor(base_estimator=cat_model, n_estimators=10, random_state=42)

# Train the Bagging model
bagging_model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = bagging_model.predict(X_val)

# Calculate MAPE
val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
print(f'Validation MAPE with Bagging and PCA: {val_mape:.4f}%')
