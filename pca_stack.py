import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import BaggingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
import lightgbm as lgb
from xgboost import XGBRegressor

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

# Initialize the best models
cat_model = CatBoostRegressor(iterations=1500, learning_rate=0.01, depth=6, random_seed=42, verbose=0)
bagging_model = BaggingRegressor(base_estimator=cat_model, n_estimators=10, random_state=42)
lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, num_leaves=63, random_state=42)
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)

# Train individual models
cat_model.fit(X_train, y_train)
bagging_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Define the stacking model with different meta-models
stacking_model_ridge = StackingRegressor(
    estimators=[
        ('cat', cat_model),
        ('bagging', bagging_model),
        ('lgb', lgb_model),
        ('xgb', xgb_model)
    ],
    final_estimator=RidgeCV()
)
stacking_model_lasso = StackingRegressor(
    estimators=[
        ('cat', cat_model),
        ('bagging', bagging_model),
        ('lgb', lgb_model),
        ('xgb', xgb_model)
    ],
    final_estimator=LassoCV()
)
stacking_model_elasticnet = StackingRegressor(
    estimators=[
        ('cat', cat_model),
        ('bagging', bagging_model),
        ('lgb', lgb_model),
        ('xgb', xgb_model)
    ],
    final_estimator=ElasticNetCV()
)

# Train and evaluate stacking models
for name, model in [("Ridge", stacking_model_ridge), ("Lasso", stacking_model_lasso), ("ElasticNet", stacking_model_elasticnet)]:
    model.fit(X_train, y_train)
    stacking_pred = model.predict(X_val)
    val_mape = mean_absolute_percentage_error(y_val, stacking_pred) * 100
    print(f'Validation MAPE with {name} Stacking: {val_mape:.4f}%')
