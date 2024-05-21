from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import StackingRegressor

# Load the cleaned dataset
df = pd.read_csv('house_prices_cleaned_v2.csv')

# Separate features and target
X = df.drop(columns=['SalePrice', 'Id'])
y = df['SalePrice']

# Identify categorical and numerical features
cat_features = [X.columns.get_loc(col) for col in X.select_dtypes(include='object').columns]
num_features = X.select_dtypes(include=[np.number]).columns

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the best models
cat_model = CatBoostRegressor(iterations=1500, learning_rate=0.01, depth=6, cat_features=cat_features, random_seed=42, verbose=0)
lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, num_leaves=63, random_state=42)
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)

# Train individual models
cat_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Create and train the voting regressor
voting_model = VotingRegressor(estimators=[
    ('cat', cat_model),
    ('lgb', lgb_model),
    ('xgb', xgb_model)
])
voting_model.fit(X_train, y_train)
y_val_pred = voting_model.predict(X_val)
val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
print(f'Validation MAPE with Voting Regressor: {val_mape:.4f}%')
