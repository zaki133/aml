import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import json

# Load the dataset
df = pd.read_csv('dataset.csv')

# Step 1: Handling Missing Values for columns that we don't know anything about..
imputer_mode = SimpleImputer(strategy='most_frequent')
categorical_cols = ['Electrical']
df[categorical_cols] = imputer_mode.fit_transform(df[categorical_cols])

# Fill meaningful NaN values
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df.YearBuilt)
df['Alley'] = df['Alley'].fillna('NoAlley')
df['PoolQC'] = df['PoolQC'].fillna('NoPool')
df['Fence'] = df['Fence'].fillna('NoFence')
df['MiscFeature'] = df['MiscFeature'].fillna('None')
df['BsmtQual'] = df['BsmtQual'].fillna('NoBasement')
df['BsmtCond'] = df['BsmtCond'].fillna('NoBasement')
df['BsmtExposure'] = df['BsmtExposure'].fillna('NoBasement')
df['BsmtFinType1'] = df['BsmtFinType1'].fillna('NoBasement')
df['BsmtFinType2'] = df['BsmtFinType2'].fillna('NoBasement')
df['FireplaceQu'] = df['FireplaceQu'].fillna('NoFireplace')
df['GarageType'] = df['GarageType'].fillna('NoGarage')
df['GarageFinish'] = df['GarageFinish'].fillna('NoGarage')
df['GarageQual'] = df['GarageQual'].fillna('NoGarage')
df['GarageCond'] = df['GarageCond'].fillna('NoGarage')
df['MasVnrType'] = df['MasVnrType'].fillna('None')

# Step 2: Encoding Categorical Variables
label_enc = LabelEncoder()
ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'MSSubClass', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
for col in ordinal_cols:
    df[col] = label_enc.fit_transform(df[col])

df['CentralAir'] = df['CentralAir'].map({'Y': 1, 'N': 0})

# Relevant features for LotFrontage imputation
features_for_imputation = ['LotArea', 'LotConfig', 'LotShape', 'Alley', 'MSZoning', 'BldgType', 'Neighborhood', 'Condition1', 'Condition2', 'GarageCars']

# Create the training data for LotFrontage imputation
train_data = df[~df['LotFrontage'].isnull()]
test_data = df[df['LotFrontage'].isnull()]

X_train = train_data[features_for_imputation]
y_train = train_data['LotFrontage']
X_test = test_data[features_for_imputation]

# Dummify categorical variables and normalize the data
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Ensure the test set has the same dummy variables as the training set
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[X_train.columns]

# Define the models to compare
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.001),
    'KNeighbors': KNeighborsRegressor(n_neighbors=5),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42)
}

# Perform k-fold cross-validation and compare models
kf = KFold(n_splits=10, shuffle=True, random_state=3)
model_scores = {}

for name, model in models.items():
    mae_scores = []
    rmse_scores = []
    r2_scores = []
    for trn_idx, tst_idx in kf.split(X_train):
        model.fit(X_train.iloc[trn_idx], y_train.iloc[trn_idx])
        preds = model.predict(X_train.iloc[tst_idx])
        mae_scores.append(mean_absolute_error(y_train.iloc[tst_idx], preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_train.iloc[tst_idx], preds)))
        r2_scores.append(r2_score(y_train.iloc[tst_idx], preds))
    model_scores[name] = {
        'MAE': np.mean(mae_scores),
        'RMSE': np.mean(rmse_scores),
        'R2': np.mean(r2_scores)
    }
    print(f'{name} MAE: {np.mean(mae_scores):.3f}, RMSE: {np.mean(rmse_scores):.3f}, R2: {np.mean(r2_scores):.3f}')

# Save model performances to a JSON file
with open('model_imputation_performances.json', 'w') as f:
    json.dump(model_scores, f, indent=4)

# Select the best model
best_model_name = min(model_scores, key=lambda x: model_scores[x]['MAE'])
best_model = models[best_model_name]
print(f'Best model: {best_model_name}')

# Train the best model on the entire training data
best_model.fit(X_train, y_train)

# Impute the missing LotFrontage values
df.loc[df['LotFrontage'].isnull(), 'LotFrontage'] = best_model.predict(X_test)

nominal_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'Electrical', 'Functional', 'GarageType', 'GarageFinish', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
df = pd.get_dummies(df, columns=nominal_cols)

# Feature Engineering
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']

df.drop(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], axis=1, inplace=True)

# Additional Feature Engineering
df['TotalBath'] = df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5 + df['FullBath'] + df['HalfBath'] * 0.5
df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

# Data Cleaning (without SalePrice)
# Remove values below threshold
# df = df[df['SalePrice'] <= 450000]
# df = df[df['SalePrice'] >= 50000]

# Save the cleaned dataframe
df.to_csv("house_prices_cleaned_v9_unfeatured.csv", index=False)

# Check the cleaned dataframe
print(df.info())
print(df.head())
