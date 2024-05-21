import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
df = pd.read_csv('dataset.csv')

# Step 1: Handling Missing Values
# Columns with more than 20% missing values
df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)

# Columns with missing values less than 5%
for col in ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']:
    df[col] = df[col].fillna(df[col].median())

# Columns with missing values between 5% and 20%
for col in ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    df[col] = df[col].fillna(df[col].mode()[0])

# Step 2: Encoding Categorical Variables
# Label Encoding for ordinal variables
label_enc = LabelEncoder()
for col in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']:
    df[col] = label_enc.fit_transform(df[col])

# Encoding binary variable
df['CentralAir'] = df['CentralAir'].map({'Y': 1, 'N': 0})

# One-Hot Encoding for nominal variables
df = pd.get_dummies(df, columns=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'Electrical', 'Functional', 'GarageType', 'GarageFinish', 'PavedDrive', 'SaleType', 'SaleCondition'])

# Step 3: Feature Engineering
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']

# Drop the original columns after feature engineering
df.drop(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], axis=1, inplace=True)

# Ensure no remaining missing values
if df.isnull().sum().any():
    print("There are still missing values in the dataframe.")
else:
    print("All missing values have been handled.")

# Step 4: Scaling Numerical Features
#numerical_features = df.select_dtypes(include=[np.number])
#scaler = StandardScaler()
#df[numerical_features.columns] = scaler.fit_transform(numerical_features)

# Save the cleaned dataframe
df.to_csv("house_prices_cleaned_v2.csv", index=False)

# Check the cleaned dataframe
print(df.info())
print(df.head())
