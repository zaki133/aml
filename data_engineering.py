import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('dataset.csv')

# Step 1: Handling Missing Values
df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)

# Fill missing values for specific columns
imputer_median = SimpleImputer(strategy='median')
df['LotFrontage'] = imputer_median.fit_transform(df[['LotFrontage']])
df['MasVnrArea'] = imputer_median.fit_transform(df[['MasVnrArea']])
df['GarageYrBlt'] = imputer_median.fit_transform(df[['GarageYrBlt']])

# Fill missing values for categorical columns with mode
imputer_mode = SimpleImputer(strategy='most_frequent')
categorical_cols = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
df[categorical_cols] = imputer_mode.fit_transform(df[categorical_cols])

# Step 2: Encoding Categorical Variables
label_enc = LabelEncoder()
ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
for col in ordinal_cols:
    df[col] = label_enc.fit_transform(df[col])

df['CentralAir'] = df['CentralAir'].map({'Y': 1, 'N': 0})

nominal_cols = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'Electrical', 'Functional', 'GarageType', 'GarageFinish', 'PavedDrive', 'SaleType', 'SaleCondition']
df = pd.get_dummies(df, columns=nominal_cols)

# Step 3: Feature Engineering
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']

df.drop(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], axis=1, inplace=True)

# Additional Feature Engineering
df['TotalBath'] = df['BsmtFullBath'] + df['BsmtHalfBath']*0.5 + df['FullBath'] + df['HalfBath']*0.5
df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

# Step 4: Scaling Numerical Features
numerical_features = df.select_dtypes(include=[np.number])

# # Log transformation on skewed numerical features
# skewed_cols = numerical_features.apply(lambda x: x.skew()).sort_values(ascending=False)
# skewness_threshold = 0.75
# high_skewness = skewed_cols[skewed_cols > skewness_threshold]

# for col in high_skewness.index:
#     df[col] = np.log1p(df[col])  # Using log1p to handle zeros

# # Step 5: Scaling Numerical Features
# scaler = StandardScaler()
# df[numerical_features.columns] = scaler.fit_transform(df[numerical_features.columns])

# Save the cleaned dataframe
df.to_csv("house_prices_cleaned_v5_unscaled.csv", index=False)

# Check the cleaned dataframe
print(df.info())
print(df.head())
