import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('dataset.csv')

# Step 1: Handling Missing Values for columns that we dont know anything about..
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

# Classifier with tuned parameters
clf = svm.SVR(kernel='rbf', C=100, gamma=0.001)

# Set initial scores
acc = 0
acc1 = 0
acc2 = 0

# Define k-fold object for 10-fold validation
kf = KFold(n_splits=10, shuffle=True, random_state=3)

# Main evaluator loop over the 10 folds
for trn, tst in kf.split(train_data):
    # Compute benchmark score prediction based on mean neighbourhood LotFrontage
    fold_train_samples = train_data.iloc[trn]
    fold_test_samples = train_data.iloc[tst]
    neigh_means = fold_train_samples.groupby('Neighborhood')['LotFrontage'].mean()
    all_mean = fold_train_samples['LotFrontage'].mean()
    y_pred_neigh_means = fold_test_samples.join(neigh_means, on='Neighborhood', lsuffix='benchmark')['LotFrontage']
    y_pred_all_mean = [all_mean] * fold_test_samples.shape[0]

    # Compute benchmark score prediction based on overall mean LotFrontage
    acc1 += mean_absolute_error(fold_test_samples['LotFrontage'], y_pred_neigh_means)
    acc2 += mean_absolute_error(fold_test_samples['LotFrontage'], y_pred_all_mean)

    # Perform model fitting
    clf.fit(X_train.iloc[trn], y_train.iloc[trn])

    # Record all scores for averaging
    acc += mean_absolute_error(fold_test_samples['LotFrontage'], clf.predict(X_train.iloc[tst]))

print('10-Fold Validation Mean Absolute Error results:')
print(f'\tSVR: {acc / 10:.3f}')
print(f'\tSingle mean: {acc2 / 10:.3f}')
print(f'\tNeighbourhood mean: {acc1 / 10:.3f}')

# Final imputation on the full dataset
clf.fit(X_train, y_train)
df.loc[df['LotFrontage'].isnull(), 'LotFrontage'] = clf.predict(X_test)

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

# Step 4: Scaling Numerical Features
numerical_features = df.select_dtypes(include=[np.number])

# Data Cleaning (without SalePrice)

# # Replace outlier with next lower value
# max_lot_frontage = df['LotFrontage'].max()
# second_max_value = df['LotFrontage'][df['LotFrontage'] != max_lot_frontage].max()
# df['LotFrontage'] = df['LotFrontage'].replace(max_lot_frontage, second_max_value)


# # Remove the 2 highest outliers and replace them with the next lower value
# top_two_values = df['MasVnrArea'].nlargest(2).unique()
# next_lower_value = df['MasVnrArea'][~df['MasVnrArea'].isin(top_two_values)].max()
# df['MasVnrArea'] = df['MasVnrArea'].apply(lambda x: next_lower_value if x in top_two_values else x)

# # Remove highest outlier
# max_value = df['BsmtFinSF1'].max()
# df = df[df['BsmtFinSF1'] != max_value]

# # Replace highest outlier with next lower value
# max_value = df['BsmtFinSF2'].max()
# second_max_value = df['BsmtFinSF2'][df['BsmtFinSF2'] != max_value].max()
# df['BsmtFinSF2'] = df['BsmtFinSF2'].replace(max_value, second_max_value)

# # Remove highest outlier
# max_value = df['TotalBsmtSF'].max()
# df = df[df['TotalBsmtSF'] != max_value]

# # Remove highest outlier
# max_value = df['1stFlrSF'].max()
# df = df[df['1stFlrSF'] != max_value]

# Remove values below threshold
df = df[df['LotArea'] <= 100000]
df = df[df['GrLivArea'] <= 4000]
df = df[df['BsmtFullBath'] <= 2]
df = df[df['BedroomAbvGr'] <= 7]
df = df[df['KitchenAbvGr'] < 3]
df = df[df['Fireplaces'] < 3]
df = df[df['OpenPorchSF'] < 400]
df = df[df['EnclosedPorch'] < 400]
df = df[df['3SsnPorch'] < 400]
df = df[df['MiscVal'] < 500]
df = df[df['TotalPorchSF'] < 800]

# Save the cleaned dataframe
df.to_csv("house_prices_cleaned_v7_unscaled.csv", index=False)

# Check the cleaned dataframe
print(df.info())
print(df.head())
