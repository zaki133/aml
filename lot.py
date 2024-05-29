import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('dataset.csv')

# Step 1: Handling Missing Values for other columns
imputer_median = SimpleImputer(strategy='median')
df['MasVnrArea'] = imputer_median.fit_transform(df[['MasVnrArea']])
df['GarageYrBlt'] = imputer_median.fit_transform(df[['GarageYrBlt']])

imputer_mode = SimpleImputer(strategy='most_frequent')
categorical_cols = ['MasVnrType', 'Electrical']
df[categorical_cols] = imputer_mode.fit_transform(df[categorical_cols])

# Fill meaningful NaN values
df['Alley'].fillna('NoAlley', inplace=True)
df['PoolQC'].fillna('NoPool', inplace=True)
df['Fence'].fillna('NoFence', inplace=True)
df['MiscFeature'].fillna('None', inplace=True)
df['BsmtQual'].fillna('NoBasement', inplace=True)
df['BsmtCond'].fillna('NoBasement', inplace=True)
df['BsmtExposure'].fillna('NoBasement', inplace=True)
df['BsmtFinType1'].fillna('NoBasement', inplace=True)
df['BsmtFinType2'].fillna('NoBasement', inplace=True)
df['FireplaceQu'].fillna('NoFireplace', inplace=True)
df['GarageType'].fillna('NoGarage', inplace=True)
df['GarageFinish'].fillna('NoGarage', inplace=True)
df['GarageQual'].fillna('NoGarage', inplace=True)
df['GarageCond'].fillna('NoGarage', inplace=True)
df['MasVnrType'].fillna('None', inplace=True)

# Step 2: Encoding Categorical Variables
label_enc = LabelEncoder()
ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
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

# Save the dataset with imputed LotFrontage
df.to_csv('house_prices_cleaned_v6_unscaled.csv', index=False)

# Check the cleaned dataframe
print(df.info())
print(df.head())
