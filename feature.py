import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Load the cleaned dataset
df = pd.read_csv('house_prices_cleaned_v2.csv')

# Feature Engineering: Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
numeric_features = df.select_dtypes(include=[np.number]).drop(columns=['SalePrice', 'Id'])
poly_features = poly.fit_transform(numeric_features)
poly_features_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numeric_features.columns))

# Combine with original dataset
df_poly = pd.concat([df.reset_index(drop=True), poly_features_df], axis=1)
df_poly.drop(columns=numeric_features.columns, inplace=True)

# Save the new dataset
df_poly.to_csv('house_prices_poly.csv', index=False)
