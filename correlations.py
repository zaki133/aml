import pandas as pd

# Load the dataset
df = pd.read_csv("house_prices_cleaned.csv")

# Compute the correlation matrix
corr_matrix = df.corr()

# Get correlations with the SalePrice column
sale_price_corr = corr_matrix['SalePrice']

# Filter for correlations greater than 0.6
high_corr = sale_price_corr[abs(sale_price_corr) > 0.6]

# Display the filtered correlations
print(high_corr)
