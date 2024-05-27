import pandas as pd

# Setting display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

# Load the dataset
# df = pd.read_csv("house_prices_cleaned.csv")
df2 = pd.read_csv("house_prices_cleaned_v5.csv")
# Display the full DataFrame information
# print(df.dtypes)
# print(df.info())
print(df2.dtypes)
print(df2.info())
print(df2.describe())
