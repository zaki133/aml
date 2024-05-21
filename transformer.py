import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
print(torch.cuda.is_available())

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, ModelConfig, TrainerConfig

# Load the cleaned dataset
df = pd.read_csv('house_prices_cleaned_v2.csv')

# Separate features and target
X = df.drop(columns=['SalePrice', 'Id'])
y = df['SalePrice']

# Identify categorical and numerical features
cat_features = X.select_dtypes(include='object').columns.tolist()
num_features = X.select_dtypes(include=[np.number]).columns.tolist()

# Encode categorical features
for col in cat_features:
    X[col] = LabelEncoder().fit_transform(X[col])

# Standardize numerical features
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# Combine features and target into one DataFrame for pytorch-tabular
df_combined = X.copy()
df_combined['SalePrice'] = y

# Train-test split
train_df, val_df = train_test_split(df_combined, test_size=0.2, random_state=42)

# Define the data configuration
data_config = DataConfig(
    target=["SalePrice"],
    continuous_cols=num_features,
    categorical_cols=cat_features
)

# Define the model configuration
model_config = ModelConfig(
    task="regression",
    model_name="tab_transformer",
    embed_categorical=True,
    embed_continuous=True,
    embedding_dim=32,
    attention_dim=32,
    num_heads=4,
    num_attn_blocks=2,
    transformer_activation="relu",
    dropout=0.1
)

# Define the trainer configuration
trainer_config = TrainerConfig(
    max_epochs=100,
    gpus=1 if torch.cuda.is_available() else 0,
)

# Initialize the TabularModel
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    trainer_config=trainer_config,
)

# Train the model
tabular_model.fit(train=train_df, validation=val_df)

# Predict on the validation set
val_pred = tabular_model.predict(val_df)
val_pred = val_pred['SalePrice_PREDICTION'].values

# Calculate MAPE
val_mape = mean_absolute_percentage_error(val_df['SalePrice'], val_pred) * 100
print(f'Validation MAPE with TabTransformer: {val_mape:.4f}%')
