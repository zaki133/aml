import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# Load the cleaned dataset
df = pd.read_csv('house_prices_cleaned_v6_unscaled.csv')

# Separate features and target
X = df.drop(columns=['SalePrice', 'Id'])
y = df['SalePrice']

# Ensure all columns are numeric
for col in X.select_dtypes(include='object').columns:
    if len(X[col].unique()) <= 10:  # Label encode for columns with fewer unique values
        X[col] = LabelEncoder().fit_transform(X[col])
    else:  # One-hot encode for columns with more unique values
        X = pd.get_dummies(X, columns=[col], drop_first=True)

# Convert to numpy arrays
X = X.values
y = y.values

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert scaled data back to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# Define the neural network with adjusted dropout rate
class HousePriceNN(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Adjusted dropout rate
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Adjusted dropout rate
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Adjusted dropout rate
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize the model
input_dim = X_train.shape[1]
model = HousePriceNN(input_dim)

# Define custom MAPE loss function
def mape_loss(output, target):
    return torch.mean(torch.abs((target - output) / target)) * 100

# Define optimizer with a different learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with early stopping and learning rate scheduler
epochs = 5000
best_val_loss = float('inf')
patience = 200
trigger_times = 0

for epoch in range(epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train)
    loss = mape_loss(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = mape_loss(val_outputs, y_val)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
    else:
        trigger_times += 1
        
    if trigger_times >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Evaluate the model on validation set
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    val_loss = mape_loss(val_outputs, y_val)

print(f'Final Validation MAPE: {val_loss.item():.4f}%')
