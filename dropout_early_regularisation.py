import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import itertools
from tqdm import tqdm

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Convert scaled data back to tensors and move to GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

# Define custom MAPE loss function
def mape_loss(output, target):
    return torch.mean(torch.abs((target - output) / target)) * 100

# Define the neural network with adjustable parameters
class HousePriceNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout=0.2):
        super(HousePriceNN, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Function to train and evaluate the model
def train_and_evaluate(input_dim, hidden_dims, dropout, learning_rate, batch_size):
    model = HousePriceNN(input_dim, hidden_dims, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = mape_loss

    train_loader = torch.utils.data.DataLoader(dataset=list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False)

    epochs = 2000
    best_val_loss = float('inf')
    patience = 50
    trigger_times = 0

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1

        if trigger_times >= patience:
            break

    return best_val_loss.item()

# Hyperparameter grid
hidden_dims_grid = [[256, 128, 64, 32], [512, 256, 128], [128, 64, 32]]
dropout_grid = [0.1, 0.2, 0.3]
learning_rate_grid = [0.0001, 0.001]
batch_size_grid = [32, 64]

# Grid search
best_params = None
best_val_loss = float('inf')

input_dim = X_train.shape[1]  # Define input_dim

for hidden_dims, dropout, learning_rate, batch_size in tqdm(itertools.product(hidden_dims_grid, dropout_grid, learning_rate_grid, batch_size_grid), desc="Hyperparameter Tuning", total=len(hidden_dims_grid) * len(dropout_grid) * len(learning_rate_grid) * len(batch_size_grid)):
    val_loss = train_and_evaluate(input_dim, hidden_dims, dropout, learning_rate, batch_size)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = (hidden_dims, dropout, learning_rate, batch_size)

print(f'Best params: {best_params}')
print(f'Best validation loss: {best_val_loss}')

# Train the final model with the best hyperparameters
best_hidden_dims, best_dropout, best_learning_rate, best_batch_size = best_params
model = HousePriceNN(input_dim, best_hidden_dims, best_dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=best_learning_rate)
criterion = mape_loss

train_loader = torch.utils.data.DataLoader(dataset=list(zip(X_train, y_train)), batch_size=best_batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=list(zip(X_val, y_val)), batch_size=best_batch_size, shuffle=False)

epochs = 5000
best_val_loss = float('inf')
patience = 200
trigger_times = 0

for epoch in tqdm(range(epochs), desc="Final Training"):
    model.train()
    for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

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
    val_loss = criterion(val_outputs, y_val)

print(f'Final Validation MAPE: {val_loss.item():.4f}%')
