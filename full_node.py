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
df = pd.read_csv('house_prices_cleaned.csv')

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

# Define a NODE Block
class NODEBlock(nn.Module):
    def __init__(self, input_dim, num_trees, tree_depth, output_dim):
        super(NODEBlock, self).__init__()
        self.num_trees = num_trees
        self.tree_depth = tree_depth

        # Define the layers
        self.layers = nn.ModuleList()
        for _ in range(num_trees):
            tree_layers = nn.Sequential()
            for depth in range(tree_depth):
                tree_layers.add_module(f"tree_layer_{depth}", nn.Linear(input_dim if depth == 0 else 2**depth, 2**(depth+1)))
                tree_layers.add_module(f"tree_relu_{depth}", nn.ReLU())
            self.layers.append(tree_layers)

        self.output_layer = nn.Linear(num_trees * 2**tree_depth, output_dim)

    def forward(self, x):
        # Pass input through each tree
        tree_outputs = []
        for tree in self.layers:
            tree_outputs.append(tree(x))

        # Concatenate the outputs from all trees
        x = torch.cat(tree_outputs, dim=1)
        return self.output_layer(x)

# Define the NODE Network
class NODE(nn.Module):
    def __init__(self, input_dim, num_trees, tree_depth, output_dim):
        super(NODE, self).__init__()
        self.node_block = NODEBlock(input_dim, num_trees, tree_depth, output_dim)

    def forward(self, x):
        return self.node_block(x)

# Function to train and evaluate the model
def train_and_evaluate(input_dim, num_trees, tree_depth, learning_rate, batch_size):
    model = NODE(input_dim, num_trees, tree_depth, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = mape_loss

    train_loader = torch.utils.data.DataLoader(dataset=list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False)

    epochs = 4000
    best_val_loss = float('inf')
    patience = 20
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
            print(best_val_loss)
            break

    return best_val_loss.item()

# Hyperparameter grid
num_trees_grid = [5, 10, 20]
tree_depth_grid = [3, 4, 5]
learning_rate_grid = [0.0001, 0.001]
batch_size_grid = [32, 64]

# Grid search
best_params = None
best_val_loss = float('inf')

input_dim = X_train.shape[1]  # Define input_dim

for num_trees, tree_depth, learning_rate, batch_size in tqdm(itertools.product(num_trees_grid, tree_depth_grid, learning_rate_grid, batch_size_grid), desc="Hyperparameter Tuning", total=len(num_trees_grid) * len(tree_depth_grid) * len(learning_rate_grid) * len(batch_size_grid)):
    val_loss = train_and_evaluate(input_dim, num_trees, tree_depth, learning_rate, batch_size)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = (num_trees, tree_depth, learning_rate, batch_size)

print(f'Best params: {best_params}')
print(f'Best validation loss: {best_val_loss}')

# Train the final model with the best hyperparameters
best_num_trees, best_tree_depth, best_learning_rate, best_batch_size = best_params
model = NODE(input_dim, best_num_trees, best_tree_depth, 1).to(device)
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
