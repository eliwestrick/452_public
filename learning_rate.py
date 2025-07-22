
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from pyswarms.single.global_best import GlobalBestPSO
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Node Definitions
# these are the anatomical landmarks tracked in the motion capture data
node_names = [
    "lhjc", "left_hip", "lsjc", "lejc", "lkjc", "lajc", "lwjc",
    "rhjc", "right_hip", "rsjc", "rejc", "rkjc", "rajc", "rwjc"
]

# Particle Position Log for Visualization
# stores particle positions from PSO so we can visualize hyperparameter evolution
position_log = []

# Reshape Raw Data into Tensors
# converts raw dataframe into swing tensors and target labels
def reshape_biomech_df(df, node_names, swing_col="session_swing", time_col="time"):
    swings, targets = [], []
    for swing_id, group in df.groupby(swing_col):  # group by swing
        group = group.sort_values(time_col)  # sort by time
        # extract 3D coordinates for each node across time
        node_features = [group[[f"{node}_x", f"{node}_y", f"{node}_z"]].values.T for node in node_names]
        swing_tensor = np.stack(node_features, axis=0).transpose(2, 0, 1)  # shape: (T, N, 3)
        swings.append(swing_tensor)
        row = group.iloc[0]  # use first row to get target labels
        targets.append([row["exit_velo_mph_x"], row["la"], row["dist"]])
    return np.stack(swings), np.array(targets)

# Load and Reshape Data
df = pd.read_excel(r"C:\Users\eliwe\Documents\Data_files\data_resampled.xlsx")
swing_tensor, swing_targets = reshape_biomech_df(df, node_names)

# Train/Val/Test Split
# 60/20/20 split for training, validation, and testing
X_temp, X_test, y_temp, y_test = train_test_split(swing_tensor, swing_targets, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# PyTorch Dataset Wrapper
# wraps swing tensors and targets into a PyTorch dataset
class SwingDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # convert to torch tensors
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)  # number of swings
    def __getitem__(self, idx): return self.X[idx], self.y[idx]  # return one swing and its target

#DataLoaders for Batching
# creates train and val dataloaders for batching and shuffling
train_loader = DataLoader(SwingDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(SwingDataset(X_val, y_val), batch_size=32)

#GNN-GRU Hybrid Model with LRP
# defines the model that encodes spatial and temporal features and predicts swing outcomes
class GNN_GRU_Model(nn.Module):
    def __init__(self, input_dim=3, num_nodes=14, hidden_dim=64, num_layers=2, output_dim=3):
        super().__init__()
        # spatial encoder: stacked linear layers applied to each node
        self.gnn_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) if i == 0 else nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        # temporal encoder: GRU processes node embeddings over time
        self.gru = nn.GRU(input_size=hidden_dim * num_nodes, hidden_size=hidden_dim, batch_first=True)
        # final layer: maps GRU output to 3 swing metrics
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        B, T, N, D = x.shape  # batch, time, nodes, dimensions
        x = x.view(B * T, N, D)  # flatten time and batch for node-wise processing
        for layer in self.gnn_layers:
            x = F.relu(layer(x))  # apply spatial transformation
        x = x.view(B, T, -1)  # reshape for GRU input
        _, h = self.gru(x)  # GRU returns hidden state
        return self.fc(h.squeeze(0))  # final prediction

    def relevance_propagation(self, x, y_pred):
        # estimates relevance of each node's input features to the prediction
        # returns a tensor of shape (B, T, N, D)
        B, T, N, D = x.shape
        relevances = []
        for b in range(B):
            sample_relevance = []
            for t in range(T):
                relevance = y_pred[b].unsqueeze(0).unsqueeze(1) * x[b, t]
                sample_relevance.append(relevance)
            relevances.append(torch.stack(sample_relevance))
        return relevances

# Model Builder for PSO 
# builds a model with given hyperparameters
def build_gnn_model(num_nodes, num_layers):
    return GNN_GRU_Model(hidden_dim=num_nodes, num_layers=num_layers)

#Training and Evaluation Loop
# trains model for a few epochs and returns validation loss (used for PSO)
def train_and_evaluate(model, optimizer, train_loader, val_loader, epochs=10):
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            val_loss += loss_fn(output, y_batch).item()
    return val_loss / len(val_loader)

# Objective Function for PSO
# evaluates each particle's hyperparameter combo and returns its validation loss
def objective_function(hyperparams):
    global position_log
    position_log.append(hyperparams.copy())  # log current particle positions
    losses = []
    for p in hyperparams:
        num_nodes = int(p[0])
        lr = float(p[1])
        num_layers = int(round(p[2]))
        model = build_gnn_model(num_nodes=num_nodes, num_layers=num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss = train_and_evaluate(model, optimizer, train_loader, val_loader, epochs=10)
        losses.append(loss)
    return np.array(losses)

# Run PSO to Find Best Hyperparameters
# runs particle swarm optimization to find best combo of nodes, layers, and learning rate
bounds = (np.array([16, 0.001, 2]), np.array([256, 0.1, 3]))
optimizer = GlobalBestPSO(n_particles=8, dimensions=3, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, bounds=bounds)
best_cost, best_pos = optimizer.optimize(objective_function, iters=10)

#Visualization of Hyperparameter Evolution
# plots how each particle explored the hyperparameter space over time
position_log_np = np.array(position_log)
params = ["Nodes per Layer", "Learning Rate", "Number of Layers"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i in range(3):
    param_vals = position_log_np[:, :, i]
    for particle in range(param_vals.shape[1]):
        axes[i].plot(param_vals[:, particle], alpha=0.6)
    axes[i].set_title(params[i])
    axes[i].set_xlabel("Iteration")
    axes[i].set_ylabel("Value")
    axes[i].grid(True)

plt.suptitle("Hyperparameter Evolution Across Particles")
plt.tight_layout()
plt.show()

# Print Best Hyperparameters Found
print(f"nodes: {int(best_pos[0])}")
print(f"lr: {best_pos[1]:.5f}")
print(f"layers: {int(round(best_pos[2]))}")


def train_and_validate(model, train_loader, val_loader, optimizer, epochs=50, return_avg_val_loss_only=False):
    criterion = nn.MSELoss()  # Mean squared error for training
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    train_targets_over_epochs, train_preds_over_epochs = [], []
    val_targets_over_epochs, val_preds_over_epochs = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_targets, epoch_train_preds = [], []
        total_train_loss, total_train_mae = 0, 0

        # Training Loop
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_mae += F.l1_loss(output, y_batch).item()
            epoch_train_targets.append(y_batch.cpu().numpy())
            epoch_train_preds.append(output.detach().cpu().numpy())

        train_targets_over_epochs.append(epoch_train_targets)
        train_preds_over_epochs.append(epoch_train_preds)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_mae = total_train_mae / len(train_loader)
        train_losses.append(avg_train_loss)
        train_maes.append(avg_train_mae)

        # Validation Loop 
        model.eval()
        epoch_val_targets, epoch_val_preds = [], []
        total_val_loss, total_val_mae = 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = criterion(output, y_batch)
                total_val_loss += loss.item()
                total_val_mae += F.l1_loss(output, y_batch).item()
                epoch_val_targets.append(y_batch.cpu().numpy())
                epoch_val_preds.append(output.cpu().numpy())

        val_targets_over_epochs.append(epoch_val_targets)
        val_preds_over_epochs.append(epoch_val_preds)

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_mae = total_val_mae / len(val_loader)
        val_losses.append(avg_val_loss)
        val_maes.append(avg_val_mae)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f}, MAE: {avg_train_mae:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.4f}")

    if return_avg_val_loss_only:
        return avg_val_loss

    return train_losses, val_losses, train_maes, val_maes, train_targets_over_epochs, train_preds_over_epochs, val_targets_over_epochs, val_preds_over_epochs



best_model = build_gnn_model(num_nodes=151, num_layers=3)


best_optimizer = torch.optim.Adam(best_model.parameters(), lr=0.07452)


train_losses, val_losses, train_maes, val_maes, \
train_targets_over_epochs, train_preds_over_epochs, \
val_targets_over_epochs, val_preds_over_epochs = train_and_validate(
    best_model,
    train_loader,
    val_loader,
    best_optimizer,
    epochs=50,
    return_avg_val_loss_only=False
)


test_loader = DataLoader(SwingDataset(X_test, y_test), batch_size=32)


best_model.eval()
total_test_loss, total_test_mae = 0, 0

with torch.no_grad():  
    for X_batch, y_batch in test_loader:
        output = best_model(X_batch)
        total_test_loss += F.mse_loss(output, y_batch).item()
        total_test_mae  += F.l1_loss(output, y_batch).item()

avg_test_loss = total_test_loss / len(test_loader)
avg_test_mae  = total_test_mae / len(test_loader)


print("\n=== Final Test Set Performance ===")
print(f"Test Loss (MSE): {avg_test_loss:.4f}")
print(f"Test MAE       : {avg_test_mae:.4f}")
