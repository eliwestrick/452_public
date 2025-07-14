
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from pyswarms.single.global_best import GlobalBestPSO
import torch.nn as nn
import torch.nn.functional as F


node_names = ["lhjc", "left_hip", "lsjc", "lejc", "lkjc", "lajc", "lwjc", "rhjc", "right_hip", "rsjc", "rejc", "rkjc", "rajc", "rwjc"]
#for the visualization
position_log = []



#reshape into tensors
def reshape_biomech_df(df, node_names, swing_col = "session_swing", time_col = "time"):
    swings, targets = [], []
    for swing_id, group in df.groupby(swing_col):
        group = group.sort_values(time_col)
        node_features = [group[[f"{node}_x", f"{node}_y", f"{node}_z"]].values.T for node in node_names]
        swing_tensor = np.stack(node_features, axis = 0).transpose(2, 0, 1)
        swings.append(swing_tensor)
        row = group.iloc[0]
        targets.append([row["exit_velo_mph_x"], row["la"], row["dist"]])
    return np.stack(swings), np.array(targets)

df = pd.read_excel(r"C:\Users\eliwe\Documents\Data_files\data_resampled.xlsx")
swing_tensor, swing_targets = reshape_biomech_df(df, node_names)

#used a package to make test train split
X_temp, X_test, y_temp, y_test = train_test_split(swing_tensor, swing_targets, test_size = 0.2, random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25, random_state = 42)

class SwingDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(SwingDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(SwingDataset(X_val, y_val), batch_size=32)

#define the model
class GNN_GRU_Model(nn.Module):
    def __init__(self, input_dim = 3, num_nodes = 14, hidden_dim = 64, num_layers = 2, output_dim = 3):
        super().__init__()
        self.gnn_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) if i == 0 else nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.gru = nn.GRU(input_size = hidden_dim * num_nodes, hidden_size = hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        B, T, N, D = x.shape
        x = x.view(B * T, N, D)
        for layer in self.gnn_layers:
            x = F.relu(layer(x))
        x = x.view(B, T, -1)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

def build_gnn_model(num_nodes, num_layers):
    return GNN_GRU_Model(hidden_dim = num_nodes, num_layers = num_layers)

def train_and_evaluate(model, optimizer, train_loader, val_loader, epochs = 10):
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

def objective_function(hyperparams):
    global position_log
    position_log.append(hyperparams.copy())  # Log current positions

    losses = []
    for p in hyperparams:
        num_nodes = int(p[0])
        lr = float(p[1])
        num_layers = int(round(p[2]))

        model = build_gnn_model(num_nodes=num_nodes, num_layers = num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        loss = train_and_evaluate(model, optimizer, train_loader, val_loader, epochs = 10)
        losses.append(loss)
    return np.array(losses)


#run pyswarms 
bounds = (np.array([16, 0.001, 2]), np.array([256, 0.1, 3]))
optimizer = GlobalBestPSO(n_particles = 8, dimensions = 3, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, bounds = bounds)
best_cost, best_pos = optimizer.optimize(objective_function, iters = 10)


import matplotlib.pyplot as plt

#make it a numpy array for easier plotting
position_log_np = np.array(position_log)

params = ["Nodes per Layer", "Learning Rate", "Number of Layers"]
fig, axes = plt.subplots(1, 3, figsize = (15, 4))

for i in range(3):
    param_vals = position_log_np[:, :, i]  
    for particle in range(param_vals.shape[1]):
        axes[i].plot(param_vals[:, particle], alpha = 0.6)
    axes[i].set_title(params[i])
    axes[i].set_xlabel("Iteration")
    axes[i].set_ylabel("Value")
    axes[i].grid(True)

plt.suptitle("Hyperparameter Evolution Across Particles")
plt.tight_layout()
plt.show()



print(f"nodes: {int(best_pos[0])}")
print(f"lr: {best_pos[1]:.5f}")
print(f"layers: {int(round(best_pos[2]))}")
