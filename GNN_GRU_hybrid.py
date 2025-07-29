# This script prepares biomechanical swing data, defines a GNN-GRU model,
# and trains + evaluates it on predicting exit velocity.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# list of joint/node names in the biomechanical model
node_names = [
    "lhjc", "left_hip", "lsjc", "lejc", "lkjc", "lajc", "lwjc",
    "rhjc", "right_hip", "rsjc", "rejc", "rkjc", "rajc", "rwjc"
]

# edges define the spatial dependancies between nodes
edges = [
    ("lsjc", "lejc"), ("lejc", "lwjc"), ("lwjc", "lhjc"),
    ("rsjc", "rejc"), ("rejc", "rwjc"), ("rwjc", "rhjc"),
    ("left_hip", "lkjc"), ("lkjc", "lajc"),
    ("right_hip", "rkjc"), ("rkjc", "rajc"),
    ("left_hip", "lsjc"), ("right_hip", "rsjc"),
    ("lsjc", "rsjc"), ("left_hip", "right_hip")
]

# map node names to integer indices for tensor indexing
node_to_idx = {name: i for i, name in enumerate(node_names)}

# convert edge pairs into index-based tuples
edge_index = [(node_to_idx[a], node_to_idx[b]) for a, b in edges]

#turns the raw data into a tensor of shape (num_swings, time_steps, num_nodes, 3)
def reshape_biomech_df(df, node_names, swing_col="session_swing", time_col="time"):

    swings, targets = [], []

    #iterate over each swing session
    for swing_id, group in df.groupby(swing_col):
        #sort frames by timestamp
        group = group.sort_values(time_col)

        #collect (3,) features for each node over time
        node_features = [
            group[[f"{node}_x", f"{node}_y", f"{node}_z"]].values.T
            for node in node_names
        ]

        # stack into (num_nodes, 3, time_steps) then transpose to (time_steps, num_nodes, 3)
        swing_tensor = np.stack(node_features, axis=0).transpose(2, 0, 1)
        swings.append(swing_tensor)

        # target vector is the first exit velocity reading for that swing
        targets.append([group.iloc[0]["exit_velo_mph_x"]])

    # return full dataset arrays
    return np.stack(swings), np.array(targets)


#load and reshape the biomechanical data
df = pd.read_excel(r"C:\Users\eliwe\Documents\Data_files\data_resampled.xlsx")
swing_tensor, swing_targets = reshape_biomech_df(df, node_names)

#split into train / validation / test
X_temp, X_test, y_temp, y_test = train_test_split(
    swing_tensor, swing_targets, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

#dataloader requires a pytorch dataset wrapper
#so we have to wrap the numpy arrays
class SwingDataset(Dataset):
   
    def __init__(self, X, y):
        # convert to float32 tensors for model consumption
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # return (features, target) pair
        return self.X[idx], self.y[idx]


#create DataLoaders for batching
train_loader = DataLoader(SwingDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(SwingDataset(X_val,   y_val),   batch_size=32)
test_loader  = DataLoader(SwingDataset(X_test,  y_test),  batch_size=32)



#this will project the raw node features into a hidden space,
# then run through a few GNN layers for spatial attention,
# then flatten and creat a final linear layer to predict the single exit velocity.
class GNN_GRU_Model(nn.Module):
  
    def __init__(self, input_dim=3, num_nodes=14, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        # initial linear projection of 3D coords to hidden space
        self.node_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers: same hidden size in and out
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # GRU handles sequences of length T with flattened node features
        self.gru = nn.GRU(
            input_size=hidden_dim * num_nodes,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # final head predicts a single regression value
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        
        B, T, N, D = x.shape

        # project each node feature from D -> hidden_dim
        x = self.node_proj(x)

        # spatial GNN layers
        for layer in self.gnn_layers:
            # buffer for new features
            x_new = torch.zeros_like(x)
            # loop edges to aggregate messages
            for i, j in edge_index:
                # message from j to i
                x_new[:, :, i] += layer(x[:, :, j])
                # message from i to j (undirected)
                x_new[:, :, j] += layer(x[:, :, i])
            # apply nonlinearity
            x = F.relu(x_new)

        # flatten node dimension for GRU: (B, T, N*hidden_dim)
        x_flat = x.view(B, T, -1)

        # get final hidden state from GRU
        _, h = self.gru(x_flat)

        # map hidden state to output
        return self.fc(h.squeeze(0))

#this will run the training loop with MSE loss, and 
#do eval on the validatoin set after each epoch
def train_and_validate(model, train_loader, val_loader, optimizer, edge_index, epochs=50):
   
    mse_crit = nn.MSELoss()
    mae_crit = nn.L1Loss()

    for epoch in range(epochs):
        model.train()
        train_mse, train_mae = 0.0, 0.0

        # training phase
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch, edge_index)           # forward pass
            loss_mse = mse_crit(preds, y_batch)          # compute MSE
            loss_mae = mae_crit(preds, y_batch)          # compute MAE
            loss_mse.backward()                          # backprop
            optimizer.step()                             # update weights

            train_mse += loss_mse.item()
            train_mae += loss_mae.item()

        model.eval()
        val_mse, val_mae = 0.0, 0.0

        # validation phase
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch, edge_index)
                val_mse += mse_crit(preds, y_batch).item()
                val_mae += mae_crit(preds, y_batch).item()

        # compute averages over batches
        n_train = len(train_loader)
        n_val   = len(val_loader)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train MSE: {train_mse/n_train:.4f} | Train MAE: {train_mae/n_train:.4f} | "
            f"Val MSE: {val_mse/n_val:.4f} | Val MAE: {val_mae/n_val:.4f}"
        )


# set hyperparameters from prior tuning
best_hidden_dim   = 114
best_learning_rate = 0.0819337
best_num_layers    = 2

# instantiate model and optimizer
model = GNN_GRU_Model(
    num_nodes=len(node_names),
    hidden_dim=best_hidden_dim,
    num_layers=best_num_layers
)
optimizer = torch.optim.Adam(model.parameters(), lr=best_learning_rate)

# run the training + validation loop
train_and_validate(model, train_loader, val_loader, optimizer, edge_index, epochs=50)

# evaluate on the test split
model.eval()
test_mse, test_mae = 0.0, 0.0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch, edge_index)
        test_mse += nn.MSELoss()(preds, y_batch).item()
        test_mae += nn.L1Loss()(preds, y_batch).item()

n_test = len(test_loader)
print(f"\nTest MSE: {test_mse/n_test:.4f} | Test MAE: {test_mae/n_test:.4f}")
