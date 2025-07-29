Python Code report for DSCI 452

```python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from pyswarms.single.global_best import GlobalBestPSO
import torch.nn as nn
import torch.nn.functional as F

```
Now we will import our dataset that we have created from the Rstudio markdown file

```python
df = pd.read_excel(r"C:\Users\eliwe\Documents\Data_files\data_resampled.xlsx")

```

We will define the nodes and edges for our data based on the spatial dependancies

```python

node_names = [
    "lhjc", "left_hip", "lsjc", "lejc", "lkjc", "lajc", "lwjc",
    "rhjc", "right_hip", "rsjc", "rejc", "rkjc", "rajc", "rwjc"
]

#edges are based on spatial dependancies, so nodes are connected based on their nearest spatial neighbor in sequence with the body
edges = [
    ("lsjc", "lejc"), ("lejc", "lwjc"), ("lwjc", "lhjc"),
    ("rsjc", "rejc"), ("rejc", "rwjc"), ("rwjc", "rhjc"),
    ("left_hip", "lkjc"), ("lkjc", "lajc"),
    ("right_hip", "rkjc"), ("rkjc", "rajc"),
    ("left_hip", "lsjc"), ("right_hip", "rsjc"),
    ("lsjc", "rsjc"), ("left_hip", "right_hip")
]

#this creates the index for the edges
node_to_idx = {name: i for i, name in enumerate(node_names)}
edge_index = [(node_to_idx[a], node_to_idx[b]) for a, b in edges]

```





And then we can reshape our data so that it will work well for the type of model we are creating


```python
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
        targets.append([row["exit_velo_mph_x"]])  # only predict distance
    return np.stack(swings), np.array(targets)

swing_tensor, swing_targets = reshape_biomech_df(df, node_names)

```

Next we can make our test / train / validate splits 


```python

# Train/Val/Test Split
# 60/20/20 split for training, validation, and testing
X_temp, X_test, y_temp, y_test = train_test_split(
    swing_tensor, swing_targets, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

```

And then convert our data into tensors for use in a GNN model,
this uses the dataloader function from torch

```python

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
val_loader = DataLoader(SwingDataset(X_val, y_val), batch_size=32)
test_loader = DataLoader(SwingDataset(X_test, y_test), batch_size=32)

```

Now we will define our GNN model


```python

# GNN-GRU hybrid model for spatio-temporal node feature learning.

# Final output is a prediction per sequence(biomechanical metric)

class GNN_GRU_Model(nn.Module):
    def __init__(self, input_dim=3, num_nodes=14, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        
        # Project raw node features to hidden dimension
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        
        # Stack of GNN layers (simple edge-based aggregation via shared linear transformation)
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # GRU to model temporal dynamics across flattened node features
        self.gru = nn.GRU(
            input_size=hidden_dim * num_nodes,  #flatten node features per timestep
            hidden_size=hidden_dim,             # GRU hidden state size
            batch_first=True                    #input shape: (B, T, F)
        )
        
        #final layer to make output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # x: (B, T, N, D) - batch of spatio-temporal node features
        B, T, N, D = x.shape
        
        #project node features to hidden space
        x = self.node_proj(x)  # (B, T, N, hidden_dim)
        
        # apply GNN layers with edge-based message passing
        for layer in self.gnn_layers:
            x_new = torch.zeros_like(x)
            for i, j in edge_index:
                # uses a linear layer
                x_new[:, :, i] += layer(x[:, :, j])
                x_new[:, :, j] += layer(x[:, :, i])
            x = F.relu(x_new) 
        
        # flattens node features across spatial dimension for GRU input
        x_flat = x.view(B, T, -1)  # (B, T, N * hidden_dim)
        
        #GRU processes temporal sequence of flattened node features
        _, h = self.gru(x_flat)  # h: (1, B, hidden_dim)
        
        # final prediction from GRU hidden state
        return self.fc(h.squeeze(0))  # (B, output_dim)

#function to build GNN-GRU model with size and depth given by pyswarm
def build_gnn_model(num_nodes, num_layers):
    return GNN_GRU_Model(hidden_dim=num_nodes, num_layers=num_layers)


```

Note that in the model creation, we do not specify the number of nodes or layers.
We are going to get those values through hyperparamter tuning with pyswarm.


```python
#Training and Evaluation Loop
# trains model for a few epochs and returns validation loss (used for PSO)

# trains model and returns average validation loss after all epochs
def train_and_evaluate(model, optimizer, train_loader, val_loader, epochs=10):
    loss_fn = nn.MSELoss()
    
    # training loop
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
    
    # evaluation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            val_loss += loss_fn(output, y_batch).item()
    
    return val_loss / len(val_loader)

# objective function for pso
# takes list of hyperparameter sets and returns their validation losses
def objective_function(hyperparams):
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


```
And create the best model with those parameters


```python

best_num_nodes = int(best_pos[0])
best_lr        = float(best_pos[1])
best_num_layers = int(round(best_pos[2]))

best_model = GNN_GRU_Model(hidden_dim=best_num_nodes, num_layers=best_num_layers)
best_optimizer = torch.optim.Adam(best_model.parameters(), lr=best_lr)

```

And now we can train that model on the data, first by defining the training function


```python

def train_and_validate(best_model, train_loader, val_loader, best_optimizer, edge_index, epochs=50):
    mse_crit = nn.MSELoss()
    mae_crit = nn.L1Loss()

    for epoch in range(epochs):
        model.train()
        train_mse, train_mae = 0.0, 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            #loss mse and mae calculations
            preds = model(X_batch, edge_index)
            loss_mse = mse_crit(preds, y_batch)
            loss_mae = mae_crit(preds, y_batch)
            loss_mse.backward()
            optimizer.step()

            train_mse += loss_mse.item()
            train_mae += loss_mae.item()
            #keeping mse and mae for the calculations
        model.eval()
        val_mse, val_mae = 0.0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch, edge_index)
                val_mse += mse_crit(preds, y_batch).item()
                val_mae += mae_crit(preds, y_batch).item()
                #updating the loss values from that batch
        n_train = len(train_loader)
        n_val   = len(val_loader)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train MSE: {train_mse/n_train:.4f} | Train MAE: {train_mae/n_train:.4f} | "
            f"Val MSE: {val_mse/n_val:.4f} | Val MAE: {val_mae/n_val:.4f}"
        )

```

```python

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

#full call to train the model

```

```python

model.eval()
test_loss = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        test_loss += F.mse_loss(model(X_batch, edge_index), y_batch).item()
print(f"\nTest MSE: {test_loss/len(test_loader):.4f}")

#model eval
```


```python

model.eval()
test_mse, test_mae = 0.0, 0.0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch, edge_index)
        test_mse += nn.MSELoss()(preds, y_batch).item()
        test_mae += nn.L1Loss()(preds, y_batch).item()

n_test = len(test_loader)
print(f"\nTest MSE: {test_mse/n_test:.4f} | Test MAE: {test_mae/n_test:.4f}")
#more model eval on the test set

```

With this model and tuned hyperparamters, we are able to get the following results:

MSE: 49.1973
MAE: 5.4552

This is a pretty decent model. Based on our data, we know that exit velocity has a mean of 90.12 MPH, and a standard deviation of 7.19. 
This means that our model is able to predict exit velocity well within a single standard deviation from the dataset.


The graph transformer model is pretty similar in performance, but slightly different in architecture. 


```python

# applies attention across nodes (joints) and models motion over time
class GraphTransformerModel(nn.Module):
    def __init__(self, input_dim = 3, num_nodes = 14, hidden_dim = 64, num_heads = 4, num_layers = 2, output_dim = 1):
        # input_dim = spatial coords (x, y, z)
        # num_nodes = number of joints tracked
        # hidden_dim = size of node embeddings
        # num_heads = how many heads to use for attention
        # num_layers = stacked transformer blocks
        # output_dim = predicting EV, LA, and dist
        super().__init__()

        # maps input (x,y,z) into hidden space
        self.node_proj = nn.Linear(input_dim, hidden_dim)

        
        # this applies attention over nodes at each time step independently
        encoder_layer = nn.TransformerEncoderLayer(d_model = hidden_dim, nhead = num_heads, batch_first = True)
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

       
        # GRU models how spatial embeddings change across time
        self.gru = nn.GRU(input_size = hidden_dim * num_nodes, hidden_size = hidden_dim, batch_first = True)

       
        # projects hidden state to a single regression target
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        B, T, N, D = x.shape  # B=batch, T=time steps, N=nodes, D=dimensions
        x = self.node_proj(x)  # project (B, T, N, D) → (B, T, N, hidden)
        
        # apply transformer to each frame independently
        spatial_encoded = []
        for t in range(T):
            frame = x[:, t]  # (B, N, hidden)
            frame = self.spatial_transformer(frame)  # apply attention across nodes
            spatial_encoded.append(frame)
        x_spatial = torch.stack(spatial_encoded, dim = 1)  # (B, T, N, hidden)

        # flatten node embeddings for temporal input
        x_flat = x_spatial.view(B, T, -1)  # (B, T, N*hidden)
        _, h = self.gru(x_flat)  # GRU outputs hidden state
        return self.fc(h.squeeze(0))  # (B, 3)


def build_graph_transformer_model(num_nodes, num_heads, num_layers):
    return GraphTransformerModel(
        input_dim = 3,
        num_nodes = num_nodes,    
        hidden_dim = 64,
        num_heads = num_heads,
        num_layers = num_layers,
        output_dim = 1
    )

#no pyswarm for this model type, so we will just build it as follows
transformer_model = build_graph_transformer_model(num_nodes=14, num_heads=4, num_layers=3)

# define optimizer for transformer model
transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.05)


# train the graph transformer model and log metrics
gt_train_losses, gt_val_losses, gt_train_maes, gt_val_maes, \
gt_train_targets_over_epochs, gt_train_preds_over_epochs, \
gt_val_targets_over_epochs, gt_val_preds_over_epochs = train_and_validate(
    transformer_model,
    train_loader,
    val_loader,
    transformer_optimizer,
    epochs=50,
    return_avg_val_loss_only=False
)

```

By default in this graph transformer model, the transformer endocder layer has full edge attention.
This means that there is an edge between every possible combination of two nodes. In this model, we get a test MSE of 49.45, and a test MAE of 5.42. This is slightly better than the GNN model, but the difference is pretty negligible. Again, this model is able to predict exit velocity well within 1 standard deviation. 