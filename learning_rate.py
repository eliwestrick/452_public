# === Core Libraries ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from pyswarms.single.global_best import GlobalBestPSO
import torch.nn as nn
import torch.nn.functional as F


# node definitions, these are the points of the body we are tracking in the df
node_names = ["lhjc", "left_hip", "lsjc", "lejc", "lkjc", "lajc", "lwjc",
              "rhjc", "right_hip", "rsjc", "rejc", "rkjc", "rajc", "rwjc"]


#stores particle positions so we can visualize them later
position_log = []


# converts the data which is time series data of x, y, z as well as 3 targets
# into a 3D tensor of shape (swings, nodes, time, 3)
def reshape_biomech_df(df, node_names, swing_col = "session_swing", time_col = "time"):
    swings, targets = [], []
    for swing_id, group in df.groupby(swing_col):
        group = group.sort_values(time_col)
        # Extract 3D coordinates for each node across time
        node_features = [group[[f"{node}_x", f"{node}_y", f"{node}_z"]].values.T for node in node_names]
        swing_tensor = np.stack(node_features, axis=0).transpose(2, 0, 1)  # (T, N, 3)
        swings.append(swing_tensor)
        # Extract swing outcome targets
        row = group.iloc[0]
        targets.append([row["exit_velo_mph_x"], row["la"], row["dist"]])
    return np.stack(swings), np.array(targets)

#load the data and call the reshape funtion to make it 3d tensors
df = pd.read_excel(r"C:\Users\eliwe\Documents\Data_files\data_resampled.xlsx")
swing_tensor, swing_targets = reshape_biomech_df(df, node_names)


# 60/20/20 split for train/val/test
X_temp, X_test, y_temp, y_test = train_test_split(swing_tensor, swing_targets, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)



# we now have 14 nodes of 3 dimensions (x,y,z) and 3 targets (exit velocity, launch angle, distance)






# this uses the pytorch wrapper for the data
class SwingDataset(torch.utils.data.Dataset): #defines class that inherits from pytorch dataset class
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32) #converts our tensors to pytorch tensors
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X) #gets how many swings we have so it knows what to expect
    def __getitem__(self, idx): return self.X[idx], self.y[idx] #retrieves the swing at index (idx) and will pass it to the model during training

# wraps the dataset in a dataloader so we can batch and shuffle ect
train_loader = DataLoader(SwingDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(SwingDataset(X_val, y_val), batch_size=32)

# gnn-gru hybrid model)
class GNN_GRU_Model(nn.Module):
    def __init__(self, input_dim=3, num_nodes=14, hidden_dim=64, num_layers=2, output_dim=3): #dim is how many dimensions (x,y,z),
        # num nodes is how many nodes we got, hiddden dim is nodes per hidden layer, layers is how many layers, and output dim is what dimension we
        # want output in, so here it is 3 because we are predicting 3 things 
        super().__init__()
        # this is for the spatial data, does linear transformation across nodes
        self.gnn_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) if i == 0 else nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        # I believe this is for learning temporal dynamics
        self.gru = nn.GRU(input_size=hidden_dim * num_nodes, hidden_size=hidden_dim, batch_first=True)
        # this maps the current shape to the shape we want for the output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x): 
        B, T, N, D = x.shape
        x = x.view(B * T, N, D)  #flatted the input to (B*T, N, D) for the gnn
        for layer in self.gnn_layers:
            x = F.relu(layer(x))  # more spatial
        x = x.view(B, T, -1)      # reshape it again
        _, h = self.gru(x)        # GRU outputs hidden state
        return self.fc(h.squeeze(0))  # this outputs the final predictions

# call to build the model
def build_gnn_model(num_nodes, num_layers):
    return GNN_GRU_Model(hidden_dim=num_nodes, num_layers=num_layers)


#trains model over a given epochs and returns the validation loss
def train_and_evaluate(model, optimizer, train_loader, val_loader, epochs=10):
    loss_fn = nn.MSELoss() #makes our loss function mean squared error
    for epoch in range(epochs):
        model.train() #sets to training mode
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch) #iterates over batches, and updates model weights
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
    #now do it on validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            val_loss += loss_fn(output, y_batch).item()
    return val_loss / len(val_loader)

#this gets the best hyperparameters using pyswarms
def objective_function(hyperparams):
    global position_log #calls in the position log to the function
    position_log.append(hyperparams.copy())  # log current particle positions

    losses = []
    for p in hyperparams: #loops over each hyperparameter set and gets loss for it
        num_nodes = int(p[0])
        lr = float(p[1]) #creates the parameter values, nodes, lr, and layers
        num_layers = int(round(p[2]))

        model = build_gnn_model(num_nodes=num_nodes, num_layers = num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr = lr) #creates a new model and optimizer for each hyperparameter set, and gets the loss for it
        loss = train_and_evaluate(model, optimizer, train_loader, val_loader, epochs=10)
        losses.append(loss)
    return np.array(losses) #uses the records from the losses of each set to find the best hyperparameters


bounds = (np.array([16, 0.001, 2]), np.array([256, 0.1, 3])) #create bounds for the hyperparameters
optimizer = GlobalBestPSO(n_particles=8, dimensions=3, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, bounds = bounds)
best_cost, best_pos = optimizer.optimize(objective_function, iters = 10) # runs the pyswarm loop at 10 iterations

#import for the plots
import matplotlib.pyplot as plt

position_log_np = np.array(position_log)  #make a numpy array so we can plot it
params = ["Nodes per Layer", "Learning Rate", "Number of Layers"] #define params
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i in range(3):
    param_vals = position_log_np[:, :, i]  # (iters, particles)
    for particle in range(param_vals.shape[1]):
        axes[i].plot(param_vals[:, particle], alpha=0.6)
    axes[i].set_title(params[i])
    axes[i].set_xlabel("Iteration")
    axes[i].set_ylabel("Value")
    axes[i].grid(True)
#loops for each parameter and plots the values across the 10 iterations
plt.suptitle("Hyperparameter Evolution Across Particles")
plt.tight_layout()
plt.show()

#prints the best hyperparameters
print(f"nodes: {int(best_pos[0])}")
print(f"lr: {best_pos[1]:.5f}")
print(f"layers: {int(round(best_pos[2]))}")
