import numpy as np
from utils import load_and_preprocess, split_data

# Prepare data and save split files
X, y, scaler = load_and_preprocess()
clients = split_data(X, y, num_clients=3)

# Save each client's data
for i, (X_client, y_client) in enumerate(clients):
    np.save(f"X_client_{i}.npy", X_client)
    np.save(f"y_client_{i}.npy", y_client)
