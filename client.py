import flwr as fl
import numpy as np
from utils import create_model
import sys

class AirQualityClient(fl.client.NumPyClient):
    def __init__(self, X, y):
        self.model = create_model()
        self.X, self.y = X, y

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X, self.y, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.X), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss = self.model.evaluate(self.X, self.y, verbose=0)
        return loss, len(self.X), {}

if __name__ == "__main__":
    client_id = int(sys.argv[1])  # pass client ID as arg
    X = np.load(f"X_client_{client_id}.npy")
    y = np.load(f"y_client_{client_id}.npy")
    client = AirQualityClient(X, y)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
