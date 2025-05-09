import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def load_and_preprocess(path="city_hour.csv", city="Delhi", feature="PM2.5", time_steps=24):
    df = pd.read_csv(path)
    df = df[df["City"] == city]
    df = df[["Datetime", feature]].dropna()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)

    scaler = MinMaxScaler()
    df[feature] = scaler.fit_transform(df[[feature]])

    def create_sequences(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i+time_steps])
            y.append(data[i+time_steps])
        return np.array(X), np.array(y)

    X, y = create_sequences(df[feature].values, time_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def split_data(X, y, num_clients):
    size = len(X) // num_clients
    return [(X[i*size:(i+1)*size], y[i*size:(i+1)*size]) for i in range(num_clients)]

def create_model(input_shape=(24, 1)):
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
