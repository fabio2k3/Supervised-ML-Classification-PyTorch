import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data(path):
    df = pd.read_csv(path)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    return TensorDataset(X, y)

