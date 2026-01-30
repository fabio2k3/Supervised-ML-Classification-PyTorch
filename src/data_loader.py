import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset


def load_data(path):
    df = pd.read_csv(path)

    # 1️⃣ Eliminar columnas ID
    for col in df.columns:
        if "id" in col.lower():
            df = df.drop(columns=[col])

    # 2️⃣ Separar features y target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 3️⃣ One-hot para features categóricas
    X = pd.get_dummies(X)

    # 4️⃣ 🔥 LABEL ENCODING REAL (CLAVE)
    y = y.astype("category")
    y = y.cat.codes  # fuerza 0..C-1

    # 5️⃣ Forzar numérico
    X = X.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 6️⃣ NumPy float / int
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    return TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y)
    )
