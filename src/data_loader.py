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

    # 2️⃣ Separar X e y
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 3️⃣ One-Hot Encoding para categóricas
    X = pd.get_dummies(X)

    # 4️⃣ Codificar target si no es numérico
    if y.dtype == "object":
        y = y.astype("category").cat.codes

    # 5️⃣ Convertir TODO a float (sin excepciones)
    X = X.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    y = pd.to_numeric(y, errors="coerce")

    # 6️⃣ Reemplazar NaN e infinitos
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y.fillna(0)

    # 7️⃣ 🔥 FORZAR NumPy float32 (CLAVE)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    return TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y)
    )
