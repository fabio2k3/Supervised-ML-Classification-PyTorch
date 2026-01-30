import torch
import numpy as np
import os
from torch import nn, optim
from src.model import Classifier
from src.data_loader import load_data


def train():
    # 🔥 REPRODUCIBILIDAD
    torch.manual_seed(42)
    np.random.seed(42)
    
    dataset = load_data("data/train.csv")
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    X_sample, y_sample = dataset[0]
    input_size = X_sample.shape[0]
    num_classes = len(torch.unique(dataset.tensors[1]))

    model = Classifier(input_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        total_loss = 0  # 🔥 MEJORA: calcular loss promedio
        
        for X, y in loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)  # 🔥 Loss promedio del epoch
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

    # 🔥 CORRECCIÓN: crear directorio si no existe
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/model.pth")
    print("\n✅ Modelo guardado en outputs/model.pth")