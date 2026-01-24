import torch
from torch import nn, optim
from src.model import Classifier
from src.data_loader import load_data

def train():
    dataset = load_data("data/train.csv")
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    input_size = dataset.tensors[0].shape[1]
    model = Classifier(input_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        for X, y in loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "outputs/model.pth")

        