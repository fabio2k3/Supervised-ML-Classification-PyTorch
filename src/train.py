import torch
from torch import nn, optim
from src.model import Classifier
from src.data_loader import load_data

def train():
    dataset = load_data("data/train.csv")
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    X_sample, y_sample = dataset[0]
    input_size = X_sample.shape[0]
    num_classes = len(torch.unique(dataset.tensors[1]))  # 🔥

    model = Classifier(input_size, num_classes)

    criterion = nn.CrossEntropyLoss()  # 🔥
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        for X, y in loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "outputs/model.pth")
