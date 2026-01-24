import torch
from torch.utils.data import DataLoader
from src.model import Classifier
from src.data_loader import load_data


def evaluate():
    dataset = load_data("data/test.csv")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    input_size = dataset.tensors[0].shape[1]

    model = Classifier(input_size)
    model.load_state_dict(torch.load("outputs/model.pth"))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            outputs = model(X)
            predictions = (outputs >= 0.5).float()

            correct += (predictions == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    print(f"Model Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    evaluate()
