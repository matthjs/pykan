"""
This is a simple self-contained script that tries to fit KANs to MNIST.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from kan import KAN


# Define MLP Model
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def mnist_experiment() -> None:
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5

    # MNIST Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    """
    Uncomment this if you want MLP results
    # Model, Loss, and Optimizer
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Testing Loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Accuracy of the MLP on the MNIST test set: {100 * correct / total:.2f}%')
    """
    run_kan(train_dataset, test_dataset)


def run_kan(train_dataset, test_dataset) -> None:
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def flatten_images(dataset) -> tuple[torch.Tensor, torch.Tensor]:
        flattened_images = []
        labels = []
        for img, label in dataset:
            flattened_images.append(img.view(-1))  # Flatten the image
            labels.append(label)
        return torch.stack(flattened_images).to(device=dev), torch.tensor(labels).to(device=dev)

    train_images, train_labels = flatten_images(train_dataset)
    test_images, test_labels = flatten_images(test_dataset)
    train_dataset = {
        'train_input': train_images,
        'train_label': train_labels,
        'test_input': test_images,
        'test_label': test_labels
    }

    # Create KAN model
    model: KAN = KAN(width=[2, 5, 1], grid=3, k=3, seed=1,
                     device=dev
                     )

    # Assuming the `KAN.fit` method expects the data in a certain format, adjust as necessary
    model.fit(train_dataset, opt="LBFGS", steps=20, lamb=0.01)
    model.plot()


def main() -> None:
    mnist_experiment()


if __name__ == "__main__":
    main()
