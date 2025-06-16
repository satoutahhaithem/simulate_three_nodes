#!/bin/bash

# This script modifies the MNIST example to use 3 nodes and runs it in CPU-only mode

# Set environment variable to ensure CPU-only mode
export CUDA_VISIBLE_DEVICES=""

# Create a modified version of the MNIST example with 3 nodes
cat > /opt/DistributedSim/examples/mnist_3nodes.py << 'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from exogym.trainer import Trainer
from exogym.strategy import SimpleReduceStrategy

# Define a simple CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def main():
    # Training settings
    batch_size = 64
    test_batch_size = 1000
    epochs = 2
    lr = 1.0
    gamma = 0.7
    
    # Set device to CPU
    device = "cpu"
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create model
    model = Net()
    
    # Define loss function and optimizer
    criterion = F.nll_loss
    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    
    # Create strategy
    strategy = SimpleReduceStrategy(model, optimizer)
    
    # Create trainer with 3 nodes
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        criterion=criterion,
        strategy=strategy,
        device=device,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        epochs=epochs,
        num_nodes=3,  # Use 3 nodes
        scheduler=scheduler
    )
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
EOF

# Run the modified example
cd /opt/DistributedSim
source .venv/bin/activate
python examples/mnist_3nodes.py