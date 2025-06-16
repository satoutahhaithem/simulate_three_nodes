import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from exogym import LocalTrainer
from exogym.strategy import FedAvgStrategy
from exogym.strategy.optim import OptimSpec

# Define a simple CNN model for MNIST
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, batch):
        # If input is a tuple (data, target), extract the data
        if isinstance(batch, tuple) and len(batch) == 2:
            x, target = batch
        else:
            # This shouldn't happen, but just in case
            x = batch
            target = None
            
        # Forward pass through the network
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
        
        # If we have targets, compute the loss
        if target is not None:
            loss = F.nll_loss(output, target)
            return loss
        else:
            # Otherwise, just return the output
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
    
    # Define optimizer specification for FedAvgStrategy
    optim_spec = OptimSpec(torch.optim.Adadelta, lr=lr)
    
    # Create strategy
    strategy = FedAvgStrategy(inner_optim=optim_spec, H=1)
    
    # Create trainer
    trainer = LocalTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=test_dataset  # Use test_dataset as validation dataset
    )
    
    # Train the model
    trainer.fit(
        num_epochs=epochs,
        strategy=strategy,
        num_nodes=3,  # Use 3 nodes
        device=device,
        batch_size=batch_size,
        val_size=test_batch_size,
        shuffle=True,
        eval_interval=100
    )

if __name__ == "__main__":
    main()