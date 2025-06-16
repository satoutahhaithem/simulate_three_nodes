import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from exogym import LocalTrainer
from exogym.strategy import FedAvgStrategy
from exogym.strategy.optim import OptimSpec

# Import our network simulator
from network_simulator import create_network_simulator, patch_distributed_functions, NetworkTopology

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
    
    # Network simulation settings
    # Choose from: "local", "datacenter", "wan", "mobile", "poor", or "custom"
    network_profile = "wan"  # Simulating a wide-area network
    
    # For custom network settings, uncomment and modify these lines:
    # custom_network_params = {
    #     "bandwidth_range": (5, 20),      # 5-20 MB/s
    #     "delay_range": (0.03, 0.1),      # 30-100ms
    #     "packet_loss_prob": 0.005,       # 0.5% packet loss
    #     "jitter": 0.15                   # 15% jitter
    # }
    # network_profile = "custom"
    
    # Network topology (fully_connected, ring, or star)
    topology_type = "fully_connected"
    
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
    # Increase H to reduce communication frequency (useful for slow networks)
    strategy = FedAvgStrategy(inner_optim=optim_spec, H=5)  
    
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
        eval_interval=100,
        # This callback will be called after the distributed processes are created
        # but before training starts, allowing us to set up the network simulation
        setup_callback=lambda rank, world_size: setup_network_simulation(
            rank, world_size, network_profile, topology_type
        )
    )

def setup_network_simulation(rank, world_size, network_profile, topology_type):
    """
    Set up network simulation for a specific rank.
    This function is called by each process after initialization.
    """
    print(f"[Rank {rank}] Setting up network simulation with profile: {network_profile}")
    
    # Create network simulator with the specified profile
    if network_profile == "custom":
        # Use custom parameters if defined
        network_sim = create_network_simulator(
            profile="custom", 
            custom_params=custom_network_params
        )
    else:
        network_sim = create_network_simulator(profile=network_profile)
    
    # Create network topology
    topology = NetworkTopology(world_size, topology_type=topology_type)
    
    # Patch PyTorch distributed functions
    restore_fn = patch_distributed_functions(network_sim, topology)
    
    # Return the restore function so it can be called later if needed
    return restore_fn

if __name__ == "__main__":
    main()