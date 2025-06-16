# Step-by-Step Guide: Creating Docker Images and Running Three Containers for ExoGym

This document provides detailed, step-by-step instructions for creating Docker images and running three containers for the ExoGym distributed simulation framework.

## Step 1: Navigate to Your Project Directory

```bash
cd /home/sattoutah/Bureau/Exo_gym/gym
```

## Step 2: Build the Docker Image

```bash
# Build the GPU-enabled image
docker build -t exogym:gpu -f Dockerfile .
```

This command builds a Docker image named "exogym" with the tag "gpu" using the specified Dockerfile.

## Step 3: Create a Docker Network

```bash
# Create a network for the containers to communicate
docker network create exogym-network
```

This creates a virtual network that allows the containers to communicate with each other.

## Step 4: Run the First Container (Node 1)

```bash
# Run the first container
docker run -d --name exogym-node1 \
  --gpus all \
  --network exogym-network \
  -p 22001:22 \
  exogym:gpu
```

This command:
- `-d`: Runs the container in detached mode (in the background)
- `--gpus all`: Gives the container access to all available GPUs
- `--name exogym-node1`: Names the container "exogym-node1"
- `--network exogym-network`: Connects the container to our network
- `-p 22001:22`: Maps port 22 in the container to port 22001 on your host
- `exogym:gpu`: Uses the image we built in Step 2

## Step 5: Run the Second Container (Node 2)

```bash
# Run the second container
docker run -d --name exogym-node2 \
  --gpus all \
  --network exogym-network \
  -p 22002:22 \
  exogym:gpu
```

## Step 6: Run the Third Container (Node 3)

```bash
# Run the third container
docker run -d --name exogym-node3 \
  --gpus all \
  --network exogym-network \
  -p 22003:22 \
  exogym:gpu
```

## Step 7: Verify All Containers Are Running

```bash
# Check that all containers are running
docker ps
```

You should see all three containers listed as running.

## Step 8: Set Up SSH Keys for Communication

### 8.1: Generate SSH Key in Node 1

```bash
# Connect to the first container
docker exec -it exogym-node1 bash

# Generate SSH key (inside container)
ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa

# Display the public key (inside container)
cat ~/.ssh/id_rsa.pub

# Copy the displayed key (you'll need it in the next steps)
```

### 8.2: Add the Public Key to Node 1

While still inside the first container:

```bash
# Create authorized_keys file
mkdir -p ~/.ssh
echo "PASTE_THE_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

# Exit the container
exit
```

### 8.3: Add the Public Key to Node 2

```bash
# Connect to the second container
docker exec -it exogym-node2 bash

# Create authorized_keys file (inside container)
mkdir -p ~/.ssh
echo "PASTE_THE_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

# Exit the container
exit
```

### 8.4: Add the Public Key to Node 3

```bash
# Connect to the third container
docker exec -it exogym-node3 bash

# Create authorized_keys file (inside container)
mkdir -p ~/.ssh
echo "PASTE_THE_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys

# Exit the container
exit
```

## Step 9: Get Container IP Addresses

```bash
# Get IP addresses of all containers
docker inspect -f '{{.Name}} - {{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' exogym-node1 exogym-node2 exogym-node3
```

This will display something like:
```
/exogym-node1 - 172.18.0.2
/exogym-node2 - 172.18.0.3
/exogym-node3 - 172.18.0.4
```

## Step 10: Update Hosts Files in Each Container

### 10.1: Update Hosts File in Node 1

```bash
# Connect to the first container
docker exec -it exogym-node1 bash

# Add entries to hosts file (inside container)
echo "172.18.0.2 exogym-node1" >> /etc/hosts
echo "172.18.0.3 exogym-node2" >> /etc/hosts
echo "172.18.0.4 exogym-node3" >> /etc/hosts

# Exit the container
exit
```

Replace the IP addresses with the ones you got in Step 9.

### 10.2: Update Hosts File in Node 2

```bash
# Connect to the second container
docker exec -it exogym-node2 bash

# Add entries to hosts file (inside container)
echo "172.18.0.2 exogym-node1" >> /etc/hosts
echo "172.18.0.3 exogym-node2" >> /etc/hosts
echo "172.18.0.4 exogym-node3" >> /etc/hosts

# Exit the container
exit
```

### 10.3: Update Hosts File in Node 3

```bash
# Connect to the third container
docker exec -it exogym-node3 bash

# Add entries to hosts file (inside container)
echo "172.18.0.2 exogym-node1" >> /etc/hosts
echo "172.18.0.3 exogym-node2" >> /etc/hosts
echo "172.18.0.4 exogym-node3" >> /etc/hosts

# Exit the container
exit
```

## Step 11: Create a Distributed Training Script

```bash
# Connect to the first container
docker exec -it exogym-node1 bash

# Create the script (inside container)
cat > /opt/DistributedSim/run_distributed.sh << 'EOF'
#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Start the master node
python run_mnist_3nodes_with_network_sim.py \
  --master_addr exogym-node1 \
  --master_port 29500 \
  --nnodes 3 \
  --node_rank 0 &

# SSH into node2 and start the worker
ssh -o StrictHostKeyChecking=no exogym-node2 \
  "cd /opt/DistributedSim && \
   source .venv/bin/activate && \
   python run_mnist_3nodes_with_network_sim.py \
   --master_addr exogym-node1 \
   --master_port 29500 \
   --nnodes 3 \
   --node_rank 1" &

# SSH into node3 and start the worker
ssh -o StrictHostKeyChecking=no exogym-node3 \
  "cd /opt/DistributedSim && \
   source .venv/bin/activate && \
   python run_mnist_3nodes_with_network_sim.py \
   --master_addr exogym-node1 \
   --master_port 29500 \
   --nnodes 3 \
   --node_rank 2" &

# Wait for all background processes to finish
wait
EOF

# Make the script executable
chmod +x /opt/DistributedSim/run_distributed.sh
```

## Step 12: Modify the Training Script to Accept Command Line Arguments

```bash
# While still in the first container
cat > /opt/DistributedSim/run_mnist_3nodes_with_network_sim_distributed.py << 'EOF'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse

from exogym import LocalTrainer
from exogym.strategy import FedAvgStrategy
from exogym.strategy.optim import OptimSpec

# Import our network simulator
from network_simulator import create_network_simulator, patch_distributed_functions, NetworkTopology

# Parse command line arguments
parser = argparse.ArgumentParser(description='Distributed MNIST training with ExoGym')
parser.add_argument('--master_addr', type=str, default='localhost', help='Master node address')
parser.add_argument('--master_port', type=str, default='29500', help='Master node port')
parser.add_argument('--nnodes', type=int, default=3, help='Number of nodes')
parser.add_argument('--node_rank', type=int, default=0, help='Node rank')
args = parser.parse_args()

# Set environment variables for PyTorch distributed
import os
os.environ['MASTER_ADDR'] = args.master_addr
os.environ['MASTER_PORT'] = args.master_port

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
    network_profile = "wan"  # Simulating a wide-area network
    topology_type = "fully_connected"
    
    # Set device to GPU
    device = "cuda"
    
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
    strategy = FedAvgStrategy(inner_optim=optim_spec, H=5)  
    
    # Create trainer
    trainer = LocalTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=test_dataset
    )
    
    # Train the model
    trainer.fit(
        num_epochs=epochs,
        strategy=strategy,
        num_nodes=args.nnodes,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=int(args.master_port),
        device=device,
        batch_size=batch_size,
        val_size=test_batch_size,
        shuffle=True,
        eval_interval=100,
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
    network_sim = create_network_simulator(profile=network_profile)
    
    # Create network topology
    topology = NetworkTopology(world_size, topology_type=topology_type)
    
    # Patch PyTorch distributed functions
    restore_fn = patch_distributed_functions(network_sim, topology)
    
    # Return the restore function so it can be called later if needed
    return restore_fn

if __name__ == "__main__":
    main()
EOF
```

## Step 13: Run the Distributed Training

```bash
# While still in the first container
cd /opt/DistributedSim
./run_distributed.sh
```

This will start the distributed training across all three containers.

## Step 14: Monitor the Training Progress

You can monitor the training progress by checking the logs in each container:

```bash
# Check logs for node 1
docker logs exogym-node1

# Check logs for node 2
docker logs exogym-node2

# Check logs for node 3
docker logs exogym-node3
```

## Step 15: Stop the Containers When Done

```bash
# Stop all containers
docker stop exogym-node1 exogym-node2 exogym-node3

# Remove the containers
docker rm exogym-node1 exogym-node2 exogym-node3

# Remove the network
docker network rm exogym-network
```

## Alternative: Using Docker Compose

If you prefer a simpler approach, you can use Docker Compose:

### Step 1: Create a docker-compose.yml File

```bash
# Create the docker-compose.yml file
cat > docker-compose.yml << 'EOF'
version: '3'

services:
  exogym-node1:
    image: exogym:gpu
    container_name: exogym-node1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "22001:22"
    networks:
      - exogym-network

  exogym-node2:
    image: exogym:gpu
    container_name: exogym-node2
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "22002:22"
    networks:
      - exogym-network

  exogym-node3:
    image: exogym:gpu
    container_name: exogym-node3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "22003:22"
    networks:
      - exogym-network

networks:
  exogym-network:
    driver: bridge
EOF
```

### Step 2: Start the Containers with Docker Compose

```bash
# Start all containers
docker-compose up -d
```

### Step 3: Follow Steps 8-13 as Above

Continue with the SSH key setup and other steps as described above.

### Step 4: Stop the Containers When Done

```bash
# Stop and remove all containers
docker-compose down
```

## Troubleshooting

If you encounter issues:

1. **SSH Connection Issues**:
   ```bash
   # Test SSH connection
   docker exec -it exogym-node1 ssh -o StrictHostKeyChecking=no exogym-node2 echo "Connection test"
   ```

2. **Container Network Issues**:
   ```bash
   # Check network
   docker network inspect exogym-network
   ```

3. **GPU Issues**:
   ```bash
   # Check if GPU is available in the container
   docker exec -it exogym-node1 bash -c "source .venv/bin/activate && python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'"
   ```

4. **Python Environment Issues**:
   ```bash
   # Check Python environment
   docker exec -it exogym-node1 bash -c "source .venv/bin/activate && python -c 'import torch; print(torch.__version__)'"