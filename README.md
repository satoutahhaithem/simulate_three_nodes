# ExoGym: Distributed Training Simulator with Docker

## Overview

ExoGym is a framework that simulates distributed training across multiple nodes. While it can run on a single machine using separate processes, this guide focuses on using Docker to create a more realistic distributed environment with three separate containers.

```
Docker Environment
+------------------------------------------+
|                                          |
|   +--------+    +--------+    +--------+ |
|   | Node 1 |    | Node 2 |    | Node 3 | |
|   |Container|   |Container|   |Container| |
|   +--------+    +--------+    +--------+ |
|                                          |
+------------------------------------------+
```

## Three-Node Docker Simulation Guide

This guide will help you set up and run ExoGym with three Docker containers, creating a more realistic distributed training environment.

### Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)
- Git (to clone the repository)

### Installation and Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gym
   ```

2. Build the Docker image:
   ```bash
   docker build -t exogym:latest -f Dockerfile .
   ```

### Running Three-Node Simulation with Docker

#### Option 1: Using Docker Commands

1. Create a Docker network:
   ```bash
   docker network create exogym-network
   ```

2. Run three containers:
   ```bash
   # Run the first container (Node 1)
   docker run -d --name exogym-node1 \
     --network exogym-network \
     -p 22001:22 \
     exogym:latest

   # Run the second container (Node 2)
   docker run -d --name exogym-node2 \
     --network exogym-network \
     -p 22002:22 \
     exogym:latest

   # Run the third container (Node 3)
   docker run -d --name exogym-node3 \
     --network exogym-network \
     -p 22003:22 \
     exogym:latest
   ```

3. Set up SSH keys for communication:
   ```bash
   # Generate SSH key in Node 1
   docker exec -it exogym-node1 bash -c "ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa && cat ~/.ssh/id_rsa.pub"
   
   # Copy the displayed key and add it to all three nodes
   # For each node (replace PUBLIC_KEY with the actual key):
   docker exec -it exogym-node1 bash -c "mkdir -p ~/.ssh && echo 'PUBLIC_KEY' >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
   docker exec -it exogym-node2 bash -c "mkdir -p ~/.ssh && echo 'PUBLIC_KEY' >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
   docker exec -it exogym-node3 bash -c "mkdir -p ~/.ssh && echo 'PUBLIC_KEY' >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"
   ```

4. Configure hosts files:
   ```bash
   # Get container IP addresses
   docker inspect -f '{{.Name}} - {{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' exogym-node1 exogym-node2 exogym-node3
   
   # Update hosts files in each container (replace IP_NODE1, IP_NODE2, IP_NODE3 with actual IPs)
   docker exec -it exogym-node1 bash -c "echo 'IP_NODE1 exogym-node1' >> /etc/hosts && echo 'IP_NODE2 exogym-node2' >> /etc/hosts && echo 'IP_NODE3 exogym-node3' >> /etc/hosts"
   docker exec -it exogym-node2 bash -c "echo 'IP_NODE1 exogym-node1' >> /etc/hosts && echo 'IP_NODE2 exogym-node2' >> /etc/hosts && echo 'IP_NODE3 exogym-node3' >> /etc/hosts"
   docker exec -it exogym-node3 bash -c "echo 'IP_NODE1 exogym-node1' >> /etc/hosts && echo 'IP_NODE2 exogym-node2' >> /etc/hosts && echo 'IP_NODE3 exogym-node3' >> /etc/hosts"
   ```

5. Run distributed training:
   ```bash
   # Connect to Node 1
   docker exec -it exogym-node1 bash
   
   # Inside Node 1, create and run a distributed training script
   cd /opt/DistributedSim
   source .venv/bin/activate
   
   # Run the distributed training
   # (This will SSH into the other nodes and start the workers)
   ```

#### Option 2: Using Docker Compose (Recommended)

1. Create a `docker-compose.yml` file:
   ```bash
   cat > docker-compose.yml << 'EOF'
   version: '3'

   services:
     exogym-node1:
       image: exogym:latest
       container_name: exogym-node1
       ports:
         - "22001:22"
       networks:
         - exogym-network

     exogym-node2:
       image: exogym:latest
       container_name: exogym-node2
       ports:
         - "22002:22"
       networks:
         - exogym-network

     exogym-node3:
       image: exogym:latest
       container_name: exogym-node3
       ports:
         - "22003:22"
       networks:
         - exogym-network

   networks:
     exogym-network:
       driver: bridge
   EOF
   ```

2. Start the containers:
   ```bash
   docker-compose up -d
   ```

3. Follow steps 3-5 from Option 1 to set up SSH keys, configure hosts, and run distributed training.

### Simple Test Script

For a quick verification of the three-node functionality without Docker, you can use the provided simple test script:

```bash
python simple_3node_test.py
```

This script simulates three nodes and their communication with minimal dependencies, useful for basic verification.

## Customizing Your Docker Simulation

### Creating a Distributed Training Script

To run distributed training across the three Docker containers, create a script in the first container:

```bash
docker exec -it exogym-node1 bash

# Inside the container
cd /opt/DistributedSim
cat > run_distributed.sh << 'EOF'
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

chmod +x run_distributed.sh
./run_distributed.sh
```

### Customizing Network Conditions

You can customize network conditions by modifying the `run_mnist_3nodes_with_network_sim.py` script:

```bash
docker exec -it exogym-node1 bash

# Inside the container
cd /opt/DistributedSim
# Edit the script to change network parameters
```

Available network profiles:
- `local`: Very fast connections (1-10 GB/s, 0.1-1ms delay)
- `datacenter`: Fast datacenter network (100-1000 MB/s, 0.5-5ms delay)
- `wan`: Wide Area Network (10-100 MB/s, 10-50ms delay)
- `mobile`: Mobile network (1-10 MB/s, 50-200ms delay)
- `poor`: Poor network conditions (0.1-1 MB/s, 200-1000ms delay)

Available network topologies:
- `fully_connected`: All nodes can communicate with all other nodes
- `ring`: Each node can only communicate with its adjacent nodes
- `star`: One central node connects to all other nodes

## Monitoring and Troubleshooting

### Monitoring Training Progress

You can monitor the training progress by checking the logs in each container:

```bash
# Check logs for node 1
docker logs exogym-node1

# Check logs for node 2
docker logs exogym-node2

# Check logs for node 3
docker logs exogym-node3
```

### Common Issues and Solutions

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
   ```

## Cleaning Up

When you're done with your experiments, you can stop and remove the containers:

```bash
# If using Docker commands
docker stop exogym-node1 exogym-node2 exogym-node3
docker rm exogym-node1 exogym-node2 exogym-node3
docker network rm exogym-network

# If using Docker Compose
docker-compose down
```

## Conclusion

Using Docker with ExoGym provides a more realistic distributed training environment by creating separate containers for each node. This approach allows you to:

1. Simulate true network conditions between separate machines
2. Test distributed training algorithms in a controlled environment
3. Experiment with different network topologies and conditions
4. Develop and debug distributed training code without requiring multiple physical machines

For more detailed information, refer to the documentation in the `docs` directory:
- `ExoGym_Documentation_Simplified.md`: Overview of ExoGym
- `ExoGym_Network_Simulation_Guide_Simplified.md`: Detailed guide on network simulation
- `docker_guide.md`: Comprehensive guide for Docker-based simulation