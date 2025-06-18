# Three-Node Simulation Guide

This guide explains how to run a distributed simulation across three nodes using multiple GPUs on a single machine.

## Prerequisites

- A machine with at least 3 NVIDIA GPUs (ideally)
- CUDA installed and configured
- PyTorch installed with CUDA support
- The DistributedSim repository cloned

## System Requirements

The simulation has been tested on a system with:
- 3 NVIDIA GPUs:
  - GPU 0: Quadro RTX 6000
  - GPU 1: NVIDIA TITAN RTX
  - GPU 2: NVIDIA TITAN RTX
- CUDA Version: 12.2
- Linux operating system

## Setup Instructions

1. **Verify GPU Availability**

   Check that your GPUs are properly detected:
   ```bash
   nvidia-smi
   ```

   This should display information about all available GPUs.

2. **Verify PyTorch CUDA Support**

   Ensure PyTorch can access your GPUs:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Number of GPUs:', torch.cuda.device_count())"
   ```

   This should output `CUDA available: True` and show the number of available GPUs.

3. **Make the Script Executable**

   ```bash
   chmod +x run_three_nodes_gpu.sh
   ```

4. **Run the Simulation**

   ```bash
   ./run_three_nodes_gpu.sh
   ```

   Follow the on-screen prompts to select the type of simulation you want to run.

## Available Simulation Types

1. **Simple 3-Node Test**
   - A basic test that doesn't require PyTorch
   - Simulates computation and communication between nodes
   - Good for verifying that the distributed setup works

2. **MNIST 3-Node Training**
   - Trains a simple CNN model on the MNIST dataset
   - Distributes the training across 3 nodes
   - Each node runs on a separate GPU

3. **MNIST 3-Node Training with Network Simulation**
   - Same as option 2, but with simulated network conditions
   - Allows testing how different network topologies and conditions affect training
   - Options include local, datacenter, WAN, mobile, and poor network conditions

## How It Works

The script performs the following steps:

1. Checks for PyTorch and CUDA availability
2. Creates a temporary directory for the distributed run
3. Sets up the master address and port for distributed communication
4. Launches three separate processes, each assigned to a different GPU
5. Each process runs the same script but with different rank values (0, 1, 2)
6. The processes communicate with each other using PyTorch's distributed package
7. After completion, the temporary files are cleaned up

## GPU Assignment

- Node 0 (Rank 0) runs on GPU 0
- Node 1 (Rank 1) runs on GPU 1
- Node 2 (Rank 2) runs on GPU 2

## Customizing the Simulation

### Changing Network Conditions

For the network simulation option, you can modify the network conditions by editing the `run_mnist_3nodes_with_network_sim.py` file:

```python
# Network simulation settings
# Choose from: "local", "datacenter", "wan", "mobile", "poor", or "custom"
network_profile = "wan"  # Change this to your desired profile
```

### Custom Network Parameters

You can also define custom network parameters:

```python
custom_network_params = {
    "bandwidth_range": (5, 20),      # 5-20 MB/s
    "delay_range": (0.03, 0.1),      # 30-100ms
    "packet_loss_prob": 0.005,       # 0.5% packet loss
    "jitter": 0.15                   # 15% jitter
}
network_profile = "custom"
```

### Network Topology

You can change the network topology:

```python
# Network topology (fully_connected, ring, or star)
topology_type = "fully_connected"  # Change to "ring" or "star" as needed
```

## Troubleshooting

### "Address already in use" Error

If you see an error like "Address already in use", try changing the `MASTER_PORT` value in the script to a different port number.

### GPU Memory Issues

If you encounter GPU memory errors, try reducing the batch size in the training script.

### Process Hanging

If a process hangs, you may need to manually kill it:

```bash
ps aux | grep python
kill -9 <PID>
```

## Advanced Usage

### Running on Specific GPUs

To run the simulation on specific GPUs, modify the GPU_ID assignments in the script:

```bash
run_node 0 3 2 $SCRIPT  # Run node 0 on GPU 2
run_node 1 3 0 $SCRIPT  # Run node 1 on GPU 0
run_node 2 3 1 $SCRIPT  # Run node 2 on GPU 1
```

### Changing Batch Size or Learning Rate

To modify training parameters, edit the corresponding Python script:

```python
# Training settings
batch_size = 64  # Change as needed
test_batch_size = 1000
epochs = 2  # Change as needed
lr = 1.0  # Change as needed
```

## Understanding the Output

Each node will output logs prefixed with its rank:

```
[Rank 0] Starting simulation...
[Rank 1] Starting simulation...
[Rank 2] Starting simulation...
```

This helps you track what each node is doing during the simulation.

## Conclusion

This guide should help you run distributed simulations across three nodes using multiple GPUs on a single machine. The provided script makes it easy to experiment with different simulation types and configurations.