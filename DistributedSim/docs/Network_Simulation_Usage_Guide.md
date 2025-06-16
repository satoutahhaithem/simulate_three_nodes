# Network Simulation Usage Guide

This guide provides practical instructions for using the network simulation code with ExoGym. It explains how to run the code, experiment with different network conditions, and analyze the results.

## Prerequisites

Before using the network simulation code, make sure you have:

1. ExoGym installed in a virtual environment
2. PyTorch and torchvision installed
3. The network simulation files in your project directory:
   - `network_simulator.py`
   - `run_mnist_3nodes_with_network_sim.py`

## Running the Code

### Basic Usage

To run the MNIST example with network simulation:

1. Activate the virtual environment:
   ```bash
   cd DistributedSim
   source venv/bin/activate
   ```

2. Run the example script:
   ```bash
   python ../run_mnist_3nodes_with_network_sim.py
   ```

This will run the MNIST example with 3 nodes and the default "wan" network profile.

### Output

When you run the code, you'll see output like:

```
[Rank 0] Setting up network simulation with profile: wan
[Rank 0] Network Simulator initialized with:
[Rank 0] - Bandwidth: 10-100 MB/s
[Rank 0] - Delay: 10.0-50.0 ms
[Rank 0] - Packet Loss: 0.10%
[Rank 0] - Jitter: 10.00%
[Rank 0] Network Topology: fully_connected
[Rank 0] PyTorch distributed functions patched with network simulation
```

This shows that the network simulation is active and using the specified parameters.

## Experimenting with Different Network Conditions

### Changing the Network Profile

To experiment with different network conditions, modify the `network_profile` variable in `run_mnist_3nodes_with_network_sim.py`:

```python
# Network simulation settings
# Choose from: "local", "datacenter", "wan", "mobile", "poor", or "custom"
network_profile = "mobile"  # Change this to try different profiles
```

Available profiles:

| Profile | Bandwidth | Delay | Packet Loss | Jitter | Description |
|---------|-----------|-------|-------------|--------|-------------|
| local | 1-10 GB/s | 0.1-1ms | 0% | 1% | Very fast local connections |
| datacenter | 100-1000 MB/s | 0.5-5ms | 0.01% | 5% | Typical datacenter network |
| wan | 10-100 MB/s | 10-50ms | 0.1% | 10% | Wide Area Network |
| mobile | 1-10 MB/s | 50-200ms | 1% | 30% | Mobile network |
| poor | 0.1-1 MB/s | 200-1000ms | 5% | 50% | Poor network conditions |

### Creating Custom Network Conditions

To define your own network conditions, uncomment and modify the `custom_network_params` section:

```python
# For custom network settings, uncomment and modify these lines:
custom_network_params = {
    "bandwidth_range": (5, 20),      # 5-20 MB/s
    "delay_range": (0.03, 0.1),      # 30-100ms
    "packet_loss_prob": 0.005,       # 0.5% packet loss
    "jitter": 0.15                   # 15% jitter
}
network_profile = "custom"
```

### Changing the Network Topology

You can also experiment with different network topologies by changing the `topology_type` variable:

```python
# Network topology (fully_connected, ring, or star)
topology_type = "ring"  # Change this to try different topologies
```

Available topologies:

| Topology | Description |
|----------|-------------|
| fully_connected | All nodes can communicate with all other nodes |
| ring | Each node can only communicate with its adjacent nodes |
| star | One central node connects to all other nodes |

### Adjusting Communication Frequency

The `H` parameter in the `FedAvgStrategy` controls how often nodes synchronize:

```python
# Create strategy
# Increase H to reduce communication frequency (useful for slow networks)
strategy = FedAvgStrategy(inner_optim=optim_spec, H=5)
```

- `H=1`: Nodes synchronize after every batch (high communication)
- `H=5`: Nodes synchronize after every 5 batches (medium communication)
- `H=10`: Nodes synchronize after every 10 batches (low communication)

Increasing `H` reduces the communication frequency, which can be beneficial for slow networks.

## Analyzing the Results

### Training Time

One of the most obvious effects of network conditions is on training time. Compare how long it takes to train with different network profiles:

```bash
time python ../run_mnist_3nodes_with_network_sim.py
```

### Convergence

Network conditions can also affect convergence. Look at the final loss values and how quickly the model converges with different network conditions.

### Communication vs. Computation

By adjusting the `H` parameter and network conditions, you can observe the trade-off between communication and computation:

- With fast networks, frequent communication (low `H`) may lead to faster convergence
- With slow networks, less frequent communication (high `H`) may be more efficient

### Example Experiment

Here's an example experiment to compare different network conditions:

1. Run with local network:
   ```bash
   # Modify network_profile to "local" and H to 1
   time python ../run_mnist_3nodes_with_network_sim.py > results_local.txt
   ```

2. Run with datacenter network:
   ```bash
   # Modify network_profile to "datacenter" and H to 1
   time python ../run_mnist_3nodes_with_network_sim.py > results_datacenter.txt
   ```

3. Run with WAN network:
   ```bash
   # Modify network_profile to "wan" and H to 5
   time python ../run_mnist_3nodes_with_network_sim.py > results_wan.txt
   ```

4. Run with mobile network:
   ```bash
   # Modify network_profile to "mobile" and H to 10
   time python ../run_mnist_3nodes_with_network_sim.py > results_mobile.txt
   ```

5. Compare the results:
   ```bash
   grep "train_loss" results_*.txt
   ```

## Advanced Usage

### Modifying the Network Simulator

If you want to modify the network simulator itself, you can edit `network_simulator.py`. Some possible modifications:

- Add new network profiles
- Implement more sophisticated bandwidth models
- Add time-varying network conditions
- Implement more complex network topologies

### Creating Your Own Experiments

You can create your own experiments by:

1. Copying `run_mnist_3nodes_with_network_sim.py` to a new file
2. Modifying the model, dataset, or training parameters
3. Adjusting the network simulation parameters

### Visualizing Network Effects

To better understand the effects of network conditions, you could add code to log:

- Communication time vs. computation time
- Number of bytes transferred
- Actual delays experienced

Then visualize these metrics using a plotting library like matplotlib.

## Troubleshooting

### Common Issues

1. **Slow Training**: If training is extremely slow, you might have set the network conditions too restrictive. Try increasing the bandwidth or reducing the delay.

2. **Memory Issues**: If you encounter memory issues, try reducing the batch size or using a smaller model.

3. **Convergence Problems**: If the model doesn't converge well with slow networks, try increasing the `H` parameter to reduce communication frequency.

### Debugging

To debug network simulation issues, you can add more print statements in `network_simulator.py`:

```python
def simulate_delay(self):
    """Simulate network delay with jitter."""
    base_delay = random.uniform(*self.delay_range)
    jitter_amount = base_delay * self.jitter * random.uniform(-1, 1)
    delay = max(0, base_delay + jitter_amount)
    print(f"[Rank {self.rank}] Simulating delay: {delay*1000:.2f}ms")
    time.sleep(delay)
```

## Conclusion

The network simulation code provides a flexible framework for experimenting with different network conditions in distributed training. By adjusting the parameters and analyzing the results, you can gain insights into how network constraints affect distributed training algorithms and develop more robust approaches for real-world scenarios.