# ExoGym Network Simulation

This repository contains tools and examples for simulating network conditions in distributed training using ExoGym.

## Files

- `ExoGym_Network_Simulation_Guide.md`: Comprehensive guide explaining how to simulate network conditions in ExoGym
- `network_simulator.py`: Implementation of the network simulator for ExoGym
- `run_mnist_3nodes.py`: Basic example of running ExoGym with 3 nodes
- `run_mnist_3nodes_with_network_sim.py`: Example of running ExoGym with 3 nodes and network simulation

## Getting Started

1. Make sure you have ExoGym installed:
   ```bash
   cd DistributedSim
   source venv/bin/activate
   ```

2. Run the basic example:
   ```bash
   python run_mnist_3nodes.py
   ```

3. Run the example with network simulation:
   ```bash
   python run_mnist_3nodes_with_network_sim.py
   ```

## Network Simulation Features

The network simulator allows you to simulate:

- **Bandwidth limitations**: Control how much data can be transferred per second
- **Communication delays**: Add latency to network communications
- **Packet loss**: Simulate dropped packets
- **Jitter**: Add random variations in delay
- **Network topologies**: Simulate different connection patterns between nodes

## Predefined Network Profiles

The `network_simulator.py` includes several predefined network profiles:

- `local`: Very fast local connections (1-10 GB/s, 0.1-1ms delay)
- `datacenter`: Typical datacenter network (100-1000 MB/s, 0.5-5ms delay)
- `wan`: Wide Area Network (10-100 MB/s, 10-50ms delay)
- `mobile`: Mobile network (1-10 MB/s, 50-200ms delay)
- `poor`: Poor network conditions (0.1-1 MB/s, 200-1000ms delay)

You can select a profile in the example script by changing the `network_profile` variable.

## Custom Network Conditions

To define custom network conditions, modify the `custom_network_params` dictionary in the example script:

```python
custom_network_params = {
    "bandwidth_range": (5, 20),      # 5-20 MB/s
    "delay_range": (0.03, 0.1),      # 30-100ms
    "packet_loss_prob": 0.005,       # 0.5% packet loss
    "jitter": 0.15                   # 15% jitter
}
network_profile = "custom"
```

## Network Topologies

You can also simulate different network topologies:

- `fully_connected`: All nodes can communicate with all other nodes
- `ring`: Each node can only communicate with its adjacent nodes
- `star`: One central node connects to all other nodes

Change the topology by modifying the `topology_type` variable in the example script.

## Analyzing Results

When running with network simulation, the script will output information about:

- The network conditions being simulated
- Packet loss events
- Training progress and performance

You can compare training times and convergence across different network conditions to understand how network constraints affect distributed training algorithms.

## Advanced Usage

For more advanced usage, refer to the comprehensive guide in `ExoGym_Network_Simulation_Guide.md`.