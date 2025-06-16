# Network Simulation Usage Guide (Simplified)

This guide provides practical instructions for using the network simulation with ExoGym. It explains how to run experiments with different network conditions and analyze the results.

## Prerequisites

Before using the network simulation, make sure you have:

1. ExoGym installed in a virtual environment
2. PyTorch and torchvision installed
3. The network simulation files in your project directory

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
   python run_mnist_3nodes_with_network_sim.py
   ```

This will run the MNIST example with 3 nodes and the default "wan" network profile.

## Experimenting with Different Network Conditions

### Available Network Profiles

| Profile | Bandwidth | Delay | Packet Loss | Jitter | Description |
|---------|-----------|-------|-------------|--------|-------------|
| local | 1-10 GB/s | 0.1-1ms | 0% | 1% | Very fast local connections |
| datacenter | 100-1000 MB/s | 0.5-5ms | 0.01% | 5% | Typical datacenter network |
| wan | 10-100 MB/s | 10-50ms | 0.1% | 10% | Wide Area Network |
| mobile | 1-10 MB/s | 50-200ms | 1% | 30% | Mobile network |
| poor | 0.1-1 MB/s | 200-1000ms | 5% | 50% | Poor network conditions |

### Available Network Topologies

```
Fully Connected:    Ring:             Star:
                                        
  1 --- 2           1 --- 2             2
  |     |           |     |            /
  |     |           |     |           /
  3 --- 4           4 --- 3          1 --- 4
                                      \
                                       \
                                        3
```

| Topology | Description |
|----------|-------------|
| fully_connected | All nodes can communicate with all other nodes |
| ring | Each node can only communicate with its adjacent nodes |
| star | One central node connects to all other nodes |

### Adjusting Communication Frequency

The `H` parameter in the `FedAvgStrategy` controls how often nodes synchronize:

- `H=1`: Nodes synchronize after every batch (high communication)
- `H=5`: Nodes synchronize after every 5 batches (medium communication)
- `H=10`: Nodes synchronize after every 10 batches (low communication)

Increasing `H` reduces the communication frequency, which can be beneficial for slow networks.

## Analyzing the Results

### Training Time

One of the most obvious effects of network conditions is on training time. Compare how long it takes to train with different network profiles:

```bash
time python run_mnist_3nodes_with_network_sim.py
```

### Convergence

Network conditions can also affect convergence. Look at the final loss values and how quickly the model converges with different network conditions.

### Communication vs. Computation

By adjusting the `H` parameter and network conditions, you can observe the trade-off between communication and computation:

```
Training Time Comparison:
                                                  
Fast ↑                                            
      |  *                                        
      |     *                                     
Time  |        *                                  
      |           *                               
      |              *                            
Slow ↓                 *                          
      +---------------------------------------→    
        Good                Network               Poor
```

- With fast networks, frequent communication (low `H`) may lead to faster convergence
- With slow networks, less frequent communication (high `H`) may be more efficient

## Example Experiment

Here's an example experiment to compare different network conditions:

1. Run with local network (modify network_profile to "local" and H to 1)
2. Run with datacenter network (modify network_profile to "datacenter" and H to 1)
3. Run with WAN network (modify network_profile to "wan" and H to 5)
4. Run with mobile network (modify network_profile to "mobile" and H to 10)
5. Compare the results

## Conclusion

The network simulation provides a flexible framework for experimenting with different network conditions in distributed training. By adjusting the parameters and analyzing the results, you can gain insights into how network constraints affect distributed training algorithms and develop more robust approaches for real-world scenarios.