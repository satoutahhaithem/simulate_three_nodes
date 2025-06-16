# ExoGym Network Simulation Guide

This document explains how to simulate network conditions like bandwidth limitations and communication delays in ExoGym distributed training simulations.

## Understanding the Current Simulation

In our current setup with 3 nodes:

1. **Process-Based Simulation**: ExoGym creates 3 separate Python processes, each representing a node in a distributed system.

2. **Communication Pattern**: The nodes communicate using PyTorch's distributed package (`torch.distributed`), which by default uses the fastest available communication method with no artificial constraints.

3. **Synchronization Strategy**: We're using `FedAvgStrategy`, which implements Federated Averaging - nodes train locally for a certain number of steps (controlled by parameter `H`), then synchronize by averaging their models.

4. **Default Network Behavior**: By default, ExoGym doesn't simulate network constraints - communication between processes happens at the maximum speed allowed by your hardware.

## Simulating Network Conditions

To simulate realistic network conditions like bandwidth limitations and communication delays, you need to modify ExoGym's communication layer. Here's how to do it:

### 1. Understanding the Communication Layer

ExoGym's communication happens primarily in two places:

- **Strategy Classes**: These implement the communication patterns between nodes (e.g., `FedAvgStrategy`, `DiLoCoStrategy`)
- **PyTorch Distributed Functions**: The actual data transfer happens through functions like `all_reduce`, `broadcast`, etc.

### 2. Adding Network Simulation

To simulate network conditions, you need to:

1. **Create a Network Simulator Class**:

```python
# network_simulator.py
import time
import random
import torch.distributed as dist
from functools import wraps

class NetworkSimulator:
    def __init__(self, bandwidth_range=(10, 100), delay_range=(0.01, 0.5), 
                 packet_loss_prob=0.0, jitter=0.0):
        """
        Simulate network conditions.
        
        Args:
            bandwidth_range: Tuple of (min, max) bandwidth in MB/s
            delay_range: Tuple of (min, max) delay in seconds
            packet_loss_prob: Probability of packet loss (0.0 to 1.0)
            jitter: Random variation in delay (0.0 to 1.0)
        """
        self.bandwidth_range = bandwidth_range
        self.delay_range = delay_range
        self.packet_loss_prob = packet_loss_prob
        self.jitter = jitter
        
    def simulate_delay(self):
        """Simulate network delay with jitter."""
        base_delay = random.uniform(*self.delay_range)
        jitter_amount = base_delay * self.jitter * random.uniform(-1, 1)
        delay = max(0, base_delay + jitter_amount)
        time.sleep(delay)
        
    def simulate_bandwidth(self, data_size_bytes):
        """Simulate bandwidth limitation."""
        bandwidth = random.uniform(*self.bandwidth_range) * 1024 * 1024  # Convert to bytes/s
        transfer_time = data_size_bytes / bandwidth
        time.sleep(transfer_time)
        
    def simulate_packet_loss(self):
        """Simulate packet loss. Returns True if packet should be dropped."""
        return random.random() < self.packet_loss_prob

# Create patched versions of PyTorch distributed functions
def patch_distributed_functions(network_simulator):
    """Patch PyTorch distributed functions to simulate network conditions."""
    original_all_reduce = dist.all_reduce
    
    @wraps(dist.all_reduce)
    def patched_all_reduce(tensor, *args, **kwargs):
        # Simulate delay
        network_simulator.simulate_delay()
        
        # Simulate bandwidth limitation based on tensor size
        data_size = tensor.element_size() * tensor.nelement()
        network_simulator.simulate_bandwidth(data_size)
        
        # Simulate packet loss
        if network_simulator.simulate_packet_loss():
            # In real networks, packet loss would trigger retransmission
            # Here we just add extra delay to simulate that
            network_simulator.simulate_delay()
        
        # Call the original function
        return original_all_reduce(tensor, *args, **kwargs)
    
    # Replace the original function with our patched version
    dist.all_reduce = patched_all_reduce
    
    # Similarly patch other distributed functions like broadcast, gather, etc.
    # ...
```

2. **Modify Your Training Script**:

```python
from network_simulator import NetworkSimulator, patch_distributed_functions

def main():
    # ... (existing code)
    
    # Create and configure network simulator
    network_sim = NetworkSimulator(
        bandwidth_range=(10, 50),  # 10-50 MB/s bandwidth
        delay_range=(0.05, 0.2),   # 50-200ms delay
        packet_loss_prob=0.01,     # 1% packet loss
        jitter=0.2                 # 20% jitter
    )
    
    # Patch PyTorch distributed functions
    patch_distributed_functions(network_sim)
    
    # ... (rest of your code)
```

### 3. Implementing Different Network Topologies

For more complex network topologies (e.g., star, ring, hierarchical):

```python
class NetworkTopology:
    def __init__(self, num_nodes, topology_type="fully_connected"):
        """
        Simulate different network topologies.
        
        Args:
            num_nodes: Number of nodes in the network
            topology_type: Type of topology ("fully_connected", "ring", "star", etc.)
        """
        self.num_nodes = num_nodes
        self.topology_type = topology_type
        self.adjacency_matrix = self._create_adjacency_matrix()
        
    def _create_adjacency_matrix(self):
        """Create adjacency matrix based on topology type."""
        matrix = [[0 for _ in range(self.num_nodes)] for _ in range(self.num_nodes)]
        
        if self.topology_type == "fully_connected":
            # All nodes connected to all other nodes
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j:
                        matrix[i][j] = 1
        
        elif self.topology_type == "ring":
            # Each node connected only to adjacent nodes
            for i in range(self.num_nodes):
                matrix[i][(i + 1) % self.num_nodes] = 1
                matrix[i][(i - 1) % self.num_nodes] = 1
        
        elif self.topology_type == "star":
            # Node 0 is the center, connected to all other nodes
            for i in range(1, self.num_nodes):
                matrix[0][i] = 1
                matrix[i][0] = 1
                
        return matrix
        
    def can_communicate(self, source_rank, target_rank):
        """Check if two nodes can communicate directly."""
        return self.adjacency_matrix[source_rank][target_rank] == 1
```

## Practical Implementation Steps

To implement network simulation in your ExoGym experiments:

1. **Create the Network Simulator Files**:
   - Create `network_simulator.py` with the code above
   - Adjust parameters to match your desired network conditions

2. **Modify Your Training Script**:
   - Import and initialize the network simulator
   - Apply the patches to PyTorch distributed functions

3. **Run Experiments with Different Network Conditions**:
   - Test with different bandwidth ranges
   - Test with different delay ranges
   - Test with different packet loss probabilities
   - Test with different network topologies

4. **Analyze the Impact**:
   - Compare training time and convergence with different network conditions
   - Analyze how different distributed strategies perform under network constraints

## Example: Simulating a Slow Network

Here's how to modify our MNIST example to simulate a slow network:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from exogym import LocalTrainer
from exogym.strategy import FedAvgStrategy
from exogym.strategy.optim import OptimSpec

# Import our network simulator
from network_simulator import NetworkSimulator, patch_distributed_functions

# ... (rest of the model definition)

def main():
    # ... (existing code)
    
    # Create and configure network simulator for a slow network
    network_sim = NetworkSimulator(
        bandwidth_range=(1, 5),     # 1-5 MB/s (slow bandwidth)
        delay_range=(0.1, 0.5),     # 100-500ms (high latency)
        packet_loss_prob=0.02,      # 2% packet loss
        jitter=0.3                  # 30% jitter
    )
    
    # Patch PyTorch distributed functions
    patch_distributed_functions(network_sim)
    
    # ... (rest of your code)
```

## Advanced: Simulating Real-World Network Conditions

For more realistic simulations, you can model real-world network conditions:

1. **Mobile Network**:
   - Bandwidth: 1-10 MB/s
   - Delay: 50-300ms
   - Packet Loss: 1-5%
   - High Jitter: 30-50%

2. **Datacenter Network**:
   - Bandwidth: 100-1000 MB/s
   - Delay: 0.1-5ms
   - Packet Loss: 0.001-0.1%
   - Low Jitter: 1-5%

3. **Cross-Continental Connection**:
   - Bandwidth: 10-100 MB/s
   - Delay: 100-500ms
   - Packet Loss: 0.1-1%
   - Medium Jitter: 10-20%

## Conclusion

By implementing network simulation in ExoGym, you can:

1. Test how distributed training algorithms perform under realistic network conditions
2. Compare different strategies for their robustness to network limitations
3. Develop new algorithms specifically designed for constrained network environments

This approach allows you to conduct comprehensive experiments on distributed training without needing multiple physical machines with actual network constraints.