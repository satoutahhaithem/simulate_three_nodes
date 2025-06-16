import time
import random
import torch
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
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        print(f"[Rank {self.rank}] Network Simulator initialized with:")
        print(f"[Rank {self.rank}] - Bandwidth: {bandwidth_range[0]}-{bandwidth_range[1]} MB/s")
        print(f"[Rank {self.rank}] - Delay: {delay_range[0]*1000:.1f}-{delay_range[1]*1000:.1f} ms")
        print(f"[Rank {self.rank}] - Packet Loss: {packet_loss_prob*100:.2f}%")
        print(f"[Rank {self.rank}] - Jitter: {jitter*100:.2f}%")
        
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
        
        if dist.is_initialized():
            rank = dist.get_rank()
            print(f"[Rank {rank}] Network Topology: {topology_type}")
        
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


def patch_distributed_functions(network_simulator, topology=None):
    """
    Patch PyTorch distributed functions to simulate network conditions.
    
    Args:
        network_simulator: NetworkSimulator instance
        topology: Optional NetworkTopology instance
    """
    # Store original functions
    original_all_reduce = dist.all_reduce
    original_broadcast = dist.broadcast
    original_reduce = dist.reduce
    original_all_gather = dist.all_gather
    original_gather = dist.gather
    original_scatter = dist.scatter
    
    @wraps(dist.all_reduce)
    def patched_all_reduce(tensor, *args, **kwargs):
        # Get current rank
        rank = dist.get_rank()
        
        # Simulate delay
        network_simulator.simulate_delay()
        
        # Simulate bandwidth limitation based on tensor size
        data_size = tensor.element_size() * tensor.nelement()
        network_simulator.simulate_bandwidth(data_size)
        
        # Simulate packet loss
        if network_simulator.simulate_packet_loss():
            # In real networks, packet loss would trigger retransmission
            # Here we just add extra delay to simulate that
            print(f"[Rank {rank}] Simulating packet loss in all_reduce")
            network_simulator.simulate_delay()
        
        # Call the original function
        return original_all_reduce(tensor, *args, **kwargs)
    
    @wraps(dist.broadcast)
    def patched_broadcast(tensor, src, *args, **kwargs):
        # Get current rank
        rank = dist.get_rank()
        
        # Check if communication is allowed in the topology
        if topology and not topology.can_communicate(src, rank) and src != rank:
            raise RuntimeError(f"Rank {rank} cannot receive broadcast from rank {src} in the current topology")
        
        # Simulate delay
        network_simulator.simulate_delay()
        
        # Simulate bandwidth limitation based on tensor size
        data_size = tensor.element_size() * tensor.nelement()
        network_simulator.simulate_bandwidth(data_size)
        
        # Simulate packet loss
        if network_simulator.simulate_packet_loss():
            print(f"[Rank {rank}] Simulating packet loss in broadcast")
            network_simulator.simulate_delay()
        
        # Call the original function
        return original_broadcast(tensor, src, *args, **kwargs)
    
    # Replace the original functions with our patched versions
    dist.all_reduce = patched_all_reduce
    dist.broadcast = patched_broadcast
    
    # Similarly patch other distributed functions
    # (Implementation for other functions follows the same pattern)
    
    print(f"[Rank {network_simulator.rank}] PyTorch distributed functions patched with network simulation")
    
    # Return a function to restore original functions if needed
    def restore_original_functions():
        dist.all_reduce = original_all_reduce
        dist.broadcast = original_broadcast
        dist.reduce = original_reduce
        dist.all_gather = original_all_gather
        dist.gather = original_gather
        dist.scatter = original_scatter
        print(f"[Rank {network_simulator.rank}] Original PyTorch distributed functions restored")
        
    return restore_original_functions


# Example network profiles
NETWORK_PROFILES = {
    "local": {
        "bandwidth_range": (1000, 10000),  # 1-10 GB/s
        "delay_range": (0.0001, 0.001),    # 0.1-1ms
        "packet_loss_prob": 0.0,
        "jitter": 0.01
    },
    "datacenter": {
        "bandwidth_range": (100, 1000),    # 100-1000 MB/s
        "delay_range": (0.0005, 0.005),    # 0.5-5ms
        "packet_loss_prob": 0.0001,
        "jitter": 0.05
    },
    "wan": {
        "bandwidth_range": (10, 100),      # 10-100 MB/s
        "delay_range": (0.01, 0.05),       # 10-50ms
        "packet_loss_prob": 0.001,
        "jitter": 0.1
    },
    "mobile": {
        "bandwidth_range": (1, 10),        # 1-10 MB/s
        "delay_range": (0.05, 0.2),        # 50-200ms
        "packet_loss_prob": 0.01,
        "jitter": 0.3
    },
    "poor": {
        "bandwidth_range": (0.1, 1),       # 0.1-1 MB/s
        "delay_range": (0.2, 1.0),         # 200-1000ms
        "packet_loss_prob": 0.05,
        "jitter": 0.5
    }
}


def create_network_simulator(profile="datacenter", custom_params=None):
    """
    Create a NetworkSimulator with predefined or custom parameters.
    
    Args:
        profile: String name of predefined profile or "custom"
        custom_params: Dict of custom parameters (used if profile="custom")
    
    Returns:
        NetworkSimulator instance
    """
    if profile == "custom" and custom_params:
        return NetworkSimulator(**custom_params)
    
    if profile in NETWORK_PROFILES:
        return NetworkSimulator(**NETWORK_PROFILES[profile])
    
    # Default to datacenter profile
    print(f"Warning: Profile '{profile}' not found, using 'datacenter' profile")
    return NetworkSimulator(**NETWORK_PROFILES["datacenter"])