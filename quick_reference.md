# Three-Node Simulation Quick Reference

## Basic Usage

```bash
# Make the script executable
chmod +x run_three_nodes_gpu_advanced.sh

# Run with default settings
./run_three_nodes_gpu_advanced.sh

# Run with specific simulation type
./run_three_nodes_gpu_advanced.sh --type mnist_network

# Run with specific GPU assignment
./run_three_nodes_gpu_advanced.sh --gpus 0,1,2

# Get help
./run_three_nodes_gpu_advanced.sh --help
```

## Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `-t, --type` | Simulation type (simple, mnist, mnist_network) | `--type mnist` |
| `-n, --nodes` | Number of nodes to run | `--nodes 3` |
| `-g, --gpus` | GPU IDs to use (comma-separated) | `--gpus 0,1,2` |
| `-b, --batch-size` | Batch size for training | `--batch-size 128` |
| `-e, --epochs` | Number of epochs | `--epochs 5` |
| `-l, --log-level` | Logging level (none, minimal, verbose) | `--log-level verbose` |
| `--network-profile` | Network profile for simulation | `--network-profile mobile` |
| `--network-topology` | Network topology | `--network-topology ring` |

## Network Profiles

| Profile | Description |
|---------|-------------|
| `local` | High bandwidth (1-10 GB/s), very low latency (0.1-1ms) |
| `datacenter` | Good bandwidth (100-1000 MB/s), low latency (0.5-5ms) |
| `wan` | Medium bandwidth (10-100 MB/s), medium latency (10-50ms) |
| `mobile` | Low bandwidth (1-10 MB/s), high latency (50-200ms) |
| `poor` | Very low bandwidth (0.1-1 MB/s), very high latency (200-1000ms) |

## Network Topologies

| Topology | Description |
|----------|-------------|
| `fully_connected` | All nodes connected to all other nodes |
| `ring` | Each node connected only to adjacent nodes |
| `star` | Node 0 is the center, connected to all other nodes |

## Examples

```bash
# Run MNIST training with 3 nodes on GPUs 0, 1, and 2
./run_three_nodes_gpu_advanced.sh --type mnist --gpus 0,1,2

# Run network simulation with mobile network profile
./run_three_nodes_gpu_advanced.sh --type mnist_network --network-profile mobile

# Run with ring topology and verbose logging
./run_three_nodes_gpu_advanced.sh --network-topology ring --log-level verbose

# Run with custom batch size and epochs
./run_three_nodes_gpu_advanced.sh --batch-size 128 --epochs 5
```

## Checking Logs

Logs are stored in the `temp_distributed_run/logs/` directory:

```bash
# View logs for node 0
cat temp_distributed_run/logs/node_0.log

# Monitor logs in real-time
tail -f temp_distributed_run/logs/node_0.log
```

## Troubleshooting

If the simulation fails:

1. Check GPU availability: `nvidia-smi`
2. Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Try running with `--log-level verbose` for more detailed output
4. Check for port conflicts and try a different port: `--port 29501`
5. If nodes hang, try killing the processes: `pkill -f python`