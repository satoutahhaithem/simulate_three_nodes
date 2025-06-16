# ExoGym Documentation

Welcome to the ExoGym documentation. This collection of documents will help you understand how ExoGym works and how to use it for simulating distributed training with network conditions.

## Table of Contents

1. [ExoGym Code Explanation](ExoGym_Code_Explanation.md)
   - Detailed explanation of how ExoGym works internally
   - Overview of the architecture and key components
   - Execution flow and class relationships

2. [Network Simulation Guide](ExoGym_Network_Simulation_Guide.md)
   - Comprehensive guide to network simulation in ExoGym
   - How to simulate bandwidth limitations and communication delays
   - Implementing different network topologies
   - Advanced network simulation techniques

3. [Network Simulation Usage Guide](Network_Simulation_Usage_Guide.md)
   - Practical instructions for using the network simulation code
   - How to run experiments with different network conditions
   - Analyzing the results of network simulation
   - Troubleshooting common issues

4. [README](README.md)
   - Overview of the project
   - Getting started instructions
   - Features and capabilities

## Quick Start

To get started with ExoGym and network simulation:

1. Make sure you have ExoGym installed in a virtual environment:
   ```bash
   cd DistributedSim
   source venv/bin/activate
   ```

2. Run the basic MNIST example with 3 nodes:
   ```bash
   python ../run_mnist_3nodes.py
   ```

3. Run the MNIST example with network simulation:
   ```bash
   python ../run_mnist_3nodes_with_network_sim.py
   ```

## Learning Path

If you're new to ExoGym, we recommend following this learning path:

1. Start with the [README](README.md) to get an overview of the project
2. Read the [ExoGym Code Explanation](ExoGym_Code_Explanation.md) to understand how ExoGym works
3. Explore the [Network Simulation Guide](ExoGym_Network_Simulation_Guide.md) to learn about network simulation
4. Follow the [Network Simulation Usage Guide](Network_Simulation_Usage_Guide.md) to run your own experiments

## Additional Resources

- The `network_simulator.py` file contains the implementation of the network simulator
- The `run_mnist_3nodes_with_network_sim.py` file provides a working example of network simulation