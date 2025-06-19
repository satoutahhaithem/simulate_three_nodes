#!/bin/bash

# Run experiments with different strategies and network profiles
echo "Running experiments with different strategies and network profiles..."

# Test FedAvg strategy with different network profiles
echo "Testing FedAvg strategy..."
python run_mnist_3nodes_with_network_sim_fixed.py --strategy fedavg --network local --h 1
python run_mnist_3nodes_with_network_sim_fixed.py --strategy fedavg --network datacenter --h 5
python run_mnist_3nodes_with_network_sim_fixed.py --strategy fedavg --network wan --h 10
python run_mnist_3nodes_with_network_sim_fixed.py --strategy fedavg --network mobile --h 20
python run_mnist_3nodes_with_network_sim_fixed.py --strategy fedavg --network poor --h 50

# Test DiLoCo strategy with different network profiles
echo "Testing DiLoCo strategy..."
python run_mnist_3nodes_with_network_sim_fixed.py --strategy diloco --network local --h 1
python run_mnist_3nodes_with_network_sim_fixed.py --strategy diloco --network datacenter --h 5
python run_mnist_3nodes_with_network_sim_fixed.py --strategy diloco --network wan --h 10
python run_mnist_3nodes_with_network_sim_fixed.py --strategy diloco --network mobile --h 20
python run_mnist_3nodes_with_network_sim_fixed.py --strategy diloco --network poor --h 50

# Test SPARTA strategy with different network profiles
echo "Testing SPARTA strategy..."
python run_mnist_3nodes_with_network_sim_fixed.py --strategy sparta --network local --h 1
python run_mnist_3nodes_with_network_sim_fixed.py --strategy sparta --network datacenter --h 5
python run_mnist_3nodes_with_network_sim_fixed.py --strategy sparta --network wan --h 10
python run_mnist_3nodes_with_network_sim_fixed.py --strategy sparta --network mobile --h 20
python run_mnist_3nodes_with_network_sim_fixed.py --strategy sparta --network poor --h 50

echo "All experiments completed!"
