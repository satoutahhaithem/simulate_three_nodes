#!/bin/bash

# Script to run a 3-node simulation using the three available GPUs
# GPU 0: Quadro RTX 6000
# GPU 1: NVIDIA TITAN RTX
# GPU 2: NVIDIA TITAN RTX

echo "Starting 3-node simulation with GPU acceleration"
echo "================================================"

# Check if PyTorch is installed
python -c "import torch" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Error: PyTorch is not installed. Please install PyTorch first."
    exit 1
fi

# Check if CUDA is available
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo "Error: CUDA is not available. Please check your PyTorch installation."
    exit 1
fi

# Check number of available GPUs
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Number of available GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -lt 3 ]; then
    echo "Warning: Less than 3 GPUs detected. Some nodes will share GPUs or use CPU."
fi

# Create a temporary directory for the distributed run
TEMP_DIR="./temp_distributed_run"
mkdir -p $TEMP_DIR

# Function to clean up temporary files
cleanup() {
    echo "Cleaning up temporary files..."
    rm -rf $TEMP_DIR
    echo "Done."
}

# Register the cleanup function to be called on exit
trap cleanup EXIT

# Choose which simulation to run
echo "Please select the simulation to run:"
echo "1. Simple 3-node test (no PyTorch required)"
echo "2. MNIST 3-node training"
echo "3. MNIST 3-node training with network simulation"
read -p "Enter your choice (1-3): " CHOICE

# Set the master address and port for distributed training
MASTER_ADDR="localhost"
MASTER_PORT="29500"

# Function to run a node with specific GPU
run_node() {
    local RANK=$1
    local WORLD_SIZE=$2
    local GPU_ID=$3
    local SCRIPT=$4
    
    echo "Starting node $RANK on GPU $GPU_ID..."
    
    # Set environment variables for this node
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export RANK=$RANK
    export WORLD_SIZE=$WORLD_SIZE
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    
    # Run the selected script
    case $SCRIPT in
        "simple")
            python simple_3node_test.py --rank $RANK --world-size $WORLD_SIZE &
            ;;
        "mnist")
            python run_mnist_3nodes.py --rank $RANK --world-size $WORLD_SIZE --device cuda &
            ;;
        "mnist_network")
            python run_mnist_3nodes_with_network_sim.py --rank $RANK --world-size $WORLD_SIZE --device cuda &
            ;;
    esac
    
    # Store the PID
    echo $! > $TEMP_DIR/node_${RANK}.pid
}

# Run the selected simulation
case $CHOICE in
    1)
        echo "Running simple 3-node test..."
        SCRIPT="simple"
        ;;
    2)
        echo "Running MNIST 3-node training..."
        SCRIPT="mnist"
        ;;
    3)
        echo "Running MNIST 3-node training with network simulation..."
        SCRIPT="mnist_network"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Start the nodes on different GPUs
run_node 0 3 0 $SCRIPT
run_node 1 3 1 $SCRIPT
run_node 2 3 2 $SCRIPT

echo "All nodes started. Waiting for completion..."

# Wait for all nodes to complete
wait

echo "================================================"
echo "3-node simulation completed successfully!"