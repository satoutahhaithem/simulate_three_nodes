#!/bin/bash
#
# Advanced Three-Node Simulation Script
# This script sets up and runs a distributed simulation across three nodes
# using available GPUs on a single machine.
#
# Author: Roo
# Date: June 18, 2025

# ===== Configuration Options =====
# You can modify these variables to customize the simulation

# Master node address and port
MASTER_ADDR="localhost"
MASTER_PORT="29500"

# Number of nodes to run
NUM_NODES=3

# Default simulation type (simple, mnist, mnist_network)
DEFAULT_SIM_TYPE="mnist_network"

# Network profile for network simulation (local, datacenter, wan, mobile, poor, custom)
NETWORK_PROFILE="wan"

# Network topology (fully_connected, ring, star)
NETWORK_TOPOLOGY="fully_connected"

# Training parameters
BATCH_SIZE=64
TEST_BATCH_SIZE=1000
EPOCHS=2
LEARNING_RATE=1.0

# GPU assignment (set to "auto" for automatic assignment, or specify GPU IDs like "0,1,2")
GPU_ASSIGNMENT="auto"

# Logging options (none, minimal, verbose)
LOG_LEVEL="minimal"

# Temporary directory for the distributed run
TEMP_DIR="./temp_distributed_run"

# ===== End of Configuration Options =====

# ===== Helper Functions =====

# Print colored output
print_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $1"
}

print_warning() {
    echo -e "\033[0;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

# Display help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -t, --type TYPE            Simulation type (simple, mnist, mnist_network)"
    echo "  -n, --nodes NUM            Number of nodes to run (default: 3)"
    echo "  -g, --gpus IDS             GPU IDs to use (comma-separated, e.g., '0,1,2')"
    echo "  -p, --port PORT            Master port (default: 29500)"
    echo "  -b, --batch-size SIZE      Batch size for training (default: 64)"
    echo "  -e, --epochs NUM           Number of epochs (default: 2)"
    echo "  -l, --log-level LEVEL      Logging level (none, minimal, verbose)"
    echo "  --network-profile PROFILE  Network profile for simulation"
    echo "  --network-topology TOPOLOGY Network topology (fully_connected, ring, star)"
    echo ""
    echo "Examples:"
    echo "  $0 --type mnist --gpus 0,1,2"
    echo "  $0 --type mnist_network --network-profile mobile --nodes 3"
    echo ""
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Clean up temporary files
cleanup() {
    print_info "Cleaning up temporary files..."
    rm -rf $TEMP_DIR
    
    # Kill any remaining processes
    if [ -d "$TEMP_DIR" ]; then
        for pid_file in $TEMP_DIR/*.pid; do
            if [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                if ps -p $pid > /dev/null; then
                    print_warning "Killing process $pid..."
                    kill -9 $pid 2>/dev/null
                fi
            fi
        done
    fi
    
    print_info "Cleanup completed."
}

# Check system requirements
check_requirements() {
    print_info "Checking system requirements..."
    
    # Check if Python is installed
    if ! command_exists python; then
        print_error "Python is not installed. Please install Python first."
        exit 1
    fi
    
    # Check if PyTorch is installed
    python -c "import torch" &> /dev/null
    if [ $? -ne 0 ]; then
        print_error "PyTorch is not installed. Please install PyTorch first."
        exit 1
    fi
    
    # Check if CUDA is available
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" != "True" ]; then
        print_warning "CUDA is not available. The simulation will run on CPU only."
        USE_GPU=false
    else
        USE_GPU=true
        
        # Check number of available GPUs
        NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
        print_info "Number of available GPUs: $NUM_GPUS"
        
        if [ "$NUM_GPUS" -lt "$NUM_NODES" ]; then
            print_warning "Less than $NUM_NODES GPUs detected. Some nodes will share GPUs or use CPU."
        fi
        
        # Get GPU information
        print_info "Available GPUs:"
        python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
    fi
    
    # Check if required scripts exist
    if [ "$SIM_TYPE" = "simple" ] && [ ! -f "simple_3node_test.py" ]; then
        print_error "simple_3node_test.py not found. Please make sure you're in the correct directory."
        exit 1
    elif [ "$SIM_TYPE" = "mnist" ] && [ ! -f "run_mnist_3nodes.py" ]; then
        print_error "run_mnist_3nodes.py not found. Please make sure you're in the correct directory."
        exit 1
    elif [ "$SIM_TYPE" = "mnist_network" ] && [ ! -f "run_mnist_3nodes_with_network_sim.py" ]; then
        print_error "run_mnist_3nodes_with_network_sim.py not found. Please make sure you're in the correct directory."
        exit 1
    fi
    
    print_success "System requirements check completed."
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--type)
                SIM_TYPE="$2"
                shift 2
                ;;
            -n|--nodes)
                NUM_NODES="$2"
                shift 2
                ;;
            -g|--gpus)
                GPU_ASSIGNMENT="$2"
                shift 2
                ;;
            -p|--port)
                MASTER_PORT="$2"
                shift 2
                ;;
            -b|--batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            -e|--epochs)
                EPOCHS="$2"
                shift 2
                ;;
            -l|--log-level)
                LOG_LEVEL="$2"
                shift 2
                ;;
            --network-profile)
                NETWORK_PROFILE="$2"
                shift 2
                ;;
            --network-topology)
                NETWORK_TOPOLOGY="$2"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate simulation type
    if [ "$SIM_TYPE" != "simple" ] && [ "$SIM_TYPE" != "mnist" ] && [ "$SIM_TYPE" != "mnist_network" ]; then
        print_error "Invalid simulation type: $SIM_TYPE"
        print_info "Valid types are: simple, mnist, mnist_network"
        exit 1
    fi
    
    # Validate log level
    if [ "$LOG_LEVEL" != "none" ] && [ "$LOG_LEVEL" != "minimal" ] && [ "$LOG_LEVEL" != "verbose" ]; then
        print_error "Invalid log level: $LOG_LEVEL"
        print_info "Valid log levels are: none, minimal, verbose"
        exit 1
    fi
}

# Assign GPUs to nodes
assign_gpus() {
    if [ "$USE_GPU" = false ]; then
        # If CUDA is not available, use CPU for all nodes
        for ((i=0; i<$NUM_NODES; i++)); do
            NODE_GPUS[$i]="cpu"
        done
        return
    fi
    
    if [ "$GPU_ASSIGNMENT" = "auto" ]; then
        # Automatic GPU assignment
        for ((i=0; i<$NUM_NODES; i++)); do
            NODE_GPUS[$i]=$(($i % $NUM_GPUS))
        done
    else
        # Manual GPU assignment
        IFS=',' read -ra GPU_IDS <<< "$GPU_ASSIGNMENT"
        if [ ${#GPU_IDS[@]} -lt "$NUM_NODES" ]; then
            print_warning "Not enough GPUs specified. Some nodes will share GPUs."
        fi
        
        for ((i=0; i<$NUM_NODES; i++)); do
            if [ $i -lt ${#GPU_IDS[@]} ]; then
                NODE_GPUS[$i]=${GPU_IDS[$i]}
            else
                NODE_GPUS[$i]=$(($i % ${#GPU_IDS[@]}))
            fi
        done
    fi
    
    # Print GPU assignments
    print_info "GPU assignments:"
    for ((i=0; i<$NUM_NODES; i++)); do
        if [ "${NODE_GPUS[$i]}" = "cpu" ]; then
            print_info "  Node $i -> CPU"
        else
            print_info "  Node $i -> GPU ${NODE_GPUS[$i]}"
        fi
    done
}

# Run a node with specific GPU
run_node() {
    local RANK=$1
    local WORLD_SIZE=$2
    local GPU_ID=$3
    local SCRIPT=$4
    
    # Create log directory if needed
    if [ "$LOG_LEVEL" != "none" ]; then
        mkdir -p "$TEMP_DIR/logs"
    fi
    
    # Prepare log redirection
    if [ "$LOG_LEVEL" = "none" ]; then
        LOG_REDIRECT="/dev/null"
    elif [ "$LOG_LEVEL" = "minimal" ]; then
        LOG_REDIRECT="$TEMP_DIR/logs/node_${RANK}.log"
    else
        LOG_REDIRECT="/dev/stdout"
    fi
    
    print_info "Starting node $RANK on GPU $GPU_ID..."
    
    # Set environment variables for this node
    if [ "$GPU_ID" = "cpu" ]; then
        export CUDA_VISIBLE_DEVICES=""
        DEVICE_ARG="cpu"
    else
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        DEVICE_ARG="cuda"
    fi
    
    export RANK=$RANK
    export WORLD_SIZE=$WORLD_SIZE
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    
    # Additional arguments for network simulation
    if [ "$SCRIPT" = "mnist_network" ]; then
        NETWORK_ARGS="--network-profile $NETWORK_PROFILE --network-topology $NETWORK_TOPOLOGY"
    else
        NETWORK_ARGS=""
    fi
    
    # Run the selected script
    case $SCRIPT in
        "simple")
            python simple_3node_test.py --rank $RANK --world-size $WORLD_SIZE > $LOG_REDIRECT 2>&1 &
            ;;
        "mnist")
            python run_mnist_3nodes.py --rank $RANK --world-size $WORLD_SIZE --device $DEVICE_ARG \
                --batch-size $BATCH_SIZE --test-batch-size $TEST_BATCH_SIZE \
                --epochs $EPOCHS --lr $LEARNING_RATE > $LOG_REDIRECT 2>&1 &
            ;;
        "mnist_network")
            python run_mnist_3nodes_with_network_sim.py --rank $RANK --world-size $WORLD_SIZE --device $DEVICE_ARG \
                --batch-size $BATCH_SIZE --test-batch-size $TEST_BATCH_SIZE \
                --epochs $EPOCHS --lr $LEARNING_RATE $NETWORK_ARGS > $LOG_REDIRECT 2>&1 &
            ;;
    esac
    
    # Store the PID
    echo $! > $TEMP_DIR/node_${RANK}.pid
}

# Monitor running nodes
monitor_nodes() {
    local start_time=$(date +%s)
    local all_done=false
    
    print_info "Monitoring nodes..."
    
    while [ "$all_done" = false ]; do
        all_done=true
        
        for ((i=0; i<$NUM_NODES; i++)); do
            pid_file="$TEMP_DIR/node_${i}.pid"
            if [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                if ps -p $pid > /dev/null; then
                    all_done=false
                    
                    # Print log snippets for verbose mode
                    if [ "$LOG_LEVEL" = "verbose" ]; then
                        log_file="$TEMP_DIR/logs/node_${i}.log"
                        if [ -f "$log_file" ]; then
                            echo "--- Recent logs from Node $i ---"
                            tail -n 5 "$log_file"
                            echo "-------------------------------"
                        fi
                    fi
                fi
            fi
        done
        
        if [ "$all_done" = false ]; then
            sleep 5
            
            # Print a progress indicator
            if [ "$LOG_LEVEL" = "minimal" ]; then
                elapsed=$(($(date +%s) - start_time))
                printf "\rRunning for %02d:%02d..." $((elapsed/60)) $((elapsed%60))
            fi
        fi
    done
    
    if [ "$LOG_LEVEL" = "minimal" ]; then
        echo ""  # New line after progress indicator
    fi
    
    print_success "All nodes completed."
}

# ===== Main Script =====

# Set default simulation type
SIM_TYPE=${DEFAULT_SIM_TYPE}

# Parse command line arguments
parse_arguments "$@"

# Display banner
echo "================================================"
echo "      Advanced Three-Node Simulation Script      "
echo "================================================"
echo "Simulation Type: $SIM_TYPE"
echo "Number of Nodes: $NUM_NODES"
echo "GPU Assignment: $GPU_ASSIGNMENT"
echo "Log Level: $LOG_LEVEL"
if [ "$SIM_TYPE" = "mnist_network" ]; then
    echo "Network Profile: $NETWORK_PROFILE"
    echo "Network Topology: $NETWORK_TOPOLOGY"
fi
echo "================================================"

# Register the cleanup function to be called on exit
trap cleanup EXIT

# Create temporary directory
mkdir -p $TEMP_DIR

# Check system requirements
check_requirements

# Initialize GPU assignment array
declare -a NODE_GPUS

# Assign GPUs to nodes
assign_gpus

# Start the nodes
for ((i=0; i<$NUM_NODES; i++)); do
    run_node $i $NUM_NODES ${NODE_GPUS[$i]} $SIM_TYPE
done

# Monitor the nodes
monitor_nodes

# Print summary
echo "================================================"
print_success "Three-node simulation completed successfully!"
echo "================================================"

# If logs were saved, show where to find them
if [ "$LOG_LEVEL" = "minimal" ]; then
    print_info "Logs are available in: $TEMP_DIR/logs/"
fi

exit 0