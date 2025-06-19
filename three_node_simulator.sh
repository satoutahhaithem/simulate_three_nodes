#!/bin/bash
#
# Interactive Three-Node Simulation Script
# This script provides an interactive interface to set up and run
# a distributed simulation across three nodes using available GPUs.
#

# ===== Terminal Colors =====
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ===== Helper Functions =====

# Print colored output
print_header() {
    echo -e "${PURPLE}==== $1 ====${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_prompt() {
    echo -e "${CYAN}[PROMPT]${NC} $1"
}

# Display a spinner while waiting
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while ps -p $pid > /dev/null; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Clean up temporary files
cleanup() {
    print_info "Cleaning up temporary files..."
    
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
        
        # Remove temporary directory
        rm -rf $TEMP_DIR
    fi
    
    print_info "Cleanup completed."
}

# ===== Main Interactive Script =====

# Display welcome banner
clear
echo -e "${PURPLE}"
echo "============================================================"
echo "                                                            "
echo "             THREE-NODE SIMULATION LAUNCHER                 "
echo "                                                            "
echo "============================================================"
echo -e "${NC}"
echo "This script will guide you through setting up and running a"
echo "distributed simulation across three nodes using your GPUs."
echo ""

# Create temporary directory
TEMP_DIR="./temp_distributed_run"
mkdir -p $TEMP_DIR

# Register the cleanup function to be called on exit
trap cleanup EXIT

# ===== Step 1: System Check =====
print_header "SYSTEM CHECK"

# Check if Python is installed
if ! command_exists python; then
    print_error "Python is not installed. Please install Python first."
    exit 1
fi

print_info "Python is installed."

# Check if PyTorch is installed
python -c "import torch" &> /dev/null
if [ $? -ne 0 ]; then
    print_error "PyTorch is not installed. Please install PyTorch first."
    exit 1
fi

print_info "PyTorch is installed."

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
    
    # Get GPU information
    echo "Available GPUs:"
    python -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
fi

echo ""
print_success "System check completed."
echo ""

# ===== Step 2: Configuration =====
print_header "CONFIGURATION"

# Set default values
MASTER_ADDR="localhost"
MASTER_PORT="29500"
NUM_NODES=3
BATCH_SIZE=64
TEST_BATCH_SIZE=1000
EPOCHS=2
LEARNING_RATE=1.0
LOG_LEVEL="minimal"

# Choose simulation type
echo "Please select the simulation type:"
echo "1) Simple 3-node test (no PyTorch required)"
echo "2) MNIST 3-node training"
echo "3) MNIST 3-node training with network simulation"
print_prompt "Enter your choice (1-3) [default: 3]: "
read choice

case $choice in
    1)
        SIM_TYPE="simple"
        print_info "Selected: Simple 3-node test"
        ;;
    2)
        SIM_TYPE="mnist"
        print_info "Selected: MNIST 3-node training"
        ;;
    3|"")
        SIM_TYPE="mnist_network"
        print_info "Selected: MNIST 3-node training with network simulation"
        ;;
    *)
        print_error "Invalid choice. Using default: MNIST 3-node training with network simulation"
        SIM_TYPE="mnist_network"
        ;;
esac

# If network simulation is selected, configure network parameters
if [ "$SIM_TYPE" = "mnist_network" ]; then
    echo ""
    echo "Please select the network profile:"
    echo "1) Local (1-10 GB/s, 0.1-1ms latency)"
    echo "2) Datacenter (100-1000 MB/s, 0.5-5ms latency)"
    echo "3) WAN (10-100 MB/s, 10-50ms latency)"
    echo "4) Mobile (1-10 MB/s, 50-200ms latency)"
    echo "5) Poor (0.1-1 MB/s, 200-1000ms latency)"
    print_prompt "Enter your choice (1-5) [default: 3]: "
    read net_choice
    
    case $net_choice in
        1)
            NETWORK_PROFILE="local"
            print_info "Selected: Local network profile"
            ;;
        2)
            NETWORK_PROFILE="datacenter"
            print_info "Selected: Datacenter network profile"
            ;;
        3|"")
            NETWORK_PROFILE="wan"
            print_info "Selected: WAN network profile"
            ;;
        4)
            NETWORK_PROFILE="mobile"
            print_info "Selected: Mobile network profile"
            ;;
        5)
            NETWORK_PROFILE="poor"
            print_info "Selected: Poor network profile"
            ;;
        *)
            print_error "Invalid choice. Using default: WAN network profile"
            NETWORK_PROFILE="wan"
            ;;
    esac
    
    echo ""
    echo "Please select the network topology:"
    echo "1) Fully connected (all nodes connected to all other nodes)"
    echo "2) Ring (each node connected only to adjacent nodes)"
    echo "3) Star (node 0 is the center, connected to all other nodes)"
    print_prompt "Enter your choice (1-3) [default: 1]: "
    read topo_choice
    
    case $topo_choice in
        1|"")
            NETWORK_TOPOLOGY="fully_connected"
            print_info "Selected: Fully connected topology"
            ;;
        2)
            NETWORK_TOPOLOGY="ring"
            print_info "Selected: Ring topology"
            ;;
        3)
            NETWORK_TOPOLOGY="star"
            print_info "Selected: Star topology"
            ;;
        *)
            print_error "Invalid choice. Using default: Fully connected topology"
            NETWORK_TOPOLOGY="fully_connected"
            ;;
    esac
fi

# Configure GPU assignment if GPUs are available
if [ "$USE_GPU" = true ]; then
    echo ""
    echo "Please select GPU assignment method:"
    echo "1) Automatic (assign GPUs automatically)"
    echo "2) Manual (specify which GPUs to use)"
    print_prompt "Enter your choice (1-2) [default: 1]: "
    read gpu_choice
    
    case $gpu_choice in
        1|"")
            GPU_ASSIGNMENT="auto"
            print_info "Selected: Automatic GPU assignment"
            ;;
        2)
            print_prompt "Enter GPU IDs to use (comma-separated, e.g., '0,1,2'): "
            read gpu_ids
            if [ -z "$gpu_ids" ]; then
                print_error "No GPU IDs provided. Using automatic assignment."
                GPU_ASSIGNMENT="auto"
            else
                GPU_ASSIGNMENT="$gpu_ids"
                print_info "Selected: Manual GPU assignment ($GPU_ASSIGNMENT)"
            fi
            ;;
        *)
            print_error "Invalid choice. Using automatic GPU assignment."
            GPU_ASSIGNMENT="auto"
            ;;
    esac
else
    GPU_ASSIGNMENT="cpu"
    print_info "Using CPU for all nodes (CUDA not available)"
fi

# Configure training parameters for MNIST simulations
if [ "$SIM_TYPE" = "mnist" ] || [ "$SIM_TYPE" = "mnist_network" ]; then
    echo ""
    print_prompt "Enter batch size [default: 64]: "
    read input_batch_size
    if [ ! -z "$input_batch_size" ]; then
        BATCH_SIZE=$input_batch_size
    fi
    print_info "Batch size: $BATCH_SIZE"
    
    print_prompt "Enter number of epochs [default: 2]: "
    read input_epochs
    if [ ! -z "$input_epochs" ]; then
        EPOCHS=$input_epochs
    fi
    print_info "Epochs: $EPOCHS"
fi

# Configure logging level
echo ""
echo "Please select logging level:"
echo "1) None (no logs)"
echo "2) Minimal (save logs to files)"
echo "3) Verbose (show all logs in terminal)"
print_prompt "Enter your choice (1-3) [default: 2]: "
read log_choice

case $log_choice in
    1)
        LOG_LEVEL="none"
        print_info "Selected: No logging"
        ;;
    2|"")
        LOG_LEVEL="minimal"
        print_info "Selected: Minimal logging"
        ;;
    3)
        LOG_LEVEL="verbose"
        print_info "Selected: Verbose logging"
        ;;
    *)
        print_error "Invalid choice. Using default: Minimal logging"
        LOG_LEVEL="minimal"
        ;;
esac

echo ""
print_success "Configuration completed."
echo ""

# ===== Step 3: Verification =====
print_header "VERIFICATION"

echo "Please verify the following configuration:"
echo ""
echo "Simulation Type: $SIM_TYPE"
echo "Number of Nodes: $NUM_NODES"

if [ "$SIM_TYPE" = "mnist_network" ]; then
    echo "Network Profile: $NETWORK_PROFILE"
    echo "Network Topology: $NETWORK_TOPOLOGY"
fi

if [ "$SIM_TYPE" = "mnist" ] || [ "$SIM_TYPE" = "mnist_network" ]; then
    echo "Batch Size: $BATCH_SIZE"
    echo "Epochs: $EPOCHS"
fi

echo "GPU Assignment: $GPU_ASSIGNMENT"
echo "Log Level: $LOG_LEVEL"
echo ""

print_prompt "Is this configuration correct? (y/n) [default: y]: "
read confirm

if [ "$confirm" = "n" ]; then
    print_info "Configuration cancelled. Please run the script again."
    exit 0
fi

echo ""
print_success "Configuration verified."
echo ""

# ===== Step 4: Preparation =====
print_header "PREPARATION"

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

# Create log directory if needed
if [ "$LOG_LEVEL" != "none" ]; then
    mkdir -p "$TEMP_DIR/logs"
fi

# Assign GPUs to nodes
declare -a NODE_GPUS

if [ "$USE_GPU" = false ] || [ "$GPU_ASSIGNMENT" = "cpu" ]; then
    # If CUDA is not available, use CPU for all nodes
    for ((i=0; i<$NUM_NODES; i++)); do
        NODE_GPUS[$i]="cpu"
    done
elif [ "$GPU_ASSIGNMENT" = "auto" ]; then
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
echo "GPU assignments:"
for ((i=0; i<$NUM_NODES; i++)); do
    if [ "${NODE_GPUS[$i]}" = "cpu" ]; then
        echo "  Node $i -> CPU"
    else
        echo "  Node $i -> GPU ${NODE_GPUS[$i]}"
    fi
done

echo ""
print_success "Preparation completed."
echo ""

# ===== Step 5: Execution =====
print_header "EXECUTION"

# Function to run a node with specific GPU
run_node() {
    local RANK=$1
    local WORLD_SIZE=$2
    local GPU_ID=$3
    local SCRIPT=$4
    
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

print_info "Starting simulation with $NUM_NODES nodes..."
echo ""

# Start the nodes
for ((i=0; i<$NUM_NODES; i++)); do
    run_node $i $NUM_NODES ${NODE_GPUS[$i]} $SIM_TYPE
done

print_info "All nodes started. Monitoring progress..."
echo ""

# Monitor running nodes
start_time=$(date +%s)
all_done=false

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

echo ""
print_success "Simulation completed successfully!"
echo ""

# ===== Step 6: Results =====
print_header "RESULTS"

# Calculate total runtime
end_time=$(date +%s)
total_runtime=$((end_time - start_time))
minutes=$((total_runtime / 60))
seconds=$((total_runtime % 60))

echo "Total runtime: ${minutes}m ${seconds}s"
echo ""

# Show log file locations if minimal logging was used
if [ "$LOG_LEVEL" = "minimal" ]; then
    echo "Log files are available in: $TEMP_DIR/logs/"
    echo ""
    
    # Ask if user wants to see log summary
    print_prompt "Would you like to see a summary of the logs? (y/n) [default: y]: "
    read show_logs
    
    if [ "$show_logs" != "n" ]; then
        echo ""
        echo "Log summary:"
        echo ""
        
        for ((i=0; i<$NUM_NODES; i++)); do
            log_file="$TEMP_DIR/logs/node_${i}.log"
            if [ -f "$log_file" ]; then
                echo "=== Node $i ==="
                # Show the last 10 lines of each log file
                tail -n 10 "$log_file"
                echo ""
            fi
        done
    fi
fi

# Ask if user wants to keep the logs
if [ "$LOG_LEVEL" != "none" ]; then
    print_prompt "Would you like to keep the log files? (y/n) [default: y]: "
    read keep_logs
    
    if [ "$keep_logs" = "n" ]; then
        rm -rf "$TEMP_DIR/logs"
        print_info "Log files deleted."
    else
        print_info "Log files preserved in: $TEMP_DIR/logs/"
    fi
fi

echo ""
print_success "Three-node simulation completed successfully!"
echo -e "${PURPLE}============================================================${NC}"