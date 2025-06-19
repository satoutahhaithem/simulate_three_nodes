#!/bin/bash

# Default values
STRATEGY="diloco"
NETWORK="wan"
TOPOLOGY="fully_connected"
H=10
EPOCHS=2
BATCH_SIZE=64

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --strategy)
      STRATEGY="$2"
      shift 2
      ;;
    --network)
      NETWORK="$2"
      shift 2
      ;;
    --topology)
      TOPOLOGY="$2"
      shift 2
      ;;
    --h)
      H="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Running experiment with:"
echo "- Strategy: $STRATEGY"
echo "- Network: $NETWORK"
echo "- Topology: $TOPOLOGY"
echo "- H parameter: $H"
echo "- Epochs: $EPOCHS"
echo "- Batch size: $BATCH_SIZE"

python run_mnist_3nodes_with_network_sim_fixed.py \
  --strategy $STRATEGY \
  --network $NETWORK \
  --topology $TOPOLOGY \
  --h $H \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE
