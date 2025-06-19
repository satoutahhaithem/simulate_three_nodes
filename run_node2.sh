#!/bin/bash
cd /opt/DistributedSim
source .venv/bin/activate
export MASTER_ADDR=node0
export MASTER_PORT=29500
export WORLD_SIZE=3
export RANK=2
python example/mnist_diloco.py --rank 2 --world-size 3 --device cuda
