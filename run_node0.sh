#!/bin/bash
cd /opt/DistributedSim
source .venv/bin/activate
export MASTER_ADDR=node0
export MASTER_PORT=29500
export WORLD_SIZE=3
export RANK=0
python example/mnist_diloco.py --rank 0 --world-size 3 --device cuda
