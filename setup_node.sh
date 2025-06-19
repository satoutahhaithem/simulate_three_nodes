#!/bin/bash
cd /opt/DistributedSim
source .venv/bin/activate
pip install torchvision
cp example/mnist.py example/mnist_diloco.py
sed -i 's/from exogym.strategy import FedAvgStrategy/from exogym.strategy import DiLoCoStrategy/' example/mnist_diloco.py
sed -i 's/strategy = FedAvgStrategy(/strategy = DiLoCoStrategy(/' example/mnist_diloco.py
