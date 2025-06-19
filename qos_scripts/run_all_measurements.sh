#!/bin/bash

# Get the IP addresses of the containers
NODE1_IP=$(getent hosts node1 | awk '{ print $1 }')
NODE2_IP=$(getent hosts node2 | awk '{ print $1 }')
NODE3_IP=$(getent hosts node3 | awk '{ print $1 }')

echo "Node 1 IP: $NODE1_IP"
echo "Node 2 IP: $NODE2_IP"
echo "Node 3 IP: $NODE3_IP"

# Create output directory
mkdir -p /qos_data

# Measure bandwidth
echo "===== MEASURING BANDWIDTH ====="
# Start iperf servers on node2 and node3
echo "Starting iperf servers on node2 and node3..."
docker exec node2 python3 /qos_scripts/measure_bandwidth.py --server &
docker exec node3 python3 /qos_scripts/measure_bandwidth.py --server &
sleep 2

# Measure bandwidth from node1 to node2 and node3
echo "Measuring bandwidth from node1 to node2..."
python3 measure_bandwidth.py --client $NODE2_IP --duration 5 --iterations 3
echo "Measuring bandwidth from node1 to node3..."
python3 measure_bandwidth.py --client $NODE3_IP --duration 5 --iterations 3

# Measure latency
echo "===== MEASURING LATENCY ====="
echo "Measuring latency from node1 to node2..."
python3 measure_latency.py --host $NODE2_IP --count 20
echo "Measuring latency from node1 to node3..."
python3 measure_latency.py --host $NODE3_IP --count 20

# Measure packet loss
echo "===== MEASURING PACKET LOSS ====="
echo "Measuring packet loss from node1 to node2..."
python3 measure_packet_loss.py --host $NODE2_IP --count 100
echo "Measuring packet loss from node1 to node3..."
python3 measure_packet_loss.py --host $NODE3_IP --count 100

echo "===== ALL MEASUREMENTS COMPLETE ====="
echo "Results are saved in /qos_data directory"
