#!/usr/bin/env python3
import subprocess
import re
import time
import argparse
import os

def measure_packet_loss(host, count=100):
    """Measure packet loss to a host"""
    print(f"Measuring packet loss to {host}...")
    result = subprocess.run(
        ["ping", "-c", str(count), host],
        stdout=subprocess.PIPE,
        text=True
    )
    
    # Parse ping statistics
    match = re.search(r"(\d+) packets transmitted, (\d+) received", result.stdout)
    if match:
        transmitted = int(match.group(1))
        received = int(match.group(2))
        loss_count = transmitted - received
        loss_percent = (loss_count / transmitted) * 100
        
        print(f"Packets transmitted: {transmitted}")
        print(f"Packets received: {received}")
        print(f"Packets lost: {loss_count}")
        print(f"Packet loss: {loss_percent:.2f}%")
        
        return transmitted, received, loss_count, loss_percent
    else:
        print("No packet loss statistics found")
        return 0, 0, 0, 0

def save_results(transmitted, received, loss_count, loss_percent, host, output_dir="/qos_data"):
    """Save results to file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{output_dir}/packet_loss_{host.replace('.', '_')}_{timestamp}.txt"
    
    with open(filename, "w") as f:
        f.write(f"Packet loss measurements to {host}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Packets transmitted: {transmitted}\n")
        f.write(f"Packets received: {received}\n")
        f.write(f"Packets lost: {loss_count}\n")
        f.write(f"Packet loss: {loss_percent:.2f}%\n")
    
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Measure packet loss between containers')
    parser.add_argument('--host', type=str, required=True, help='Host to ping')
    parser.add_argument('--count', type=int, default=100, help='Number of pings')
    
    args = parser.parse_args()
    
    transmitted, received, loss_count, loss_percent = measure_packet_loss(args.host, args.count)
    save_results(transmitted, received, loss_count, loss_percent, args.host)

if __name__ == "__main__":
    main()
