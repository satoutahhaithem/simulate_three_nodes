#!/usr/bin/env python3
import subprocess
import json
import time
import argparse
import os

def run_iperf_server():
    """Start iperf3 server"""
    subprocess.Popen(["iperf3", "-s"], stdout=subprocess.PIPE)
    print("iperf3 server started")

def run_iperf_client(server_ip, duration=10):
    """Run iperf3 client and return results"""
    print(f"Measuring bandwidth to {server_ip}...")
    result = subprocess.run(
        ["iperf3", "-c", server_ip, "-t", str(duration), "-J"],
        stdout=subprocess.PIPE,
        text=True
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Error parsing iperf3 output")
        print(result.stdout)
        return None

def measure_bandwidth(server_ip, duration=10, iterations=3):
    """Measure bandwidth multiple times and return average"""
    results = []
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        result = run_iperf_client(server_ip, duration)
        if result and "end" in result and "sum_received" in result["end"]:
            bps = result["end"]["sum_received"]["bits_per_second"]
            mbps = bps / 1_000_000
            results.append(mbps)
            print(f"Measured bandwidth: {mbps:.2f} Mbps")
        else:
            print("Failed to get bandwidth measurement")
        time.sleep(1)
    
    if results:
        avg_mbps = sum(results) / len(results)
        print(f"Average bandwidth: {avg_mbps:.2f} Mbps")
        return results, avg_mbps
    else:
        print("No valid bandwidth measurements")
        return [], 0

def save_results(results, avg, server_ip, output_dir="/qos_data"):
    """Save results to file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{output_dir}/bandwidth_{server_ip.replace('.', '_')}_{timestamp}.txt"
    
    with open(filename, "w") as f:
        f.write(f"Bandwidth measurements to {server_ip}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, mbps in enumerate(results):
            f.write(f"Iteration {i+1}: {mbps:.2f} Mbps\n")
        
        f.write(f"\nAverage bandwidth: {avg:.2f} Mbps\n")
    
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Measure bandwidth between containers')
    parser.add_argument('--server', action='store_true', help='Run as server')
    parser.add_argument('--client', type=str, help='Server IP to connect to as client')
    parser.add_argument('--duration', type=int, default=10, help='Test duration in seconds')
    parser.add_argument('--iterations', type=int, default=3, help='Number of test iterations')
    
    args = parser.parse_args()
    
    if args.server:
        run_iperf_server()
    elif args.client:
        results, avg = measure_bandwidth(args.client, args.duration, args.iterations)
        if results:
            save_results(results, avg, args.client)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
