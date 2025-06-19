#!/usr/bin/env python3
import subprocess
import re
import time
import argparse
import os
import statistics

def ping_host(host, count=10):
    """Ping a host and return results"""
    print(f"Measuring latency to {host}...")
    result = subprocess.run(
        ["ping", "-c", str(count), host],
        stdout=subprocess.PIPE,
        text=True
    )
    
    # Parse ping output
    latencies = []
    for line in result.stdout.splitlines():
        match = re.search(r"time=(\d+\.\d+) ms", line)
        if match:
            latencies.append(float(match.group(1)))
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        try:
            jitter = statistics.stdev(latencies)
        except:
            jitter = 0
        
        print(f"Min latency: {min_latency:.2f} ms")
        print(f"Avg latency: {avg_latency:.2f} ms")
        print(f"Max latency: {max_latency:.2f} ms")
        print(f"Jitter: {jitter:.2f} ms")
        
        return latencies, avg_latency, min_latency, max_latency, jitter
    else:
        print("No latency measurements found")
        return [], 0, 0, 0, 0

def save_results(latencies, avg, min_val, max_val, jitter, host, output_dir="/qos_data"):
    """Save results to file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{output_dir}/latency_{host.replace('.', '_')}_{timestamp}.txt"
    
    with open(filename, "w") as f:
        f.write(f"Latency measurements to {host}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, ms in enumerate(latencies):
            f.write(f"Ping {i+1}: {ms:.2f} ms\n")
        
        f.write(f"\nMin latency: {min_val:.2f} ms\n")
        f.write(f"Avg latency: {avg:.2f} ms\n")
        f.write(f"Max latency: {max_val:.2f} ms\n")
        f.write(f"Jitter: {jitter:.2f} ms\n")
    
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Measure latency between containers')
    parser.add_argument('--host', type=str, required=True, help='Host to ping')
    parser.add_argument('--count', type=int, default=10, help='Number of pings')
    
    args = parser.parse_args()
    
    latencies, avg, min_val, max_val, jitter = ping_host(args.host, args.count)
    if latencies:
        save_results(latencies, avg, min_val, max_val, jitter, args.host)

if __name__ == "__main__":
    main()
