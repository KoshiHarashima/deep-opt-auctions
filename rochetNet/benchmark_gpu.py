from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import json
import csv
import numpy as np
import torch

from nets import *
from cfgs import *
from data import *
from trainer import *

def benchmark_device(device_name, device, cfg, Net, Generator, Trainer, num_iterations=1000, warmup_iterations=10):
    """Benchmark training iterations on a specific device"""
    print(f"\n{'='*60}")
    print(f"Benchmarking on {device_name}")
    print(f"{'='*60}")
    
    # Create network and move to device
    net = Net(cfg, 'train')
    net.to(device)
    
    # Create generator
    generator = [Generator(cfg, 'train'), Generator(cfg, 'val')]
    
    # Create trainer
    trainer = Trainer(cfg, "train", net)
    trainer.device = device
    trainer.net.to(device)
    
    # Move all parameters to device
    for param in trainer.net.parameters():
        param.data = param.data.to(device)
    
    # Warmup
    print(f"Warming up ({warmup_iterations} iterations)...")
    for _ in range(warmup_iterations):
        X = next(generator[0].gen_func)
        X_tensor = torch.from_numpy(X).to(device).float()
        
        # Forward pass
        trainer.net.eval()
        with torch.no_grad():
            alloc, pay = trainer.net.inference(X_tensor)
    
    # Synchronize if CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark ({num_iterations} iterations)...")
    times = []
    
    for i in range(num_iterations):
        # Get batch
        X = next(generator[0].gen_func)
        X_tensor = torch.from_numpy(X).to(device).float()
        
        # Synchronize before timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        # Forward pass
        trainer.net.eval()
        with torch.no_grad():
            alloc, pay = trainer.net.inference(X_tensor)
        
        # Synchronize after computation
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        if (i + 1) % 100 == 0:
            avg_time = np.mean(times[-100:])
            print(f"  Iteration {i+1}/{num_iterations}: {avg_time*1000:.2f} ms/iter")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\nResults for {device_name}:")
    print(f"  Average time: {avg_time*1000:.2f} ms/iter")
    print(f"  Std dev: {std_time*1000:.2f} ms/iter")
    print(f"  Min time: {min_time*1000:.2f} ms/iter")
    print(f"  Max time: {max_time*1000:.2f} ms/iter")
    print(f"  Total time: {sum(times):.2f} s")
    
    return {
        'device': device_name,
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'total_time_s': sum(times),
        'iterations': num_iterations
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_gpu.py <setting> [num_iterations]")
        print("Example: python benchmark_gpu.py additive_1x2_uniform 1000")
        sys.exit(1)
    
    setting = sys.argv[1]
    num_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    # Select configuration (simplified - add more as needed)
    if setting == "additive_1x2_uniform":
        cfg = additive_1x2_uniform_config.cfg
        Net = additive_net.Net
        Generator = uniform_01_generator.Generator
        Trainer = trainer.Trainer
    else:
        print(f"Setting '{setting}' not supported in benchmark script")
        print("Supported settings: additive_1x2_uniform")
        sys.exit(1)
    
    print(f"Setting: {setting}")
    print(f"Number of iterations: {num_iterations}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA available: Yes")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CUDA available: No")
    
    results = []
    
    # Benchmark CPU
    cpu_device = torch.device('cpu')
    cpu_result = benchmark_device("CPU", cpu_device, cfg, Net, Generator, Trainer, num_iterations)
    results.append(cpu_result)
    
    # Benchmark GPU if available
    if cuda_available:
        gpu_device = torch.device('cuda')
        gpu_result = benchmark_device("GPU", gpu_device, cfg, Net, Generator, Trainer, num_iterations)
        results.append(gpu_result)
        
        # Calculate speedup
        speedup = cpu_result['avg_time_ms'] / gpu_result['avg_time_ms']
        print(f"\n{'='*60}")
        print(f"Speedup: {speedup:.2f}x faster on GPU")
        print(f"{'='*60}")
    
    # Save results
    output_prefix = f"benchmark_{setting}"
    
    # Save as JSON
    json_filename = f"{output_prefix}.json"
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_filename}")
    
    # Save as CSV
    csv_filename = f"{output_prefix}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    main()

