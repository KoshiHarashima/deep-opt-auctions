#!/usr/bin/env python
"""
Batch script to train and test all gamma settings from 0.1 to 2.0 (0.1 increments)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import subprocess
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Generate all gamma setting names
k_values = [round(0.1 + i*0.1, 1) for i in range(20)]
settings = [f"additive_1x2_gamma_{k:.1f}".replace('.', '_') for k in k_values]

def run_command(cmd, setting_name):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"Running: {cmd}")
    print(f"Setting: {setting_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        elapsed_time = time.time() - start_time
        print(f"\n✓ Completed: {setting_name} (Time: {elapsed_time:.2f} seconds)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ Failed: {setting_name} (Time: {elapsed_time:.2f} seconds)")
        print(f"Error: {e}")
        return False

def main():
    """Main function to run all settings"""
    print(f"\n{'='*80}")
    print(f"Batch Training and Testing for Gamma Settings")
    print(f"Total settings: {len(settings)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Change to rochetNet directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    results = {
        'total': len(settings),
        'train_success': 0,
        'train_failed': 0,
        'test_success': 0,
        'test_failed': 0,
        'failed_settings': []
    }
    
    overall_start_time = time.time()
    
    for i, setting in enumerate(settings, 1):
        print(f"\n{'#'*80}")
        print(f"Processing {i}/{len(settings)}: {setting}")
        print(f"{'#'*80}\n")
        
        # Run training
        train_cmd = f"python run_train.py {setting}"
        train_success = run_command(train_cmd, f"{setting} (train)")
        
        if train_success:
            results['train_success'] += 1
        else:
            results['train_failed'] += 1
            results['failed_settings'].append(f"{setting} (train)")
            # Continue to next setting even if train fails
            continue
        
        # Run testing
        test_cmd = f"python run_test.py {setting}"
        test_success = run_command(test_cmd, f"{setting} (test)")
        
        if test_success:
            results['test_success'] += 1
        else:
            results['test_failed'] += 1
            results['failed_settings'].append(f"{setting} (test)")
    
    overall_elapsed_time = time.time() - overall_start_time
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total settings: {results['total']}")
    print(f"Training - Success: {results['train_success']}, Failed: {results['train_failed']}")
    print(f"Testing  - Success: {results['test_success']}, Failed: {results['test_failed']}")
    print(f"Total time: {overall_elapsed_time/3600:.2f} hours ({overall_elapsed_time/60:.2f} minutes)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if results['failed_settings']:
        print(f"\nFailed settings:")
        for failed in results['failed_settings']:
            print(f"  - {failed}")
    else:
        print(f"\n✓ All settings completed successfully!")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

