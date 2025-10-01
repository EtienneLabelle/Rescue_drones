#!/usr/bin/env python3
"""
Simple script to start TensorBoard for viewing training logs.
"""

import subprocess
import sys
import os

def start_tensorboard():
    """Start TensorBoard to view training logs."""
    log_dir = "runs"
    
    if not os.path.exists(log_dir):
        print(f"Error: Log directory '{log_dir}' not found.")
        print("Please run the training first to generate logs.")
        return
    
    print(f"Starting TensorBoard with log directory: {log_dir}")
    print("TensorBoard will be available at: http://localhost:6006")
    print("Press Ctrl+C to stop TensorBoard")
    
    try:
        subprocess.run([sys.executable, "-m", "tensorboard.main", "--logdir", log_dir, "--port", "6006"])
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except FileNotFoundError:
        print("Error: TensorBoard not found. Please install it with:")
        print("pip install tensorboard")

if __name__ == "__main__":
    start_tensorboard() 