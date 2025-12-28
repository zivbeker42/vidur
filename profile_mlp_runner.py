import sys
import os
import argparse
from vidur.profiling.mlp.main import main as mlp_main

def run_profiling():
    # Set default arguments for the profiling script
    # These effectively mimic passing command line args
    
    # Check if arguments were passed, otherwise set defaults
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--num_gpus", "1",
            "--output_dir", "profiling_outputs",
            "--models", 
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--num_tensor_parallel_workers", "1",
            "--profile_method", "record_function" # reliable timing method
        ])
    
    print(f"Running MLP profiling with args: {sys.argv}")
    
    try:
        mlp_main()
    except Exception as e:
        print(f"Profiling failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure the current directory is in python path
    sys.path.append(os.getcwd())
    run_profiling()
