import sys
import os
import argparse
from vidur.profiling.attention.main import main as attn_main

def run_profiling():
    # Set default arguments for the profiling script
    
    if len(sys.argv) == 1:
        # Attention profiling often requires specific backends. 
        # FLASHINFER is default in main.py, but let's be explicit if needed.
        sys.argv.extend([
            "--num_gpus", "1",
            "--output_dir", "profiling_outputs",
            "--models",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--num_tensor_parallel_workers", "1",
            # Standard attention input configs
            "--max_seq_len", "4096",
            "--min_batch_size", "1",
            "--max_batch_size", "128"
        ])

    print(f"Running Attention profiling with args: {sys.argv}")
    
    try:
        attn_main()
    except Exception as e:
        print(f"Profiling failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    run_profiling()
