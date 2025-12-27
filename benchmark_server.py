import sys
import subprocess
import os

# Constants matched from the notebook
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GPU_MEMORY_UTILIZATION = 0.9
PORT = 8000

def start_server():
    cmd = [
        sys.executable,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--dtype", "half",
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
        "--port", str(PORT),
        "--host", "0.0.0.0" 
    ]

    print(f"Starting vLLM server with command:\n{' '.join(cmd)}")
    sys.stdout.flush()

    # Replace the current process with the vllm process so it receives signals directly
    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    start_server()

