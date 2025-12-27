import time
import requests
import json
import numpy as np
import pandas as pd
import threading
import sys
import os
import random
import string

# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PORT = 8000
# Allow overriding the base URL via environment variable for flexibility
BASE_URL = os.getenv("VLLM_BASE_URL", f"http://localhost:{PORT}/v1")

print(f"Client configured to connect to: {BASE_URL}")

def wait_for_server(url: str, timeout: int = 600):
    start = time.time()
    print("Waiting for server to be ready...")
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/models")
            if response.status_code == 200:
                print("\nServer is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"Error checking server: {e}")
            
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(5)
        
    print("\nServer failed to start within timeout.")
    return False

def run_request(prompt_len: int, decode_len: int):
    # Generate random string to defeat cache
    # 1 token ~= 4 chars approx
    chars_len = prompt_len * 4
    prompt_text = "".join(random.choices(string.ascii_letters + string.digits, k=chars_len))

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_text,
        "max_tokens": decode_len,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True
    }

    start_time = time.time()
    first_token_time = None
    end_time = None
    token_count = 0

    try:
        response = requests.post(f"{BASE_URL}/completions", json=payload, stream=True)
        response.raise_for_status()

        for chunk in response.iter_lines():
            if chunk:
                if chunk.startswith(b"data: [DONE]"):
                    break

                # Timestamp immediately on receipt
                now = time.time()
                if first_token_time is None:
                    first_token_time = now

                token_count += 1

        end_time = time.time()

        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0 # ms
        total_latency = (end_time - start_time) # seconds
        # TPOT = (Total - TTFT) / (N-1)
        decode_time = (end_time - first_token_time) if first_token_time else 0
        tpot = (decode_time * 1000) / (token_count - 1) if (token_count and token_count > 1) else 0 # ms

        return {
            "prompt_len": prompt_len,
            "decode_len": decode_len,
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "e2e_latency_s": total_latency,
            "output_tokens": token_count,
            "status": "success"
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

def run_poisson_experiment(rps: float, duration_s: int = 30):
    print(f"\nStarting Poisson Load Test: {rps} RPS for {duration_s}s")

    # Generate arrival times
    num_requests = int(rps * duration_s)
    if num_requests == 0:
        return pd.DataFrame()
        
    intervals = np.random.exponential(1.0/rps, num_requests)
    arrival_times = np.cumsum(intervals)

    results = []
    threads = []

    def worker(delay):
        time.sleep(delay)
        # Use a fixed workload for stability: 512 prefill, 128 decode
        res = run_request(512, 128)
        results.append(res)

    for delay in arrival_times:
        t = threading.Thread(target=worker, args=(delay,))
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    return pd.DataFrame([r for r in results if r['status'] == 'success'])

def main():
    if not wait_for_server(BASE_URL):
        sys.exit(1)

    # --- Phase A: Synthetic Probes ---
    probes = [
        (128, 128),   # Short/Short
        (512, 128),   # Medium/Short
        (1024, 512),  # Long/Medium
        (2048, 128)   # Heavy Prefill
    ]

    print("\nRunning Phase A: Synthetic Probes...")
    phase_a_results = []

    for p_len, d_len in probes:
        print(f"  Probing [{p_len}, {d_len}]...")
        # Warmup
        run_request(p_len, 10)

        # Measurement (Avg of 3)
        for _ in range(3):
            res = run_request(p_len, d_len)
            if res['status'] == 'success':
                phase_a_results.append(res)

    phase_a_df = pd.DataFrame(phase_a_results)
    if not phase_a_df.empty:
        print("\nPhase A Results Summary:")
        print(phase_a_df.groupby(['prompt_len', 'decode_len'])[['ttft_ms', 'tpot_ms']].mean())

    # --- Phase B: Saturation Sweep (Poisson) ---
    # Adjusted for T4 capability (from notebook)
    rps_levels = [0.5, 1.0, 2.0, 4.0] 
    phase_b_metrics = {}

    for rps in rps_levels:
        df = run_poisson_experiment(rps)
        if not df.empty:
            phase_b_metrics[rps] = {
                "ttft_p50": df['ttft_ms'].median(),
                "ttft_p99": df['ttft_ms'].quantile(0.99),
                "tpot_p50": df['tpot_ms'].median(),
                "e2e_p99": df['e2e_latency_s'].quantile(0.99)
            }
            print(f"  -> P99 E2E Latency: {phase_b_metrics[rps]['e2e_p99']:.2f}s")
        else:
            print(f"  -> No successful requests for {rps} RPS")

    # --- Export Results ---
    output_data = {
        "experiment_id": "tinyllama-1b-t4-local-run1",
        "config": {
            "model": MODEL_NAME,
            "tp": 1,
            "pp": 1,
            "gpu": "Local-GPU"
        },
        "workload": {
            "type": "synthetic_poisson_mixed",
            "rps_levels": rps_levels,
            "phase_a_probes": probes
        },
        "metrics": {
            "phase_a_raw": phase_a_results,
            "phase_b_summary": phase_b_metrics
        }
    }

    output_file = "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nBenchmark complete! Results saved to '{output_file}'.")

if __name__ == "__main__":
    main()
