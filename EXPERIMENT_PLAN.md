# Experiment Plan: Vidur Ground Truth Calibration

**Objective:** Generate hardware baselines for **[MODEL: meta-llama/Meta-Llama-3-70B]** on **[GPU CONFIG: 4x NVIDIA A100-80GB (PCIe/NVLink)]**.

## 1. Vidur-Specific Configuration

To ensure the physical setup mirrors the simulator's theoretical constraints:

*   **Parallelism Strategy:**
    *   **Tensor Parallelism (TP):** 4
        *   *Reasoning:* Fits the 70B model (approx. 130GB in FP16) comfortably across 4x80GB cards while maximizing memory bandwidth utilization.
    *   **Pipeline Parallelism (PP):** 1
        *   *Reasoning:* Minimizes "bubble" overhead. Only increase PP if the model does not fit in VRAM with TP=4.
*   **Attention Backend:**
    *   **Kernel:** `FlashAttention-2` (Required for modern throughput).
    *   **Memory Manager:** `PagedAttention` (Block size: 16).
        *   *Vidur Config:* Ensure `block_size` in simulation matches the engine (usually 16 or 32).
*   **KV Cache Limit:**
    *   **Physical Setting:** Set `gpu_memory_utilization=0.90` (Sarathi-Serve/vLLM default).
    *   **Vidur Parameter:** Calculate `total_memory - model_weights - activation_buffer`.
        *   *Est:* 320GB Total - 140GB Weights - 4GB Act = ~176GB KV Cache.

## 2. Workload Characterization (Vidur Inputs)

We will run three distinct phases to characterize performance:

### Phase A: Synthetic Probes (Fixed Lengths)
Isolate Compute vs. Memory bound phases.
*   **Prompt/Decode Pairs:**
    1.  `[512, 128]`: Short context, latency sensitive.
    2.  `[2048, 512]`: Balanced, typical RAG workload.
    3.  `[4096, 128]`: Prefill-heavy (Compute bound initiation).

### Phase B: Saturation Sweep (Poisson Arrival)
Find the system's "knee" curve.
*   **Arrival Process:** Poisson.
*   **RPS Sweep:** `[0.5, 1.0, 2.0, 4.0, 8.0]`.
*   **Goal:** Identify the RPS where P99 latency degrades by >50%.

### Phase C: Realistic Trace
*   **Dataset:** `data/processed_traces/splitwise_conv.csv` (from Vidur repo).
    *   *Why:* Contains realistic variance in prompt/decode lengths essential for testing scheduler fragmentation.

## 3. Execution & Metrics (Ground Truth)

**Script Strategy:** Use a Python client interacting with the inference engine's API.

**Metrics to Capture:**

| Metric | Definition | Vidur Equivalent |
| :--- | :--- | :--- |
| **Pre-fill Latency (TTFT)** | `t_first_token - t_request_sent` | `prefill_e2e_time` |
| **Per-token Decode Latency (TPOT)** | `(t_end - t_first_token) / (output_tokens - 1)` | `decode_time_execution_plus_preemption_normalized` |
| **Request-level Latency (E2E)** | `t_end - t_request_sent` | `request_e2e_time` |
| **Scheduler Overhead** | *Derived:* `TTFT_measured - TTFT_ideal_compute` | `scheduling_delay` |

## 4. Calibration Loop

### Process
1.  **Hardware Run:** Execute Phase B (RPS Sweep) on physical hardware. Save logs.
2.  **Profile Import:** Ensure `vidur/profiling/` contains fresh profiles for this specific GPU/Model combo (as per `VALIDATION_GUIDE.md`).
3.  **Simulation Run:** Execute `python -m vidur.main` using the same Trace and RPS.
4.  **Diff Analysis:** Compare P50 and P99 for TTFT and TPOT.

### Tolerance Thresholds
*   **Strict (Compute/Bandwidth):** TTFT & TPOT P50 < **5% MAPE**.
    *   *If failed:* Re-run `vidur/profiling/mlp` and `vidur/profiling/attention`.
*   **Loose (Scheduler/Queue):** E2E Latency P99 < **10-15% MAPE**.
    *   *If failed:* Adjust `replica_scheduler_config_batch_size_cap` or CPU overhead parameters.

## 5. Output Template (JSON)

Save hardware experiment results in this format for easy parsing by calibration scripts:

```json
{
  "experiment_id": "llama3-70b-4xa100-run1",
  "config": {
    "model": "meta-llama/Meta-Llama-3-70B",
    "tp": 4,
    "pp": 1,
    "gpu": "A100-80GB"
  },
  "workload": {
    "type": "poisson_trace",
    "rps": 2.0,
    "dataset": "splitwise_conv"
  },
  "metrics": {
    "ttft_ms": {
      "p50": 120.5,
      "p99": 350.2,
      "mean": 135.0
    },
    "tpot_ms": {
      "p50": 25.4,
      "p99": 45.1,
      "mean": 27.8
    },
    "e2e_latency_s": {
      "p50": 3.2,
      "p99": 6.8
    },
    "throughput_token_s": 2450.5
  }
}
```
