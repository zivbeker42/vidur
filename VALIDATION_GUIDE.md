# Validation Guide: Vidur Simulator vs. Real Hardware

This guide outlines the process for validating the Vidur simulator against real hardware. The goal is to compare the simulator's predictions (TTFT, TPOT, E2E Latency) against actual execution metrics from an LLM inference engine running on your specific hardware.

## Prerequisites

- **Hardware:** Access to the GPU(s) you wish to profile (e.g., A100, H100, etc.).
- **Software:** 
  - Vidur (this repo) installed.
  - A real inference engine. **[Sarathi-Serve](https://github.com/microsoft/sarathi-serve)** is the reference implementation used by Vidur and is highly recommended for apples-to-apples comparison.

---

## Step 1: Profile Your Hardware ("Calibration")

Vidur acts as a "digital twin" of your hardware. To make it accurate, you must feed it performance profiles from your actual GPU and Network setup.

### 1.1 Compute Profiling (MLP & Attention)
Run these commands on your GPU machine.

**MLP Profiling:**
```bash
# Example for Llama-2-7b-hf on 1 GPU
python vidur/profiling/mlp/main.py \
    --models meta-llama/Llama-2-7b-hf \
    --num_gpus 1 \
    --output_dir my_profiling_data
```

**Attention Profiling:**
```bash
python vidur/profiling/attention/main.py \
    --models meta-llama/Llama-2-7b-hf \
    --num_gpus 1 \
    --output_dir my_profiling_data
```

### 1.2 Network Profiling (Collectives)
Required if you are testing multi-GPU setups (Tensor Parallelism or Pipeline Parallelism).

**All-Reduce (for Tensor Parallelism):**
```bash
python vidur/profiling/collectives/main.py \
    --num_workers_per_node_combinations 1,2,4,8 \
    --collective all_reduce \
    --output_dir my_profiling_data
```

### 1.3 Install Profiles
Move the generated CSV files into the Vidur data directory structure.

```bash
# Create directories for your custom hardware
mkdir -p data/profiling/compute/my_custom_gpu/meta-llama/Llama-2-7b-hf/
mkdir -p data/profiling/network/my_custom_network/

# Copy Compute Data
cp my_profiling_data/mlp/*/meta-llama/Llama-2-7b-hf/mlp.csv data/profiling/compute/my_custom_gpu/meta-llama/Llama-2-7b-hf/
cp my_profiling_data/attention/*/meta-llama/Llama-2-7b-hf/attention.csv data/profiling/compute/my_custom_gpu/meta-llama/Llama-2-7b-hf/

# Copy Network Data (if applicable)
cp my_profiling_data/collectives/*/all_reduce.csv data/profiling/network/my_custom_network/
```

---

## Step 2: Run Real-World Benchmark ("Ground Truth")

You need a baseline to compare against. Use the `splitwise_conv.csv` trace provided in this repo, as it represents a realistic conversational workload.

1. **Install Sarathi-Serve** (or vLLM).
2. **Run Inference:** Execute the engine with the `Llama-2-7b-hf` model and the `data/processed_traces/splitwise_conv.csv` trace.
3. **Capture Metrics:** Record the average **TTFT** (Time to First Token) and **TPOT** (Time per Output Token).

*Note: Ensure the real engine configuration (TP/PP degree, max batch size, scheduler settings) matches what you intend to simulate.*

---

## Step 3: Run Simulation

Run Vidur with your custom profiles and the same trace file.

```bash
python -m vidur.main \
    --replica_config_device my_custom_gpu \
    --replica_config_network_device my_custom_network \
    --replica_config_model_name meta-llama/Llama-2-7b-hf \
    --cluster_config_num_replicas 1 \
    --request_generator_config_type synthetic \
    --length_generator_config_type trace \
    --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
    --interval_generator_config_type trace \
    --trace_request_interval_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv
```

*Note: You may need to adjust `--replica_scheduler_config_batch_size_cap` to match your real-world engine's settings.*

---

## Step 4: Compare Results

Compare the metrics output by Vidur (logged to wandb or stdout) with your real-world logs.

**Key Metrics to Diff:**
- **TTFT (Prefill Latency):** Validates the *Compute (Attention/MLP)* profiles.
- **TPOT (Decode Latency):** Validates *Compute* profiles + *Memory Bandwidth* constraints.
- **E2E Latency:** Validates the *Scheduler* logic and *Queueing* delays.

**Acceptable Variance:**
- < 10% difference is generally considered a high-fidelity simulation.
- Larger discrepancies usually indicate a mismatch in configuration (e.g., scheduler settings, overheads) or that the "CPU Overhead" (scheduling time) on the real hardware is significant (Vidur models this, but it requires separate profiling).
