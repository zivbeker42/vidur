# Gemini Memory Context - Vidur

## 1. Project Identity
**Name:** Vidur
**Type:** LLM Inference System Simulator
**Purpose:** A high-fidelity, extensible simulator for studying Large Language Model (LLM) inference system performance. It allows researchers and engineers to:
- Analyze system metrics (TTFT, TPOT, E2E latency) under various workloads.
- Perform capacity planning and configuration search for deployments.
- Test new scheduling algorithms and optimizations (e.g., speculative decoding) without physical GPU hardware.

## 2. Tech Stack
- **Language:** Python 3.10+
- **Core Libraries:**
  - `numpy`, `pandas`: Data manipulation.
  - `scikit-learn`: Machine learning (likely for execution time prediction).
  - `wandb`: Experiment tracking and visualization.
  - `plotly`, `matplotlib`, `seaborn`: Data visualization.
  - `kaleido`: Image export for plots.
- **Environment:** `mamba`, `conda`, or `venv`.
- **Linting/Formatting:** `black`, `isort`, `flake8`.

## 3. Project Structure
- **/vidur**: Root source directory.
  - **main.py**: Entry point for the simulator.
  - **config/**: Configuration dataclasses and strict typing setup.
  - **entities/**: Domain models (`Request`, `Replica`, `Batch`, `Cluster`).
  - **scheduler/**: Scheduling logic (`GlobalScheduler`, `ReplicaScheduler`).
  - **execution_time_predictor/**: ML models (`RandomForest`, `LinearRegression`) to estimate processing times.
  - **metrics/**: Metrics collection (`cdf_sketch`, `metrics_store`).
  - **events/**: Event classes for the discrete-event simulation engine.
  - **profiling/**: Tools for profiling models and gathering ground-truth data.
- **/data**: Contains processed trace files (`processed_traces/`) and hardware profiling data (`profiling/`).
- **/assets**: Documentation images.
- **/docs**: Documentation for metrics and profiling.

## 4. Coding Standards
- **Naming Conventions:**
  - Classes: `PascalCase` (e.g., `BaseGlobalScheduler`, `SimulationConfig`).
  - Functions/Variables: `snake_case` (e.g., `on_batch_schedule`, `num_prefill_tokens`).
  - Constants: `UPPER_CASE` (in `constants.py` files).
- **Architecture:**
  - **Object-Oriented:** Heavy use of classes, inheritance (ABC), and polymorphism.
  - **Discrete Event Simulation:** Logic is driven by event processing and time steps.
  - **Strong Typing:** Extensive use of Python type hints (`List[Tuple[int, Request]]`).
  - **Configuration:** Structured configuration objects are passed down, rather than global state.
  - **Decorators:** Used for validation (e.g., `@check_scheduled`).

## 5. Contextual Rules
- **Formatting:** Code must pass `black` and `isort`. Run `make format` before committing.
- **Logging:** Use the internal `vidur.logger`. Do not use `print` statements for system logs.
- **Simulation Integrity:**
  - Ensure deterministic behavior by respecting random seeds (`set_seeds(config.seed)`).
  - Time is explicit in the simulator; ensure event timestamps are handled monotonically.
- **Data Handling:** Metrics are logged to `wandb` and local JSON/Chrome traces.
- **Testing:** No explicit unit test suite detected in root. Validation is likely performed by running simulation scenarios (e.g., `python -m vidur.main` with various flags).

## 6. Active Work Context
- **Current Focus:** Improvements to the simulator core.
- **Canary Features:** Prefix caching, advanced routing policies, and memory optimization are under active development in the `canary` branch.
- **Task:** General maintenance and feature expansion as requested by the user.
