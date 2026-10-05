# IITF-Project: Dual-Trace Telemetry Forecasting

- **Author:** Furkan Sanlav
- **Institution:** Hacettepe University, AI Engineering
- **Project Phase:** 1 (Baseline Development)

## 📌 Project Overview
**IITF (Infrastructure Intelligence & Telemetry Forecasting)** forecasts cloud VM resource demand using the **Bitbrains GWA-T-12** dataset: 1,750 VMs (1,250 `fastStorage` + 500 `rnd`) sampled every 5 minutes. A **Dual-Trace LSTM** built in **PyTorch** models two telemetry distributions in a single network: stable workloads (`fastStorage`) and volatile, bursty workloads (`rnd`).

## 🛰️ Future Roadmap
*   **Phase 2: Attention Mechanism Integration**
    Implementing **Cross-Attention Transformer layers** to specifically target the high RMSE and non-linear spikes identified in the `rnd` traces during Phase 1.
*   **Phase 3: System Deployment**
    Development of a real-time forecasting dashboard and deployment-ready inference API for cloud resource monitoring.

## 🚀 Technical Architecture
*   **Dual-Trace Modeling:** An LSTM network designed with Trace ID embeddings to differentiate between varied telemetry source behaviors.
*   **Data Pipeline:** Implementation of sliding window sequences with configurable strides to manage temporal dependencies.
*   **Objective Function:** Utilization of **Huber Loss** to provide robustness against outliers and spikes inherent in VM telemetry.
*   **Optimization:** Integration of `ReduceLROnPlateau` for learning rate adjustment and gradient norm clipping to ensure training stability.

## 📊 Performance Benchmarks
The following table summarizes the performance gains achieved through configuration adjustments and local filesystem utilization within the **WSL** environment.

| Metric | Initial Configuration | Optimized Configuration |
| :--- | :--- | :--- |
| **Mean Epoch Duration** | 5,563 seconds | **270 seconds** |
| **Window Stride** | 1 | 10 |
| **Batch Size** | 64 | 256 |
| **Computational Throughput** | ~9.8 batches/s | **~50.2 batches/s** |

*System Specifications: i5-14500HX, 16GB RAM, NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)*.

## 📈 Phase 1 Evaluation Results
Evaluation conducted on 358,000+ unseen test samples.

| Trace Type | Mean Absolute Error (MAE) | Root Mean Square Error (RMSE) |
| :--- | :--- | :--- |
| **fastStorage (Stable)** | 86.03 | 2871.40 |
| **rnd (Bursty)** | 72.86 | 4557.76 |
| **Global Average** | **78.85** | **3882.62** |

The results indicate that while the model achieves low average error (MAE), the higher RMSE in `rnd` traces highlights the impact of sudden telemetry spikes.

## 📂 Repository Structure
The project is organized to separate core logic from experimental configurations and automated verification.

```text
IITF_Project/
├── src/                 # Core implementation logic
│   ├── config.py        # Centralized configuration management
│   ├── data_loader.py   # Telemetry preprocessing and batching
│   ├── model.py         # DualTraceLSTM architecture definition
│   ├── trainer.py       # Training loop and checkpoint logic
│   └── evaluate.py      # Multi-trace performance metrics
├── tests/               # Automated verification suite (98 pytest tests)
│   ├── test_data_loader.py
│   └── test_model.py
├── checkpoints/         # Model weights and config (created at runtime, git-ignored)
├── docs/                # Technical blueprints and design documents
├── main.py              # End-to-end execution entry point
└── pyproject.toml       # Environment and dependency definitions
```

## 🛠️ Installation and Execution
This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management (Python 3.12).

```bash
# Clone the repository
git clone https://github.com/FurkanSanlav/IITF_Project.git
cd IITF_Project

# Install dependencies
uv sync
```

### Dataset setup
The dataset is not included in the repository. Download the **GWA-T-12 Bitbrains** trace from the [Grid Workloads Archive](http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains) and place the per-VM CSV files as follows:

```text
data/raw/fastStorage/   ← 1,250 VM CSVs
data/raw/rnd/           ← 500 VM CSVs
```

### Run
```bash
# Train and evaluate with default hyperparameters
uv run main.py

# Re-run with a saved configuration (written to checkpoints/ after a run)
uv run main.py --config checkpoints/config.json

# Run the test suite
uv run pytest
```

## 📄 License
Released under the [MIT License](LICENSE).
