# IITF-Project: Dual-Trace Telemetry Forecasting

- **Author:** Furkan Sanlav
- **Institution:** Hacettepe University, AI Engineering
- **Project Phase:** 1 (Baseline Development)

## 📌 Project Overview
The **Industrial IoT Telemetry Forecasting (IITF)** project addresses resource demand prediction using the **Bitbrains GWA-T-12** dataset. The system utilizes a specialized **Dual-Trace LSTM** architecture to model two distinct telemetry distributions: stable workloads (`fastStorage`) and volatile, bursty workloads (`rnd`).

## 🛰️ Future Roadmap
*   **Phase 2: Attention Mechanism Integration**
    Implementing **Cross-Attention Transformer layers** to specifically target the high RMSE and non-linear spikes identified in the `rnd` traces during Phase 1.
*   **Phase 3: System Deployment**
    Development of a real-time forecasting dashboard and deployment-ready inference API for cloud resource monitoring.

## 🚀 Technical Architecture
*   **Dual-Trace Modeling:** An LSTM network designed with Trace ID embeddings to differentiate between varied telemetry source behaviors.
*   **Data Pipeline:** Implementation of sliding window sequences with configurable strides to manage temporal dependencies.
*   **Objective Function:** Utilization of **Huber Loss** to provide robustness against outliers and spikes inherent in industrial telemetry.
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
├── tests/               # Automated verification suite (97+ tests)
│   ├── test_data_loader.py
│   └── test_model.py
├── checkpoints/         # Serialized model weights and config logs
├── docs/                # Technical blueprints and design documents
├── main.py              # End-to-end execution entry point
└── pyproject.toml       # Environment and dependency definitions
```

## 🛠️ Installation and Execution
This project utilizes `uv` for dependency management.

```bash
# Clone the repository
git clone [https://github.com/FurkanSanlav/IITF_Project.git](https://github.com/FurkanSanlav/IITF_Project.git)
cd IITF_Project

# Synchronize environment
uv sync

# Execute training and evaluation
uv run main.py --config checkpoints/config.json
```
