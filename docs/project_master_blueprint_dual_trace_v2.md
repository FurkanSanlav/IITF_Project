# PROJECT MASTER BLUEPRINT: Infrastructure Intelligence & Telemetry Forecasting (IITF) - Dual-Trace Version

## 1. THE MISSION (EXECUTIVE SUMMARY)
**Architect Role:** Gemini 3.1 (Planner/Strategic Lead)
**Agent Role:** Claude (Antigravity Code Executor)
**Environment:** WSL (Windows Subsystem for Linux)
**Dependency Management:** `uv` (Fast Python package installer and resolver)
**Objective:** Design a **Hybrid-Aware Predictive System** that handles multi-source telemetry. The system must distinguish between stable `fastStorage` and bursty `rnd` environments to provide accurate, context-aware threshold predictions.

## 2. DATA ASSETS: THE BITBRAINS DUAL-STREAM
Gemini 3.1 must manage the following data context when guiding Claude:
* **Trace A (fastStorage):** 1,250 VMs representing high-performance, consistent workloads.
* **Trace B (rnd):** 500 VMs representing unpredictable, high-variance "Random" workloads.
* **Mandatory Feature:** Include a **"Source Origin" Metadata Tag** (Embedding) so the model knows which environment the data is coming from.

## 3. ARCHITECTURAL STRATEGY: DUAL-MODAL LEARNING
Gemini 3.1 should guide Claude through these tiers:
* **Phase 1 (Hybrid Baseline):** A **Multi-Input LSTM** that takes both raw telemetry and a "Trace Identifier" embedding.
* **Phase 2 (Cross-Attention):** A Transformer-based architecture that uses **Cross-Attention** to learn how patterns in `fastStorage` might inform behavior in `rnd`.
* **Phase 3 (Domain Adaptation):** Implement a **Domain-Adversarial Neural Network (DANN)** approach to ensure the model remains robust regardless of which trace it is currently processing.

## 4. MATHEMATICAL & EVALUATION CONSTRAINTS
Gemini 3.1 must enforce these specific implementations:
* **Loss Function:** **Huber Loss** to minimize the impact of the extreme spikes found in the `rnd` trace while maintaining precision on the `fastStorage` baseline:
    $$L_{\delta}(a) = \begin{cases} \frac{1}{2} a^2 & \text{for } |a| \le \delta, \\ \delta (|a| - rac{1}{2} \delta), & \text{otherwise} \end{cases}$$
* **Evaluation Breakdown:** The system must report **individualized metrics** (RMSE, Threshold-F1) for both `fastStorage` and `rnd` separately, alongside a global score.

## 5. "CLAUDE-COMMAND" PROTOCOLS (UPDATED)
Gemini 3.1, enforce these dual-handling and tooling protocols:
1.  **Tooling Awareness:** Gemini MUST tell Claude that the project uses **`uv`**. Claude should provide commands like `uv add [package]` or `uv run [script]` rather than standard pip or conda commands.
2.  **WSL Pathing:** Claude must ensure file pathing logic is compatible with WSL (Linux-style paths) and optimized for local I/O.
3.  **Context-Aware Data Loaders:** Claude must write a loader that balances batches from both traces to prevent the model from biasing toward the larger `fastStorage` set.
4.  **Modular Feature Scaling:** Claude must implement **individual scaling parameters** for each trace, as their value ranges (min/max) differ significantly.
5.  **Visualization:** The dashboard must toggle between the two environments, showing how the prediction confidence changes based on the source.

## 6. IMMEDIATE WORKFLOW FOR GEMINI 3.1
1.  **Step 1:** Draft a **`pyproject.toml` configuration** and the necessary `uv add` commands to initialize the environment.
2.  **Step 2:** Draft a **Data Merging Strategy** for Claude that combines `fastStorage` and `rnd` while preserving their unique origin tags.
3.  **Step 3:** Design a **Gated Recurrent Unit (GRU) or LSTM architecture** with an auxiliary input for the "Trace ID".
4.  **Step 4:** Define a **Multi-Horizon Validation** setup to test the model's ability to predict 15, 30, and 60 minutes ahead across both trace types.
