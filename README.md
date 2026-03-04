# EcoTable: Cost-effective Table Integration in Data Lakes

This repository provides the official implementation of the paper: **"EcoTable: Cost-effective Table Integration in Data Lakes for Natural Language Queries"**. 

EcoTable is a natural language-driven framework designed to bridge the gap between raw data lake files (e.g., CSV/Parquet) and analytical requirements by discovering query-specific schemas and performing automated data transformations.

## 🚀 Key Features

* **Query-Driven Integration**: Tailors database schemas to specific natural language queries rather than relying on a static, pre-defined ETL process.
* **Hybrid Architecture**: Combines lightweight deep learning models (e.g., DeBERTa-v3) for efficiency with LLMs for high-precision semantic reasoning.
* **Cost-Aware Optimization**: Minimizes LLM invocation costs through Steiner tree-based join path identification and global result caching.
* **Scalable Pruning**: Features a t-spanner based edge-pruning strategy to handle massive data lakes containing thousands of tables.
* **Parallel Transformation**: Optimizes execution latency via an edge-coloring-based parallelization strategy for independent table joins.

---

## 📁 Repository Structure

### 1. Source Code (`src/`)
* **Table Identification**: Two-stage schema linking involving PLM-based coarse filtering and LLM-based fine-grained verification.
* **Graph-based Validation**: Discovery of optimal join paths using Steiner Tree search and iterative LLM validation on a weighted graph.
* **Table Transformation**: ReAct-style code generation and execution.

### 2. Datasets
The repository includes five core benchmarks featuring real-world industrial data and a large-scale noisy data lake:
* **`ad/`**: Advertisement spend and click-through tracking datasets.
* **`business/`**: Financial metrics and sales operations data.
* **`engagement/`**: User interaction and retention records.
* **`platform/`**: Engineering activity and system metadata.
* **`NYC/`**: **NYC Data Lake**, a massive benchmark with 1,214 tables and 800 queries designed for scalability and robustness testing.

---

## 🛠️ Requirements

* **Environment**: Python 3.8+
* **Backends**: 
    * PLMs: 
    * LLMs: 
* **Required Libraries**: 
