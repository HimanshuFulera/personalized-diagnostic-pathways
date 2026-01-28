# SmartPath: Reinforcement Learning for Adaptive Medical Diagnostic Pathways

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status: Patent Pending](https://img.shields.io/badge/IP-Patent_Disclosure-red.svg)](#)

## 📌 Executive Summary
It is an AI-driven Clinical Decision Support System (CDSS) that personalizes the diagnostic journey for cardiovascular patients. Unlike traditional linear protocols, this system models the diagnostic process as a **Sequential Decision Problem**. 

By utilizing a **Multi-Armed Bandit (MAB)** framework integrated with **Deep Q-Learning (DQN)**, the system dynamically selects the most informative next test for a patient, maximizing diagnostic accuracy while minimizing unnecessary medical procedures and costs.



---

## 🏗️ Technical Architecture

The system operates through a dual-stage machine learning pipeline:

### 1. Patient State Encoding (Sparse Autoencoder)
Initial clinical data (11 features: age, BP, BMI, etc.) is compressed into a 4-dimensional **"Clinical Fingerprint"** using a **Sparse Autoencoder (SAE)**. This removes noise and extracts the latent physiological state of the patient.
* **Model:** MLPRegressor (Neural Network)
* **Reduction:** 11-Dimensional Raw Data → 4-Dimensional Latent Space

### 2. Sequential Decision Engine (DQN)
A Deep Q-Network (DQN) agent navigates the diagnostic pathway. It observes the patient fingerprint and the results of previous tests to decide:
* **Perform a Test:** Choose from 9 diagnostic tests (Troponin, Echo, BNP, etc.).
* **Final Diagnosis:** Commit to a diagnosis (Healthy, CAD, Heart Failure, or Stroke).



---

## 🚀 Key Features
* **Dynamic Test Selection:** The agent learns to prioritize tests based on their "information gain" per specific patient profile.
* **Reward Shaping:** Optimized with a specialized reward function ($+50$ for correct diagnosis, $-1$ per test performed) to ensure efficiency.
* **Real-time Interface:** A Flask-based web dashboard for clinicians to simulate and monitor diagnostic pathways.
* **Explainability:** Includes t-SNE visualization scripts to audit how the model clusters clinical states in the latent space.

---

## 📂 Project Structure
* `app.py`: Flask web application for real-time diagnostic simulation.
* `build_autoencoder.py`: Script to train the SAE and generate patient fingerprints.
* `mab_environment.py`: Custom Gymnasium-style environment defining the clinical "rules."
* `run_diagnostic_final.py`: Terminal-based interface for end-to-end inference.
* `generate_results_for_ppt.py`: Benchmarking script for accuracy and test-reduction metrics.
* `generate_tsne_plot.py`: Visualization tool for latent space analysis.

---

## 📊 Performance Metrics
* **Accuracy:** >95% success rate in matching ground-truth clinical outcomes.
* **Efficiency:** 30–40% reduction in diagnostic tests compared to standard exhaustive panels.
* **Clustering:** t-SNE analysis confirms distinct clinical separation within the 4D latent space.



---

## 🛠️ Installation & Usage

### 1. Environment Setup
```bash
python -m venv my_ml_env
source my_ml_env/bin/activate  # Windows: .\my_ml_env\Scripts\activate
pip install torch flask scikit-learn pandas joblib matplotlib seaborn
