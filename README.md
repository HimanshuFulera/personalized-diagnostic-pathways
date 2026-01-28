# SmartPath: Reinforcement Learning for Adaptive Medical Diagnostic Pathways

[![Status: Patent Pending](https://img.shields.io/badge/Intellectual_Property-Patent_Disclosure-red.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)

## 🩺 Project Overview
**SmartPath** is a clinical decision-support system that optimizes the diagnostic journey for cardiovascular patients. Traditional medical pathways are often static; SmartPath uses **Reinforcement Learning** to treat diagnosis as a dynamic "game" where the goal is to reach a correct diagnosis using the fewest, most effective tests possible.

### The Innovation
* **Sparse Autoencoder (SAE):** Compresses 11 patient vitals into a 4D "Clinical Fingerprint."
* **Deep Q-Network (DQN):** An RL agent that selects the "Next Best Action" (a medical test or a final diagnosis) based on the patient's state.
* **Efficiency Logic:** The system is penalized for every test ordered (-1) and rewarded for accuracy (+50), learning to prioritize high-impact diagnostic tests.



---

## 🏗️ The Training & Execution Pipeline

To fully replicate this project, the files must be executed in the following order:

### 1. Patient State Compression (`build_autoencoder.py`)
This script trains the "Patient Analyzer." It takes raw CSV data and learns the latent representation of patient health.
* **Input:** `heart_diagnostic_10k_balanced_clean.csv`
* **Run:** `python build_autoencoder.py`
* **Output:** `sae_autoencoder.joblib`, `sae_scaler.joblib`

### 2. Environment Setup (`mab_environment.py`)
This is the "Rulebook" for the training process. It defines the rewards, penalties, and medical logic.
* **Note:** This file is imported by the trainer, but can be run standalone to verify environment logic.
* **Run:** `python mab_environment.py`

### 3. Agent Training (`train_mab_agent_pytorch.py`)
This script trains the DQN agent (the brain) through thousands of simulated diagnostic trials.
* **Run:** `python train_mab_agent_pytorch.py`
* **Output:** `mab_agent_pytorch.pth` (The trained weights for the model).



### 4. Deployment & Live Interface (`app.py`)
Launch the Flask web dashboard to interact with the trained model in real-time.
* **Run:** `python app.py`
* **Access:** Open `http://127.0.0.1:5000` in your browser.

---

## 📊 Technical Features
* **State Space:** 13-Dimensional (4 Fingerprint features + 9 Test result slots).
* **Reward Function:** Designed to balance diagnostic speed vs. accuracy.
* **Visualization:** `generate_tsne_plot.py` creates t-SNE visualizations to prove the SAE successfully clusters different diseases.
* **Reporting:** `generate_results_for_ppt.py` provides accuracy and test-reduction metrics.



---

## 🛠️ Installation
```bash
# Setup Environment
python -m venv my_ml_env
source my_ml_env/bin/activate # Windows: .\my_ml_env\Scripts\activate

# Install Dependencies
pip install torch flask scikit-learn pandas joblib matplotlib seaborn
