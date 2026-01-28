
"""
generate_results_for_ppt.py

Generates REAL, PPT-safe evaluation metrics for the
Diagnostic Test Pathway Personalization project.
"""
import matplotlib.pyplot as plt

import numpy as np
import torch
from mab_environment import DiagnosticEnvironment

NUM_EVAL_PATIENTS = 500
CSV_PATH = "heart_diagnostic_10k_balanced_clean.csv"
SCALER_PATH = "sae_scaler.joblib"
AUTOENCODER_PATH = "sae_autoencoder.joblib"
AGENT_PATH = "mab_agent_pytorch.pth"
ENCODING_DIM = 4

TEST_FEATURES = [
    'troponin', 'cholesterol_total', 'HDL', 'LDL', 'BNP',
    'NT_proBNP', 'echo_ef', 'stress_test_result', 'c_reactive_protein'
]
LABELS = ['Healthy', 'CAD', 'HF', 'STR']
TOTAL_TESTS = len(TEST_FEATURES)

class DQN(torch.nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = DiagnosticEnvironment(
    csv_path=CSV_PATH,
    scaler_path=SCALER_PATH,
    autoencoder_path=AUTOENCODER_PATH
)

agent = DQN(env.state_size, env.action_size).to(device)
agent.load_state_dict(torch.load(AGENT_PATH, map_location=device))
agent.eval()

correct = 0
test_counts = []
rewards = []

for _ in range(NUM_EVAL_PATIENTS):
    state = env.reset()
    done = False
    tests_used = 0
    ep_reward = 0

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_vals = agent(state_t).cpu().numpy()[0]

        for i in range(TOTAL_TESTS):
            if state[ENCODING_DIM + i] != -1:
                q_vals[i] = -np.inf

        action = np.argmax(q_vals)
        state, reward, done = env.step(action)

        if action < TOTAL_TESTS:
            tests_used += 1

        ep_reward += reward

        if done:
            pred = LABELS[action - TOTAL_TESTS]
            true = LABELS[env.current_patient_label]
            if pred == true:
                correct += 1

    test_counts.append(tests_used)
    rewards.append(ep_reward)

accuracy = correct / NUM_EVAL_PATIENTS
avg_tests = np.mean(test_counts)
test_reduction = ((TOTAL_TESTS - avg_tests) / TOTAL_TESTS) * 100
avg_reward = np.mean(rewards)

print("\n==== PPT READY RESULTS ====")
print(f"Accuracy              : {accuracy*100:.2f}%")
print(f"Avg Tests per Patient : {avg_tests:.2f}")
print(f"Test Reduction        : {test_reduction:.2f}%")
print(f"Avg Reward            : {avg_reward:.2f}")
print("==========================")

with open("ppt_results_summary.txt", "w") as f:
    f.write(f"Accuracy: {accuracy*100:.2f}%\n")
    f.write(f"Avg Tests: {avg_tests:.2f}\n")
    f.write(f"Test Reduction: {test_reduction:.2f}%\n")
    f.write(f"Avg Reward: {avg_reward:.2f}\n")


# ================== CREATE RESULTS IMAGE FOR PPT ==================

fig, ax = plt.subplots(figsize=(8, 5))
ax.axis('off')

text = (
    "PPT READY RESULTS\n\n"
    f"Diagnostic Accuracy      : {accuracy*100:.2f}%\n"
    f"Avg Tests per Patient    : {avg_tests:.2f}\n"
    f"Test Reduction           : {test_reduction:.2f}%\n"
    f"Average Reward           : {avg_reward:.2f}\n"
)

ax.text(
    0.05, 0.6, text,
    fontsize=16,
    family='monospace',
    verticalalignment='top'
)

plt.tight_layout()
plt.savefig("final_results.png", dpi=200)
plt.close()

print("Saved result image: final_results.png")
