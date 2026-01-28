#MAB- SMART DOCTOR

import numpy as np
import joblib
import random
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.exceptions import ConvergenceWarning
from collections import deque
from mab_environment import DiagnosticEnvironment

# --- 0. Suppress Scikit-learn Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

print("--- MAB Agent Trainer (PyTorch) ---")

# --- 1. Set Device (use GPU if available) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Define the DQN "Brain" Architecture ---
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# --- 3. Experience Replay Memory ---
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- 4. Initialize Environment ---
print("Initializing environment...")
env = DiagnosticEnvironment(
    csv_path='heart_diagnostic_10k_balanced_clean.csv',
    scaler_path='sae_scaler.joblib',
    autoencoder_path='sae_autoencoder.joblib'
)

# --- 5. Hyperparameters ---
NUM_EPISODES = 8000
TARGET_UPDATE_FREQ = 20  # How often to update the "Teacher" brain
BATCH_SIZE = 128
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995 # Slower decay for more learning
LEARNING_RATE = 1e-4
MEMORY_SIZE = 10000

n_actions = env.action_size
n_observations = env.state_size

# --- 6. Build Networks: "Student" and "Teacher" ---
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict()) # Clone the "Student"
target_net.eval() # "Teacher" is in evaluation mode

optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
memory = ReplayMemory(MEMORY_SIZE)

# --- 7. Optimize Model (The Learning Step) ---
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return # Not enough memories to learn yet
        
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
    
    batch_state = torch.FloatTensor(np.array(batch_state)).to(device)
    batch_action = torch.LongTensor(batch_action).unsqueeze(1).to(device)
    batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
    batch_next_state = torch.FloatTensor(np.array(batch_next_state)).to(device)
    batch_done = torch.BoolTensor(batch_done).unsqueeze(1).to(device)
    
    # --- This is the core of Q-Learning ---
    
    # 1. Get Q-values from the "Student" (policy_net)
    q_values = policy_net(batch_state).gather(1, batch_action)
    
    # 2. Get *next* Q-values from the "Teacher" (target_net)
    with torch.no_grad():
        next_q_values = target_net(batch_next_state).max(1)[0].unsqueeze(1)
        
    # 3. Calculate the "target" (what the Student *should* have predicted)
    target_q_values = batch_reward + (GAMMA * next_q_values * ~batch_done)
    
    # 4. Calculate the error (Loss)
    criterion = nn.SmoothL1Loss()
    loss = criterion(q_values, target_q_values)
    
    # 5. Update the "Student" brain
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) # Prevents explosions
    optimizer.step()

# --- 8. The Main Training Loop ---
print(f"Starting training for {NUM_EPISODES} episodes...")
epsilon = EPSILON_START
total_rewards = []

for episode in range(NUM_EPISODES):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    episode_reward = 0
    done = False
    
    while not done:
        # --- Epsilon-Greedy Action Selection ---
        if random.random() < epsilon:
            action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)
        
        # --- Take action and store in memory ---
        next_state_np, reward, done = env.step(action.item())
        episode_reward += reward
        
        next_state = torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)
        
        memory.push(state.cpu().numpy().squeeze(), action.cpu().numpy()[0][0], reward, next_state.cpu().numpy().squeeze(), done)
        state = next_state
        
        # --- Train from Experience Replay ---
        optimize_model()

    # --- End of Episode ---
    total_rewards.append(episode_reward)
    
    if epsilon > EPSILON_END:
        epsilon *= EPSILON_DECAY
        
    # Update the "Teacher" brain
    if (episode + 1) % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())
        
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(total_rewards[-100:])
        print(f"Episode {episode + 1}/{NUM_EPISODES} | Avg Reward (Last 100): {avg_reward:.2f} | Epsilon: {epsilon:.3f} | Memory: {len(memory)}")

# --- 9. Save the Final Trained "Student" Brain ---
SAVED_AGENT_NAME = 'mab_agent_pytorch.pth'
print("\n--- Training Complete! ---")
torch.save(policy_net.state_dict(), SAVED_AGENT_NAME)
print(f"The 'Smart Doctor' (PyTorch Agent) has been trained and saved as '{SAVED_AGENT_NAME}'")

# --- 10. Test the Trained Agent ---
print(f"\n--- Testing Trained Agent (Epsilon = 0) ---")
test_agent = DQN(n_observations, n_actions).to(device)
test_agent.load_state_dict(torch.load(SAVED_AGENT_NAME))
test_agent.eval()

for i in range(3):
    state = env.reset() # 1D numpy array
    patient_id = env.current_patient_index
    true_label = env.labels[env.current_patient_label]
    
    print(f"\n--- New Patient (ID: {patient_id}) ---")
    print(f"GROUND TRUTH: {true_label}")
    
    pathway = []
    done = False
    
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = test_agent(state_tensor).cpu().numpy()[0]
        
        # Mask out actions that are already done
        for j in range(len(env.test_features)):
            if state[env.encoding_dim + j] != -1: 
                q_values[j] = -np.inf 
        
        action = np.argmax(q_values)
        
        if 0 <= action < len(env.test_features):
            action_name = f"Test: {env.test_features[action]}"
            pathway.append(action_name)
        else:
            diagnosis = env.labels[action - len(env.test_features)]
            action_name = f"Diagnose: {diagnosis}"
            pathway.append(action_name)
            
        state, reward, done = env.step(action) # state is 1D, perfect for next loop
        
    print(f"Learned Pathway: {' -> '.join(pathway)}")
    print(f"Final Diagnosis: {diagnosis} (Correct: {true_label == diagnosis})")