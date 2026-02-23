import numpy as np
import random

# Parameters
alpha = 0.1    # Learning rate - controls how fast the system learns
gamma = 0.9    # Discount factor - tells how important future rewards are
epsilon = 0.1  # Exploration rate - probability of random action
episodes = 5000

# States:
# 0 = Low NS, Low EW
# 1 = High NS, Low EW
# 2 = Low NS, High EW
# 3 = High NS, High EW

n_states = 4
n_actions = 2    # 0 = Green NS, 1 = Green EW

# Initialize Q-table (4x2 matrix of zeros)
Q = np.zeros((n_states, n_actions))

def get_reward(state, action):
    """
    Calculate reward based on state and action taken
    - If high traffic side gets green → good reward
    - If both sides high → small positive reward
    - If wrong side gets green → penalty
    """
    # High NS traffic and NS gets green
    if state == 1 and action == 0:
        return 10
    # High EW traffic and EW gets green
    elif state == 2 and action == 1:
        return 10
    # Both sides high - small positive regardless of which gets green
    elif state == 3:
        return 5
    # Wrong side gets green or low traffic with any action
    else:
        return -5

def get_next_state():
    """Randomly generates next traffic condition - simulates changing traffic"""
    return random.randint(0, 3)

# Training loop
for episode in range(episodes):
    # Start with a random traffic situation
    state = random.randint(0, 3)
    
    # Each episode has 10 steps
    for step in range(10):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            # Exploration: choose random action
            action = random.randint(0, 1)
        else:
            # Exploitation: choose best known action
            action = np.argmax(Q[state])
        
        # Calculate reward for chosen action
        reward = get_reward(state, action)
        
        # Simulate new traffic situation
        next_state = get_next_state()
        
        # Q-learning update formula:
        # New Q = Old Q + Learning Rate × (Reward + Future Reward - Old Q)
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        
        # Move to next state
        state = next_state

# Display results
print("Trained Q-Table:")
print(Q)
print("\nBest Traffic Decisions:")
for state in range(4):
    action = np.argmax(Q[state])
    if action == 0:
        print(f"State {state} → Green: North-South")
    else:
        print(f"State {state} → Green: East-West")

# Optional: Display state descriptions for clarity
print("\nState Descriptions:")
state_descriptions = [
    "Low NS, Low EW",
    "High NS, Low EW",
    "Low NS, High EW",
    "High NS, High EW"
]
for i, desc in enumerate(state_descriptions):
    print(f"State {i}: {desc}")