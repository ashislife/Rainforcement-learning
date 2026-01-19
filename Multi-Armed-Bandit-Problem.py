import numpy as np
import random

# STEP 1: Define the environment
true_rewards = [0.2, 0.5, 0.75]  # Actual (unknown) success probabilities of each arm
num_arms = len(true_rewards)     # Total number of arms

# STEP 2: Initialize variables
estimated_values = np.zeros(num_arms)  # Estimated reward of each arm (initially 0)
arm_counts = np.zeros(num_arms)        # Number of times each arm is selected
epsilon = 0.1                          # Exploration probability (10% chance of exploration)
total_reward = 0                       # Total reward collected
iterations = 1000                      # Total number of times the agent will act

# STEP 3: Main learning loop
for t in range(iterations):
    # Decide whether to explore or exploit
    if random.random() < epsilon:
        arm = random.randint(0, num_arms - 1)  # Exploration: choose a random arm
    else:
        arm = np.argmax(estimated_values)      # Exploitation: choose the best estimated arm
    
    # STEP 4: Generate reward
    reward = 1 if random.random() < true_rewards[arm] else 0  # 1 = success, 0 = failure
    
    # STEP 5: Update values
    arm_counts[arm] += 1  # Increase count of selected arm
    old_value = estimated_values[arm]  # Store old estimated value
    
    # Update estimated value using incremental mean formula
    estimated_values[arm] = old_value + (reward - old_value) / arm_counts[arm]
    
    total_reward += reward  # Add reward to total

# STEP 6: Final results
print("Estimated values of arms:", estimated_values)
print("Number of times each arm was selected:", arm_counts)
print("Total reward collected:", total_reward)
print("Best arm according to agent:", np.argmax(estimated_values))

# Optional: Show optimal performance comparison
optimal_reward = max(true_rewards) * iterations
regret = optimal_reward - total_reward
print("\nOptimal reward if always choosing best arm:", optimal_reward)
print("Regret (optimal - actual):", regret)