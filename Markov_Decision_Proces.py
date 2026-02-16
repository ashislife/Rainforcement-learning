# SIMPLE MDP USING DYNAMIC PROGRAMMING

# States
states = ['A', 'B']

# Discount factor
gamma = 0.9

# Initial policy (randomly chosen)
policy = {'A': 'left'}

# Initial state values
V = {'A': 0, 'B': 0}

print("Initial Policy:", policy)
print("Initial Values:", V)

# POLICY ITERATION
print("\n" + "="*50)
print("POLICY ITERATION")
print("="*50)

# Reset values for policy iteration
V = {'A': 0, 'B': 0}

for i in range(5):  # Policy iteration loop
    print(f"\n--- Policy Iteration Step {i+1} ---")
    
    # Policy Evaluation (multiple iterations to converge)
    for _ in range(20):  # Enough iterations for convergence
        old_V_A = V['A']
        if policy['A'] == 'right':
            V['A'] = 10 + gamma * V['B']
        else:  # 'left'
            V['A'] = 0 + gamma * V['A']
        # Optional: break if change is very small
        if abs(V['A'] - old_V_A) < 1e-6:
            break
    
    # Policy Improvement
    left_value = 0 + gamma * V['A']
    right_value = 10 + gamma * V['B']
    
    old_policy = policy['A']
    if right_value > left_value:
        policy['A'] = 'right'
    else:
        policy['A'] = 'left'
    
    print(f"Policy: A -> {policy['A']}")
    print(f"Value of A: {V['A']:.4f}")
    print(f"Left value: {left_value:.4f}, Right value: {right_value:.4f}")
    
    # Check for convergence
    if old_policy == policy['A']:
        print(f"Policy converged after {i+1} iterations!")
        break

print(f"\nFinal Policy from Policy Iteration: {policy}")
print(f"Final Values: A={V['A']:.4f}, B={V['B']:.4f}")

# VALUE ITERATION
print("\n" + "="*50)
print("VALUE ITERATION")
print("="*50)

# Reset values for value iteration
V = {'A': 0, 'B': 0}

for i in range(30):  # More iterations for value iteration
    old_V_A = V['A']
    # Update both states if we had more dynamics
    # For state A:
    V['A'] = max(
        0 + gamma * V['A'],  # left
        10 + gamma * V['B']  # right
    )
    # For state B (assuming it's a terminal state with no actions)
    # V['B'] remains 0 in this simple example
    
    # Optional: Print progress
    if i < 5 or i % 5 == 0:
        print(f"Iteration {i+1}: V(A) = {V['A']:.4f}")
    
    # Check for convergence
    if abs(V['A'] - old_V_A) < 1e-6:
        print(f"Value iteration converged after {i+1} iterations!")
        break

# Extract optimal policy from optimal values
optimal_action = 'right' if (10 + gamma * V['B']) > (0 + gamma * V['A']) else 'left'

print(f"\nOptimal Values: A={V['A']:.4f}, B={V['B']:.4f}")
print(f"Optimal Action at A: {optimal_action}")

# FINAL RESULT
print("\n" + "="*50)
print("FINAL RESULT")
print("="*50)
print("Agent learned that the best action is to go RIGHT from state A")

# Add some analysis
print("\n" + "-"*30)
print("ANALYSIS:")
print("-"*30)
print(f"Value of going LEFT: {0 + gamma * V['A']:.4f}")
print(f"Value of going RIGHT: {10 + gamma * V['B']:.4f}")

if optimal_action == 'right':
    print("✓ RIGHT is indeed the optimal action!")
else:
    print("✗ LEFT would be the optimal action (unexpected!)")

# Extended example with more states (optional extension)
print("\n" + "="*50)
print("EXTENDED EXAMPLE WITH 3 STATES")
print("="*50)

# Define a 3-state MDP
states_ext = ['A', 'B', 'C']
V_ext = {'A': 0, 'B': 0, 'C': 0}
# Transition dynamics for value iteration
# A: left -> A, right -> B (reward 10)
# B: left -> A, right -> C (reward 5)  
# C: terminal state (reward 0)

print("Performing value iteration for 3-state MDP...")
for i in range(50):
    old_V = V_ext.copy()
    
    # Update each state
    V_ext['A'] = max(
        0 + gamma * V_ext['A'],  # left: stay in A
        10 + gamma * V_ext['B']  # right: go to B, get reward 10
    )
    V_ext['B'] = max(
        0 + gamma * V_ext['A'],  # left: go to A
        5 + gamma * V_ext['C']   # right: go to C, get reward 5
    )
    # C is terminal
    
    # Check convergence
    max_change = max(abs(V_ext[s] - old_V[s]) for s in states_ext)
    if max_change < 1e-6:
        print(f"Converged after {i+1} iterations!")
        break

print(f"\nOptimal values for 3-state MDP:")
for state in states_ext:
    print(f"  V({state}) = {V_ext[state]:.4f}")





