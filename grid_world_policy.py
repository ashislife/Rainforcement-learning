import numpy as np

# Grid size
GRID_SIZE = 4

# Actions: Up, Down, Left, Right
ACTIONS = ['U', 'D', 'L', 'R']

# Discount factor
gamma = 1.0

# Initialize value function
V = np.zeros((GRID_SIZE, GRID_SIZE))

# Initialize random policy
policy = np.random.choice(ACTIONS, (GRID_SIZE, GRID_SIZE))

# Terminal states
terminal_states = [(0, 0), (3, 3)]


# Function to move agent
def step(state, action):
    i, j = state

    if action == 'U':
        i = max(i - 1, 0)
    elif action == 'D':
        i = min(i + 1, GRID_SIZE - 1)
    elif action == 'L':
        j = max(j - 1, 0)
    elif action == 'R':
        j = min(j + 1, GRID_SIZE - 1)

    return (i, j)


# ---------------- POLICY ITERATION ----------------

is_policy_stable = False

while not is_policy_stable:

    # -------- POLICY EVALUATION --------
    for _ in range(100):  # fixed iterations to avoid infinite loop
        delta = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):

                if (i, j) in terminal_states:
                    continue

                old_value = V[i, j]
                action = policy[i, j]
                next_state = step((i, j), action)
                reward = -1

                V[i, j] = reward + gamma * V[next_state[0], next_state[1]]
                delta = max(delta, abs(old_value - V[i, j]))

        if delta < 0.01:
            break

    # -------- POLICY IMPROVEMENT --------
    is_policy_stable = True

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):

            if (i, j) in terminal_states:
                continue

            old_action = policy[i, j]
            action_values = {}

            for action in ACTIONS:
                next_state = step((i, j), action)
                action_values[action] = -1 + gamma * V[next_state[0], next_state[1]]

            best_action = max(action_values, key=action_values.get)
            policy[i, j] = best_action

            if old_action != best_action:
                is_policy_stable = False


# ---------------- OUTPUT ----------------

print("Optimal Value Function:")
print(np.round(V, 2))

print("\nOptimal Policy:")
print(policy)