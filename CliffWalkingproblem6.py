import numpy as np
import random

# Grid size
rows, cols = 4, 12

# Actions: 0=Up, 1=Down, 2=Left, 3=Right
actions = 4

# Q-table
Q = np.zeros((rows, cols, actions))

# Learning parameters
alpha = 0.1    # learning rate
gamma = 0.9    # discount factor
epsilon = 0.1   # exploration rate

# Start and Goal positions
start = (3, 0)
goal = (3, 11)

# Cliff cells
cliff = [(3, j) for j in range(1, 11)]

# Take action
def step(state, action):
    i, j = state

    if action == 0:    # Up
        i = max(i - 1, 0)
    elif action == 1:    # Down
        i = min(i + 1, rows - 1)
    elif action == 2:    # Left
        j = max(j - 1, 0)
    elif action == 3:    # Right
        j = min(j + 1, cols - 1)

    next_state = (i, j)

    if next_state in cliff:
        return start, -100

    if next_state == goal:
        return next_state, 0

    return next_state, -1

# Q-Learning
for episode in range(500):
    state = start

    while state != goal:
        i, j = state

        # Epsilon-greedy action
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[i, j])

        next_state, reward = step(state, action)
        ni, nj = next_state

        # Update
        Q[i, j, action] += alpha * (reward + gamma * np.max(Q[ni, nj]) - Q[i, j, action])

        state = next_state


print("Learned Policy:")
for i in range(rows):
    for j in range(cols):
        if (i, j) == goal:
            print("G", end=" ")
        elif (i, j) in cliff:
            print("C", end=" ")
        else:
            print(np.argmax(Q[i, j]), end=" ")
    print()