import numpy as np
import random

# Environment dimensions
grid_size = 5
start = (0, 0)
goal = (4, 4)
actions = ['up', 'down', 'left', 'right']

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 500

# Initialize the Q-table with zeros
q_table = np.zeros((grid_size, grid_size, len(actions)))

# Define rewards
def get_reward(state):
    if state == goal:
        return 10
    else:
        return -1

# Helper function: Check if a move is valid
def is_valid(state):
    return 0 <= state[0] < grid_size and 0 <= state[1] < grid_size

# Helper function: Make a move and return new state
def make_move(state, action):
    if action == 'up':
        new_state = (state[0] - 1, state[1])
    elif action == 'down':
        new_state = (state[0] + 1, state[1])
    elif action == 'left':
        new_state = (state[0], state[1] - 1)
    elif action == 'right':
        new_state = (state[0], state[1] + 1)

    # Return to old state if move is out of bounds
    return new_state if is_valid(new_state) else state

# Training the agent
for episode in range(num_episodes):
    state = start
    done = False

    while not done:
        # Choose action (with epsilon-greedy policy)
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)  # Explore
        else:
            action = actions[np.argmax(q_table[state[0], state[1]])]  # Exploit

        # Take action and observe new state and reward
        next_state = make_move(state, action)
        reward = get_reward(next_state)

        # Update Q-value
        action_index = actions.index(action)
        best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action_index] += alpha * (
            reward + gamma * q_table[next_state[0], next_state[1], best_next_action]
            - q_table[state[0], state[1], action_index]
        )

        # Move to the next state
        state = next_state

        # Check if the goal is reached
        if state == goal:
            done = True

# Testing the agent after training
print("Trained Q-Table:")
print(q_table)

# Displaying the learned policy
state = start
path = [state]
while state != goal:
    action_index = np.argmax(q_table[state[0], state[1]])
    action = actions[action_index]
    state = make_move(state, action)
    path.append(state)

print("Path taken by the agent:", path)

