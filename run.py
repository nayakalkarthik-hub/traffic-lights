# ------------------------------
# run.py
# ------------------------------

from env import TrafficMedEnv

# ------------------------------
# QAgent class
# ------------------------------
class QAgent:
    def __init__(self, q_table):
        self.q_table = q_table

    def choose_action(self, state):
        # Use ambulance_lane as state index
        state_idx = state["ambulance_lane"]
        # Pick action with highest Q-value for that state
        return self.q_table[state_idx].index(max(self.q_table[state_idx]))


# ------------------------------
# Q-table from your trained model
# ------------------------------
q_table = [
    [125.55, 80.67, 79.36, 76.74],
    [101.53, 134.93, 104.93, 107.01],
    [97.61, 93.82, 118.74, 91.75],
    [95.29, 93.65, 89.52, 126.71]
]

# ------------------------------
# Initialize environment and agent
# ------------------------------
env = TrafficMedEnv()
agent = QAgent(q_table)

episodes = 5  # number of episodes to run

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    print(f"\n[START] Episode {ep}")

    while not done:
        # Choose action using Q-table
        action = agent.choose_action(state)

        # Take step in the environment
        next_state, reward, done = env.step(action)

        # Print step info
        print(f"[STEP] action={action}, reward={reward:.2f}, state={state}")

        total_reward += reward
        step += 1
        state = next_state


# ------------------------------
# run.py
# ------------------------------

from env import TrafficMedEnv

# ------------------------------
# QAgent class
# ------------------------------
class QAgent:
    def __init__(self, q_table):
        self.q_table = q_table

    def choose_action(self, state):
        # Use ambulance_lane as state index
        state_idx = state["ambulance_lane"]
        # Pick action with highest Q-value for that state
        return self.q_table[state_idx].index(max(self.q_table[state_idx]))


# ------------------------------
# Q-table from your trained model
# ------------------------------
q_table = [
    [125.55, 80.67, 79.36, 76.74],
    [101.53, 134.93, 104.93, 107.01],
    [97.61, 93.82, 118.74, 91.75],
    [95.29, 93.65, 89.52, 126.71]
]

# ------------------------------
# Initialize environment and agent
# ------------------------------
env = TrafficMedEnv()
agent = QAgent(q_table)

episodes = 5  # number of episodes to run

for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    step = 0

    print(f"\n[START] Episode {ep}")

    while not done:
        # Choose action using Q-table
        action = agent.choose_action(state)

        # Take step in the environment
        next_state, reward, done = env.step(action)

        # Print step info
        print(f"[STEP] action={action}, reward={reward:.2f}, state={state}")

        total_reward += reward
        step += 1
        state = next_state
    print(f"[END] Total Reward: {total_reward:.2f}")