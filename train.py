from env import TrafficMedEnv
import random

# Q-table (state: 0-3 → 4 actions)
q_table = [[0 for _ in range(4)] for _ in range(4)]

# Hyperparameters
alpha = 0.1      # learning rate
gamma = 0.9      # discount factor
epsilon = 1.0    # exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 500

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # explore
    else:
        return q_table[state].index(max(q_table[state]))  # exploit


def train():
    global epsilon

    for ep in range(episodes):
        env = TrafficMedEnv()
        state_dict = env.reset()

        # 🔥 simplified state
        state = state_dict["ambulance_lane"]

        done = False
        total_reward = 0

        while not done:
            action = choose_action(state)

            next_state_dict, reward, done = env.step(action)
            next_state = next_state_dict["ambulance_lane"]

            # Q-learning update
            old_value = q_table[state][action]
            next_max = max(q_table[next_state])

            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state][action] = new_value

            state = next_state
            total_reward += reward

        # decay exploration
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {ep+1}, Total Reward: {total_reward:.2f}")

    print("\n🎯 Training Complete!")
    print("Q-Table:")
    for i, row in enumerate(q_table):
        print(f"State {i}: {row}")


if __name__ == "__main__":
    train()