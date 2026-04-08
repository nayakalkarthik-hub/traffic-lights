
from tasks import EasyTrafficEnv, MediumTrafficEnv, HardTrafficEnv

def run_task(env_class, episodes=10):
    env = env_class()
    total_reward = 0
    for episode in range(episodes):
        state_dict = env.reset()  # Get initial state dictionary
        state = state_dict["ambulance_lane"]  # Use ambulance lane as the state

        done = False
        episode_reward = 0
        while not done:
            action = env.random_action()  # Or use a chosen policy
            next_state_dict, reward, done = env.step(action)
            state = next_state_dict["ambulance_lane"]  # Update state from next dictionary
            episode_reward += reward

        total_reward += episode_reward

    average_reward = total_reward / episodes
    print(f"{env_class.__name__} Average Reward: {average_reward}")
    assert 0.0 <= average_reward <= 1.0, f"{env_class.__name__} reward out of range!"

if __name__ == "__main__":
    run_task(EasyTrafficEnv)
    run_task(MediumTrafficEnv)
from tasks import EasyTrafficEnv, MediumTrafficEnv, HardTrafficEnv

def run_task(env_class, episodes=10):
    env = env_class()
    total_reward = 0
    for episode in range(episodes):
        state_dict = env.reset()  # Get initial state dictionary
        state = state_dict["ambulance_lane"]  # Use ambulance lane as the state

        done = False
        episode_reward = 0
        while not done:
            action = env.random_action()  # Or use a chosen policy
            next_state_dict, reward, done = env.step(action)
            state = next_state_dict["ambulance_lane"]  # Update state from next dictionary
            episode_reward += reward

        total_reward += episode_reward

    average_reward = total_reward / episodes
    print(f"{env_class.__name__} Average Reward: {average_reward}")
    assert 0.0 <= average_reward <= 1.0, f"{env_class.__name__} reward out of range!"

if __name__ == "__main__":
    run_task(EasyTrafficEnv)
    run_task(MediumTrafficEnv)
    run_task(HardTrafficEnv)