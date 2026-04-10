import random

class TrafficMedEnv:
    def __init__(self):
        self.num_lanes = 4
        self.max_steps = 20
        self.reset()

    def reset(self):
        self.traffic = [random.randint(5, 20) for _ in range(self.num_lanes)]
        self.ambulance_lane = random.randint(0, self.num_lanes - 1)
        self.steps = 0
        return {
            "traffic_counts": self.traffic,
            "ambulance_detected": True,
            "ambulance_lane": self.ambulance_lane
        }

    def step(self, action):
        # Clear traffic in chosen lane
        cleared = min(5, self.traffic[action])
        self.traffic[action] -= cleared

        # Add traffic to other lanes
        for i in range(self.num_lanes):
            if i != action:
                self.traffic[i] += 1

        # Reward logic
        reward = cleared

        if action == self.ambulance_lane:
            reward += 10   # correct lane
        else:
            reward -= 5    # wrong lane

        self.steps += 1
        done = self.steps >= self.max_steps

        return {
            "traffic_counts": self.traffic,
            "ambulance_detected": True,
            "ambulance_lane": self.ambulance_lane
        }, reward, done, {}
