import random

class TrafficMedEnv:
    def __init__(self):
        self.num_lanes = 4
        self.max_steps = 20
        self.reset()

    def reset(self):
        # Random traffic in each lane
        self.traffic = [random.randint(5, 20) for _ in range(self.num_lanes)]

        # Ambulance appears in a random lane
        self.ambulance_lane = random.randint(0, self.num_lanes - 1)

        # Severity (1 = low, 2 = medium, 3 = high)
        self.severity = random.randint(1, 3)

        self.steps = 0

        return self._get_state()

    def _get_state(self):
        return {
            "traffic": self.traffic,
            "ambulance_lane": self.ambulance_lane,
            "severity": self.severity
        }

    def step(self, action):
        self.steps += 1

        reward = 0

        # 🚦 Reduce traffic in selected lane
        cars_cleared = min(5, self.traffic[action])
        self.traffic[action] -= cars_cleared

        # 🚗 Other lanes get more traffic
        for i in range(self.num_lanes):
            if i != action:
                self.traffic[i] += random.randint(0, 3)

        # 🚑 Ambulance priority reward
        if action == self.ambulance_lane:
            reward += 5 * self.severity   # more severity = more reward
        else:
            reward -= 2 * self.severity   # penalty if ignored

        # 🚗 Traffic penalty (more traffic = worse)
        reward -= sum(self.traffic) * 0.01

        # ✅ Check if ambulance cleared
        if self.traffic[self.ambulance_lane] == 0:
            reward += 10   # big reward for clearing ambulance lane
            done = True
        else:
            done = False

        # ⏹ End if max steps reached
        if self.steps >= self.max_steps:
            done = True

import random

class TrafficMedEnv:
    def __init__(self):
        self.num_lanes = 4
        self.max_steps = 20
        self.reset()

    def reset(self):
        # Random traffic in each lane
        self.traffic = [random.randint(5, 20) for _ in range(self.num_lanes)]

        # Ambulance appears in a random lane
        self.ambulance_lane = random.randint(0, self.num_lanes - 1)

        # Severity (1 = low, 2 = medium, 3 = high)
        self.severity = random.randint(1, 3)

        self.steps = 0

        return self._get_state()

    def _get_state(self):
        return {
            "traffic": self.traffic,
            "ambulance_lane": self.ambulance_lane,
            "severity": self.severity
        }

    def step(self, action):
        self.steps += 1

        reward = 0

        # 🚦 Reduce traffic in selected lane
        cars_cleared = min(5, self.traffic[action])
        self.traffic[action] -= cars_cleared

        # 🚗 Other lanes get more traffic
        for i in range(self.num_lanes):
            if i != action:
                self.traffic[i] += random.randint(0, 3)

        # 🚑 Ambulance priority reward
        if action == self.ambulance_lane:
            reward += 5 * self.severity   # more severity = more reward
        else:
            reward -= 2 * self.severity   # penalty if ignored

        # 🚗 Traffic penalty (more traffic = worse)
        reward -= sum(self.traffic) * 0.01

        # ✅ Check if ambulance cleared
        if self.traffic[self.ambulance_lane] == 0:
            reward += 10   # big reward for clearing ambulance lane
            done = True
        else:
            done = False

        # ⏹ End if max steps reached
        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done