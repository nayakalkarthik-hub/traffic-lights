from env import TrafficMedEnv
import random

class EasyTrafficEnv(TrafficMedEnv):
    def reset(self):
        self.lanes = 4
        self.traffic = [random.randint(5, 10) for _ in range(self.lanes)]
        self.ambulance_lane = random.choice([-1, 0, 1, 2, 3])
        self.severity = 1 if self.ambulance_lane != -1 else 0
        self.time = 0
        return {
            "traffic": self.traffic,
            "ambulance_lane": self.ambulance_lane,
            "severity": self.severity
        }

class MediumTrafficEnv(TrafficMedEnv):
    def reset(self):
        self.lanes = 4
        self.traffic = [random.randint(10, 20) for _ in range(self.lanes)]
        self.ambulance_lane = random.choice([-1, 0, 1, 2, 3])
        self.severity = random.randint(1, 3) if self.ambulance_lane != -1 else 0
        self.time = 0
        return {
            "traffic": self.traffic,
            "ambulance_lane": self.ambulance_lane,
            "severity": self.severity
        }

class HardTrafficEnv(TrafficMedEnv):
    def reset(self):
        self.lanes = 4
        self.traffic = [random.randint(15, 30) for _ in range(self.lanes)]
        self.ambulance_lane = random.choice([0, 1, 2, 3])
        self.severity = random.randint(2, 3)
        self.time = 0
        return {
            "traffic": self.traffic,
            "ambulance_lane": self.ambulance_lane,
            "severity": self.severity
        }