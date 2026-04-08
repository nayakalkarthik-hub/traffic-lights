import tkinter as tk
import random

# ------------------------------
# Simulation Parameters
# ------------------------------
NUM_LANES = 4
MAX_VEHICLES_PER_LANE = 20
STEP_DELAY = 1000  # milliseconds
Q_TABLE = [
    [125.55, 80.67, 79.36, 76.74],
    [101.53, 134.93, 104.93, 107.01],
    [97.61, 93.82, 118.74, 91.75],
    [95.29, 93.65, 89.52, 126.71]
]

# ------------------------------
# Traffic Environment
# ------------------------------
class TrafficMedEnv:
    def __init__(self):
        self.num_lanes = NUM_LANES
        self.max_steps = 20
        self.reset()

    def reset(self):
        # Random traffic
        self.traffic = [random.randint(5, 15) for _ in range(self.num_lanes)]
        # Ambulance in random lane
        self.ambulance_lane = random.randint(0, self.num_lanes - 1)
        self.severity = random.randint(1, 3)
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        return {
            "traffic": self.traffic.copy(),
            "ambulance_lane": self.ambulance_lane,
            "severity": self.severity
        }

    def step(self, action):
        self.steps += 1
        reward = 0

        # Reduce traffic in chosen lane
        cleared = min(5, self.traffic[action])
        self.traffic[action] -= cleared

        # Other lanes get more traffic
        for i in range(self.num_lanes):
            if i != action:
                self.traffic[i] += random.randint(0, 3)
                if self.traffic[i] > MAX_VEHICLES_PER_LANE:
                    self.traffic[i] = MAX_VEHICLES_PER_LANE

        # Reward for ambulance lane
        if action == self.ambulance_lane:
            reward += 5 * self.severity
        else:
            reward -= 2 * self.severity

        # Penalty for traffic
        reward -= sum(self.traffic) * 0.01

        # Check if ambulance cleared
        done = False
        if self.traffic[self.ambulance_lane] == 0 or self.steps >= self.max_steps:
            done = True
            if self.traffic[self.ambulance_lane] == 0:
                reward += 10

        return self._get_state(), reward, done

# ------------------------------
# GUI Simulation
# ------------------------------
root = tk.Tk()
root.title("AI Traffic Control 🚦")
root.geometry("600x400")
canvas = tk.Canvas(root, width=600, height=400, bg="black")
canvas.pack()

env = TrafficMedEnv()
state = env.reset()
done = False
current_action = 0
total_reward = 0
step_count = 0

def choose_action(state):
    """Use Q-table to select best action based on ambulance lane."""
    lane = state["ambulance_lane"]
    return Q_TABLE[lane].index(max(Q_TABLE[lane]))

def draw():
    canvas.delete("all")
    # Draw lanes
    lane_colors = ["white"]*NUM_LANES
    for i in range(NUM_LANES):
        x = 100 + i*100
        y = 350
        # Draw cars
        for j in range(state["traffic"][i]//2):
            color = "blue"
            if i == state["ambulance_lane"] and j == 0:
                color = "red"  # ambulance
            canvas.create_rectangle(x, y - j*15, x+40, y - j*15 - 10, fill=color)
        # Draw signal
        signal_color = "green" if i == current_action else "red"
        canvas.create_oval(x+10, 360, x+30, 380, fill=signal_color)

def update():
    global state, done, current_action, total_reward, step_count

    if not done:
        current_action = choose_action(state)
        next_state, reward, done = env.step(current_action)

        print(f"[STEP] action={current_action}, reward={reward:.2f}, state={state}")

        total_reward += reward
        step_count += 1
        state = next_state

        draw()
        root.after(STEP_DELAY, update)
    else:
        print(f"[END] Total Reward: {total_reward:.2f}")

# Start simulation
update()
import tkinter as tk
import random

# ------------------------------
# Simulation Parameters
# ------------------------------
NUM_LANES = 4
MAX_VEHICLES_PER_LANE = 20
STEP_DELAY = 1000  # milliseconds
Q_TABLE = [
    [125.55, 80.67, 79.36, 76.74],
    [101.53, 134.93, 104.93, 107.01],
    [97.61, 93.82, 118.74, 91.75],
    [95.29, 93.65, 89.52, 126.71]
]

# ------------------------------
# Traffic Environment
# ------------------------------
class TrafficMedEnv:
    def __init__(self):
        self.num_lanes = NUM_LANES
        self.max_steps = 20
        self.reset()

    def reset(self):
        # Random traffic
        self.traffic = [random.randint(5, 15) for _ in range(self.num_lanes)]
        # Ambulance in random lane
        self.ambulance_lane = random.randint(0, self.num_lanes - 1)
        self.severity = random.randint(1, 3)
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        return {
            "traffic": self.traffic.copy(),
            "ambulance_lane": self.ambulance_lane,
            "severity": self.severity
        }

    def step(self, action):
        self.steps += 1
        reward = 0

        # Reduce traffic in chosen lane
        cleared = min(5, self.traffic[action])
        self.traffic[action] -= cleared

        # Other lanes get more traffic
        for i in range(self.num_lanes):
            if i != action:
                self.traffic[i] += random.randint(0, 3)
                if self.traffic[i] > MAX_VEHICLES_PER_LANE:
                    self.traffic[i] = MAX_VEHICLES_PER_LANE

        # Reward for ambulance lane
        if action == self.ambulance_lane:
            reward += 5 * self.severity
        else:
            reward -= 2 * self.severity

        # Penalty for traffic
        reward -= sum(self.traffic) * 0.01

        # Check if ambulance cleared
        done = False
        if self.traffic[self.ambulance_lane] == 0 or self.steps >= self.max_steps:
            done = True
            if self.traffic[self.ambulance_lane] == 0:
                reward += 10

        return self._get_state(), reward, done

# ------------------------------
# GUI Simulation
# ------------------------------
root = tk.Tk()
root.title("AI Traffic Control 🚦")
root.geometry("600x400")
canvas = tk.Canvas(root, width=600, height=400, bg="black")
canvas.pack()

env = TrafficMedEnv()
state = env.reset()
done = False
current_action = 0
total_reward = 0
step_count = 0

def choose_action(state):
    """Use Q-table to select best action based on ambulance lane."""
    lane = state["ambulance_lane"]
    return Q_TABLE[lane].index(max(Q_TABLE[lane]))

def draw():
    canvas.delete("all")
    # Draw lanes
    lane_colors = ["white"]*NUM_LANES
    for i in range(NUM_LANES):
        x = 100 + i*100
        y = 350
        # Draw cars
        for j in range(state["traffic"][i]//2):
            color = "blue"
            if i == state["ambulance_lane"] and j == 0:
                color = "red"  # ambulance
            canvas.create_rectangle(x, y - j*15, x+40, y - j*15 - 10, fill=color)
        # Draw signal
        signal_color = "green" if i == current_action else "red"
        canvas.create_oval(x+10, 360, x+30, 380, fill=signal_color)

def update():
    global state, done, current_action, total_reward, step_count

    if not done:
        current_action = choose_action(state)
        next_state, reward, done = env.step(current_action)

        print(f"[STEP] action={current_action}, reward={reward:.2f}, state={state}")

        total_reward += reward
        step_count += 1
        state = next_state

        draw()
        root.after(STEP_DELAY, update)
    else:
        print(f"[END] Total Reward: {total_reward:.2f}")

# Start simulation
update()
root.mainloop()