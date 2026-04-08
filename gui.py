<<<<<<< HEAD
import tkinter as tk
from env import TrafficMedEnv

# Q-table
q_table = [
    [125.55, 80.67, 79.36, 76.74],
    [101.53, 134.93, 104.93, 107.01],
    [97.61, 93.82, 118.74, 91.75],
    [95.29, 93.65, 89.52, 126.71]
]

env = TrafficMedEnv()
state = env.reset()
done = False

# Create window
root = tk.Tk()
root.title("AI Traffic Control 🚦")

canvas = tk.Canvas(root, width=600, height=400, bg="black")
canvas.pack()

def draw():
    canvas.delete("all")

    traffic = state["traffic"]
    ambulance_lane = state["ambulance_lane"]

    for i in range(4):
        x = 100 + i * 100
        y = 350

        # Draw cars
        for j in range(traffic[i] // 2):
            color = "blue"
            if i == ambulance_lane and j == 0:
                color = "red"  # ambulance

            canvas.create_rectangle(x, y - j * 15, x + 40, y - j * 15 - 10, fill=color)

        # Draw signal
        signal_color = "green" if i == current_action else "red"
        canvas.create_oval(x + 10, 360, x + 30, 380, fill=signal_color)

def update():
    global state, done, current_action

    if not done:
        state_lane = state["ambulance_lane"]
        current_action = q_table[state_lane].index(max(q_table[state_lane]))

        next_state, reward, done = env.step(current_action)

        draw()

        state = next_state

        root.after(1000, update)  # update every second

current_action = 0
update()

=======
import tkinter as tk
from env import TrafficMedEnv

# Q-table
q_table = [
    [125.55, 80.67, 79.36, 76.74],
    [101.53, 134.93, 104.93, 107.01],
    [97.61, 93.82, 118.74, 91.75],
    [95.29, 93.65, 89.52, 126.71]
]

env = TrafficMedEnv()
state = env.reset()
done = False

# Create window
root = tk.Tk()
root.title("AI Traffic Control 🚦")

canvas = tk.Canvas(root, width=600, height=400, bg="black")
canvas.pack()

def draw():
    canvas.delete("all")

    traffic = state["traffic"]
    ambulance_lane = state["ambulance_lane"]

    for i in range(4):
        x = 100 + i * 100
        y = 350

        # Draw cars
        for j in range(traffic[i] // 2):
            color = "blue"
            if i == ambulance_lane and j == 0:
                color = "red"  # ambulance

            canvas.create_rectangle(x, y - j * 15, x + 40, y - j * 15 - 10, fill=color)

        # Draw signal
        signal_color = "green" if i == current_action else "red"
        canvas.create_oval(x + 10, 360, x + 30, 380, fill=signal_color)

def update():
    global state, done, current_action

    if not done:
        state_lane = state["ambulance_lane"]
        current_action = q_table[state_lane].index(max(q_table[state_lane]))

        next_state, reward, done = env.step(current_action)

        draw()

        state = next_state

        root.after(1000, update)  # update every second

current_action = 0
update()

>>>>>>> 85e45758f974eedc1504cf54d6431bd7544ce47b
root.mainloop()