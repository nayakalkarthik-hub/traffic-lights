from fastapi import FastAPI
from pydantic import BaseModel
from env import TrafficMedEnv

app = FastAPI()
env = TrafficMedEnv()

class Action(BaseModel):
    action: int

@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state}

@app.post("/step")
def step(action: Action):
    state, reward, done, info = env.step(action.action)
    return {
        "state": state,
        "reward": reward,
        "done": done,
        "info": info
    }
