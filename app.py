import os
import textwrap
import asyncio
from typing import List, Optional

import gradio as gr
from openai import OpenAI
from env import TrafficMedEnv as RLAgentEnv  # make sure env.py exists


# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# --- Constants ---
MAX_STEPS = 100
MAX_TOTAL_REWARD = 100
SUCCESS_SCORE_THRESHOLD = 0.8

LANE_TO_INDEX = {
    "NORTH": 0,
    "EAST": 1,
    "SOUTH": 2,
    "WEST": 3,
}


# --- RL/OpenAI action function ---
def get_model_action(client: Optional[OpenAI], step: int, state: dict, history: List[str]) -> str:
    # Ambulance priority
    if state.get("ambulance_lane", False):
        return "GREEN_LIGHT_NORTH"  # fallback

    # Use model if API is available
    if client:
        user_prompt = textwrap.dedent(
            f"""
            Step: {step}
            Current state: {state}
            Previous steps:
            {history[-4:] if history else "None"}
            Decide the next traffic light action for NORTH/EAST/SOUTH/WEST lanes.
            """
        ).strip()

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You control traffic lights. Optimize flow."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=50,
            )
            action_text = (completion.choices[0].message.content or "").upper()

            for lane in LANE_TO_INDEX:
                if lane in action_text:
                    return f"GREEN_LIGHT_{lane}"

        except Exception as e:
            print(f"[DEBUG] OpenAI error: {e}")

    # fallback: choose busiest lane
    traffic = state.get("traffic", [0, 0, 0, 0])
    max_index = traffic.index(max(traffic))
    lane = list(LANE_TO_INDEX.keys())[max_index]
    return f"GREEN_LIGHT_{lane}"


# --- Simulation runner ---
async def run_simulation():
    client = None
    if HF_TOKEN and MODEL_NAME:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    env = RLAgentEnv()

    history = []
    rewards = []

    state = env.reset()

    for step in range(1, MAX_STEPS + 1):
        action = get_model_action(client, step, state, history)

        if action.startswith("GREEN_LIGHT_"):
            lane_name = action.replace("GREEN_LIGHT_", "")
            action_index = LANE_TO_INDEX.get(lane_name, 0)
        else:
            action_index = 0

        next_state, reward, done = env.step(action_index)
        print(f"State: {state}, Reward: {reward}")

        rewards.append(reward)
        history.append(f"Step {step}: {action} → {reward:.2f}")

        state = next_state
        if done:
            break

    score = sum(rewards) / len(rewards) if rewards else 0
    score = min(max(score, 0.0), 1.0)

    result = f"""
Simulation Finished ✅

Steps: {len(rewards)}
Score: {score:.2f}
Success: {"YES" if score >= SUCCESS_SCORE_THRESHOLD else "NO"}
"""

    return result


# --- Gradio UI ---
def run_app():
    result = asyncio.run(run_simulation())
    return result


gr.Interface(
    fn=run_app,
    inputs=[],
    outputs="text",
    title="🚦 Traffic RL Simulation",
    description="AI-controlled traffic signal system with ambulance priority"
).launch()