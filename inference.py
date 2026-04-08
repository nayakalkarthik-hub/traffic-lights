import os
import openai
import textwrap
import asyncio
from typing import List, Optional
from openai import OpenAI
from env import TrafficMedEnv as RLAgentEnv  # Your RL environment

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN", "<your-api-key>")

openai.api_key = HF_TOKEN

# --- Constants ---
MAX_STEPS = 100
MAX_TOTAL_REWARD = 100
SUCCESS_SCORE_THRESHOLD = 0.8

# Map lane names to traffic list indices
LANE_TO_INDEX = {
    "NORTH": 0,
    "EAST": 1,
    "SOUTH": 2,
    "WEST": 3,
}

# --- Logging functions ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# --- RL/OpenAI action function ---
def get_model_action(client: Optional[OpenAI], step: int, state: dict, history: List[str]) -> str:
    # --- Ambulance always first ---
    if state.get("ambulance_detected", False):
        return "GIVE_WAY_TO_AMBULANCE"

    # --- OpenAI RL decision ---
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
                    {"role": "system", "content": "You are an RL agent controlling a traffic simulation. Optimize flow to maximize reward."},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=50,
            )
            action_text = (completion.choices[0].message.content or "").strip().upper()
            # Ensure we return a valid lane
            for lane in LANE_TO_INDEX.keys():
                if lane in action_text:
                    return f"GREEN_LIGHT_{lane}"
        except Exception as e:
            print(f"[DEBUG] OpenAI error: {e}", flush=True)

    # --- Fallback: pick lane with highest traffic ---
    traffic_counts = state.get("traffic_counts", [])
    if traffic_counts:
        max_index = traffic_counts.index(max(traffic_counts))
        lane = list(LANE_TO_INDEX.keys())[max_index]
        return f"GREEN_LIGHT_{lane}"

    return "NO_OP"

# --- Main async loop ---
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    env = RLAgentEnv()
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task="rl-task", env="traffic-sim", model=MODEL_NAME)

    state = env.reset()  # Initial state

    try:
        for step in range(1, MAX_STEPS + 1):
            action = get_model_action(client, step, state, history)

            # Convert action string to index for env.step()
            if action.startswith("GREEN_LIGHT_"):
                lane_name = action.replace("GREEN_LIGHT_", "")
                action_index = LANE_TO_INDEX.get(lane_name, 0)
            else:
                action_index = 0  # Default if GIVE_WAY_TO_AMBULANCE or NO_OP

            next_state, reward, done = env.step(action_index)
            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done, error=None)
            history.append(f"Step {step}: {action!r} -> reward {reward:.2f}")

            state = next_state
            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        # Removed env.close() since it does not exist
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())