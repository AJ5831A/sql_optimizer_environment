"""
Inference Script — SQL Query Optimizer
=======================================
Environment variables required:
    API_BASE_URL        LLM endpoint  (default: HuggingFace router)
    MODEL_NAME          Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN / API_KEY  Auth token
    DATABASE_URL        Postgres connection string for the sample DB

STDOUT FORMAT:
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from sql_optimizer import SQLOptimizerEnv, SQLAction

# ── Environment config ────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# Updated to use the docker-compose service name "postgres"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sqlopt:sqlopt@postgres:5432/sqlopt")

TASK_NAME    = "sql_optimization"
BENCHMARK    = "sql_optimizer"
MAX_STEPS    = int(os.getenv("MAX_STEPS", "10"))
TEMPERATURE  = 0.2   # low — we want deterministic rewrites, not creative ones
MAX_TOKENS   = 512

# Score threshold to count episode as success (≥10% improvement)
SUCCESS_SCORE_THRESHOLD = 0.1

# The slow query the agent will try to optimize
# Uses the sample DB seeded by docker-compose
SLOW_QUERY = textwrap.dedent("""
    SELECT *
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN regions r ON c.region_id = r.region_id
    WHERE o.status = 'completed'
    AND r.country = 'US'
""").strip()

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SQL query optimizer working as an RL agent.

    At each step you receive:
    - The current SQL query
    - A feature vector describing query performance
    - A list of legal actions you may apply

    Your job is to pick the single best action to improve query performance.

    Rules:
    - You MUST choose from the legal_actions list only
    - Reply with a single valid JSON object and nothing else
    - Format: {"action_id": <int>, "params": <dict>}
    - If no improvement seems possible, use action_id=9 (submit) with params={}

    Examples:
    {"action_id": 7, "params": {}}
    {"action_id": 4, "params": {"target_table": "customers"}}
    {"action_id": 9, "params": {}}
""").strip()


def build_user_prompt(
    step: int,
    current_query: str,
    observation_vector: List[float],
    legal_actions: List[Dict[str, Any]],
    last_reward: float,
    history: List[str],
) -> str:
    # Label the observation vector for the model
    feature_names = [
        "execution_time_ms",
        "total_plan_cost",
        "has_seq_scan",
        "has_subquery",
        "max_rows_removed",
        "num_joins",
        "has_redundant_join",
        "has_cte",
        "has_select_star",
        "estimated_vs_actual_gap",
    ]
    features = {
        name: round(val, 3)
        for name, val in zip(feature_names, observation_vector)
    }

    history_block = "\n".join(history[-4:]) if history else "None"

    actions_block = json.dumps(legal_actions, indent=2)

    return textwrap.dedent(f"""
        Step: {step}/{MAX_STEPS}
        Last reward: {last_reward:+.4f}

        Current SQL:
        {current_query}

        Performance features:
        {json.dumps(features, indent=2)}

        Legal actions:
        {actions_block}

        Recent history:
        {history_block}

        Choose the best action. Reply with JSON only.
    """).strip()


# ── Logging ───────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call ──────────────────────────────────────────────────────────────────

def get_model_action(
    client: OpenAI,
    step: int,
    current_query: str,
    observation_vector: List[float],
    legal_actions: List[Dict[str, Any]],
    last_reward: float,
    history: List[str],
) -> Dict[str, Any]:
    """Ask the LLM to pick an action. Falls back to submit on any error."""
    user_prompt = build_user_prompt(
        step, current_query, observation_vector, legal_actions, last_reward, history
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if model wraps response
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        # Validate the chosen action is actually in legal_actions
        legal_ids = {a["action_id"] for a in legal_actions}
        if parsed.get("action_id") not in legal_ids:
            print(
                f"[DEBUG] Model chose illegal action_id={parsed.get('action_id')}, "
                f"falling back to submit",
                flush=True,
            )
            return {"action_id": 9, "params": {}}

        return parsed

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_id": 9, "params": {}}


# ── Main loop ─────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Directly connect to the docker-compose server running locally
    env = SQLOptimizerEnv("http://localhost:8000")

    history: List[str]   = []
    rewards: List[float] = []
    steps_taken          = 0
    score                = 0.0
    success              = False
    baseline_ms          = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── Reset ─────────────────────────────────────────────────────────────
        result = await env.reset(
            query=SLOW_QUERY,
            db_url=DATABASE_URL,
        )

        obs         = result.observation
        last_reward = 0.0
        baseline_ms = obs.metadata.get("baseline_ms", 0.0)

        print(
            f"[DEBUG] baseline={baseline_ms:.1f}ms "
            f"features={obs.observation_vector}",
            flush=True,
        )

        # ── Step loop ─────────────────────────────────────────────────────────
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            chosen = get_model_action(
                client=client,
                step=step,
                current_query=obs.current_query,
                observation_vector=obs.observation_vector,
                legal_actions=obs.legal_actions,
                last_reward=last_reward,
                history=history,
            )

            action     = SQLAction(
                action_id=chosen["action_id"],
                params=chosen.get("params", {}),
            )
            action_str = json.dumps(chosen)

            result      = await env.step(action)
            obs         = result.observation
            reward      = result.reward or 0.0
            done        = result.done
            last_reward = reward

            rewards.append(reward)
            steps_taken = step

            improvement = obs.metadata.get("improvement_pct", 0.0)
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=None,
            )

            history.append(
                f"Step {step}: {action_str} -> "
                f"reward={reward:+.4f} improvement={improvement:.1f}%"
            )

            if done:
                break

        # ── Score ─────────────────────────────────────────────────────────────
        # Score = total cumulative reward clamped to [0, 1]
        # Positive reward = query got faster relative to baseline
        total_reward = sum(rewards)
        score        = min(max(total_reward, 0.0), 1.0)
        success      = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )


if __name__ == "__main__":
    asyncio.run(main())