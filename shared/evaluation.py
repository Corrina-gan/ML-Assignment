"""Score a policy on a fresh warehouse. Every algorithm should report this result."""

import numpy as np

from environment.config import ENV_ID, EVAL_EPISODES, EVAL_SEEDS, GAMMA, N_AGENTS
from environment.environment import make_warehouse
from shared.reward_manager import reward_is_delivery_unit
from shared.training_utils import run_episode


def _seed_text(seeds):
    if list(seeds) == list(range(seeds[0], seeds[0] + len(seeds))):
        if len(seeds) == 1:
            return str(seeds[0])
        return f"{seeds[0]}..{seeds[-1]}"
    return ", ".join(str(seed) for seed in seeds)


def evaluate(choose_actions, episodes=EVAL_EPISODES, seeds=None, on_episode_start=None):
    """Score a policy on the shared warehouse.

    Uses seeds 0 through 9 unless seeds is passed. The env is a new
    warehouse, so this does not reset an environment used for training.
    The compared number is team_return_mean: mean team reward per episode.
    """
    if seeds is None:
        seeds = EVAL_SEEDS[:episodes]
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("evaluate needs at least one seed")

    env = make_warehouse()
    try:
        played = [
            run_episode(
                env,
                choose_actions,
                seed=seed,
                on_episode_start=on_episode_start,
            )
            for seed in seeds
        ]
    finally:
        env.close()

    team_returns = [episode.team_return for episode in played]
    lengths = [episode.length for episode in played]
    robot_returns = [episode.robot_returns for episode in played]
    measurable = all(
        reward_is_delivery_unit(step.rewards)
        for episode in played
        for step in episode.transitions
    )
    deliveries = [episode.deliveries for episode in played] if measurable else None
    return {
        "env_id": ENV_ID,
        "gamma": GAMMA,
        "episodes": len(played),
        "seeds": seeds,
        "team_return_mean": float(np.mean(team_returns)),
        "team_return_std": float(np.std(team_returns, ddof=1)) if len(team_returns) > 1 else 0.0,
        "length_mean": float(np.mean(lengths)),
        "deliveries_mean": None if deliveries is None else float(np.mean(deliveries)),
        "robot_return_mean": tuple(
            float(np.mean([returns[index] for returns in robot_returns]))
            for index in range(N_AGENTS)
        ),
    }


def format_score(score):
    """One-line report. Every algorithm should print this from evaluate()."""
    robots = ", ".join(f"{value:.2f}" for value in score["robot_return_mean"])
    deliveries = score["deliveries_mean"]
    delivery_text = "n/a" if deliveries is None else f"{deliveries:.2f}"
    return (
        f"{score['env_id']}: team return {score['team_return_mean']:.2f} "
        f"± {score['team_return_std']:.2f} "
        f"over {score['episodes']} episodes "
        f"(seeds {_seed_text(score['seeds'])}, "
        f"mean deliveries {delivery_text}, "
        f"robot returns {robots})"
    )
