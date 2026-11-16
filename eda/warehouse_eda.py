"""Inspect rware-tiny-2ag-v2 and summarise a reproducible random policy.

Run from the project root:

    python -m eda.warehouse_eda

The script only reads the installed warehouse. It does not change the
reward, the transition rules, or the registered environment.
"""

from __future__ import annotations

import json
import traceback
import warnings
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rware
from matplotlib.patches import Patch
from PIL import Image
from rware.warehouse import (
    _LAYER_AGENTS,
    _LAYER_SHELFS,
    Action,
    Direction,
    RewardType,
)

# Same id the rest of the project creates with gym.make.
ENV_ID = "rware-tiny-2ag-v2"
N_EPISODES = 100
BASE_SEED = 0
SAMPLE_SEED = 0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "eda"

CELL = 36
FLOOR = (248, 248, 248)
AISLE = (230, 230, 230)
GRID = (190, 190, 190)
SHELF = (72, 61, 139)
REQUESTED = (0, 128, 128)
GOAL = (60, 60, 60)
CARRY = (255, 255, 255)
FACING = (20, 20, 20)
ROBOT_COLORS = (
    (255, 140, 0),
    (30, 136, 229),
    (67, 160, 71),
    (142, 36, 170),
)
# Clockwise order used by Agent.req_direction.
TURN_ORDER = (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)

# Gymnasium warns because this environment returns a list of rewards.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"gymnasium\.utils\.passive_env_checker",
)


def main():
    """Run inspection, the random baseline, and write every EDA output."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        env = gym.make(ENV_ID)
        inspection, observation = inspect_environment(env)
        probe = probe_actions(env)
        episodes, sample, blocked, ranges, reward_values = run_random_baseline(env)
        summary = summarise_baseline(episodes, blocked, ranges, reward_values)
        notes = algorithm_notes(inspection, observation, summary)
        save_tables(inspection, observation, probe, episodes, summary, blocked)
        save_figures(inspection, sample, episodes, summary)
        report = build_report(inspection, observation, probe, summary, notes)
        (OUTPUT_DIR / "eda_report.md").write_text(report, encoding="utf-8")
        payload = {
            "inspection": inspection,
            "observation": observation,
            "probe": probe,
            "blocked_movements": blocked,
            "summary": summary,
            "algorithm_notes": notes,
        }
        write_json(OUTPUT_DIR / "eda_results.json", payload)
    except Exception:
        print("Warehouse EDA failed.", flush=True)
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()

    print(f"Saved EDA outputs to {OUTPUT_DIR}")
    print(
        f"Random baseline over {summary['episodes']} episodes "
        f"(seeds {summary['first_seed']}..{summary['last_seed']}): "
        f"mean reward {summary['mean_reward']:.4f}, "
        f"median {summary['median_reward']:.4f}, "
        f"sample std {summary['std_reward']:.4f}, "
        f"min {summary['min_reward']:.4f}, "
        f"max {summary['max_reward']:.4f}, "
        f"mean length {summary['mean_length']:.2f}."
    )
    if summary["deliveries_measurable"]:
        print(
            f"Deliveries: {summary['total_deliveries']} total, "
            f"success rate {summary['delivery_success_rate']:.2%}."
        )
    else:
        print("Deliveries were not measured from the reward values.")


def inspect_environment(env):
    """Read the live environment and check the observation layout against it."""
    warehouse = env.unwrapped
    observation, info = env.reset(seed=SAMPLE_SEED)
    _require_reset_contract(env, observation, info)
    agent_layer, shelf_layer = _require_layers(warehouse)
    encoded = [
        encode_observation(warehouse, agent, agent_layer, shelf_layer)
        for agent in warehouse.agents
    ]
    for index, (actual, expected) in enumerate(zip(observation, encoded)):
        if not np.array_equal(np.asarray(actual, dtype=np.float32), expected):
            mismatch = np.flatnonzero(
                np.asarray(actual, dtype=np.float32) != expected
            )
            raise RuntimeError(
                f"robot {index + 1} observation does not match the warehouse "
                f"feature layout at indices {mismatch[:8].tolist()}"
            )

    rows, columns = (int(warehouse.grid_size[0]), int(warehouse.grid_size[1]))
    goals = [(int(x), int(y)) for x, y in warehouse.goals]
    storage = [
        (x, y)
        for y in range(rows)
        for x in range(columns)
        if int(warehouse.highways[y, x]) == 0
    ]
    shelves = [(int(shelf.x), int(shelf.y)) for shelf in warehouse.shelfs]
    requested = [(int(shelf.x), int(shelf.y)) for shelf in warehouse.request_queue]
    robots = [_robot_record(agent, warehouse) for agent in warehouse.agents]
    components = component_rows(warehouse)
    if len(components) != int(observation[0].shape[0]):
        raise RuntimeError(
            f"observation has {observation[0].shape[0]} values, "
            f"but the feature table has {len(components)}"
        )

    goal_highways = all(int(warehouse.highways[y, x]) == 1 for x, y in goals)
    shelves_on_goals = sum(1 for cell in shelves if cell in set(goals))
    inspection = {
        "env_id": env.spec.id if env.spec is not None else ENV_ID,
        "registration_kwargs": _plain_kwargs(env.spec.kwargs if env.spec else {}),
        "wrapper": type(env).__name__,
        "grid_rows": rows,
        "grid_columns": columns,
        "n_agents": int(warehouse.n_agents),
        "n_shelves": len(shelves),
        "storage_cells": storage,
        "shelves_match_storage": sorted(shelves) == sorted(storage),
        "goals": goals,
        "goals_are_highways": goal_highways,
        "shelves_on_goals": shelves_on_goals,
        "seed0_robots": robots,
        "seed0_requests": requested,
        "seed0_shelves": shelves,
        "column_height": int(warehouse.column_height),
        "sensor_range": int(warehouse.sensor_range),
        "msg_bits": int(warehouse.msg_bits),
        "request_queue_size": int(warehouse.request_queue_size),
        "normalised_coordinates": bool(warehouse.normalised_coordinates),
        "fast_obs": bool(warehouse.fast_obs),
        "reward_type": warehouse.reward_type.name,
        "reward_range": [float(warehouse.reward_range[0]), float(warehouse.reward_range[1])],
        "max_steps": None if warehouse.max_steps is None else int(warehouse.max_steps),
        "max_inactivity_steps": warehouse.max_inactivity_steps,
        "reset_info": info,
        "reward_mechanism": reward_mechanism_text(warehouse),
        "termination": termination_text(warehouse),
    }
    space = env.observation_space
    observation_report = {
        "space": str(space),
        "space_type": type(space).__name__,
        "per_robot_space": str(space[0]),
        "dtype": str(observation[0].dtype),
        "shape_per_robot": list(observation[0].shape),
        "n_robots": len(observation),
        "box_low": _bound_summary(space[0].low),
        "box_high": _bound_summary(space[0].high),
        "shared_observation": False,
        "individual_positions_match": all(
            int(observation[index][0]) == robots[index]["x"]
            and int(observation[index][1]) == robots[index]["y"]
            for index in range(len(robots))
        ),
        "components": components,
        "representative": [
            {
                "robot": index + 1,
                "values": np.asarray(values, dtype=np.float32).round(6).tolist(),
            }
            for index, values in enumerate(observation)
        ],
        "empty_cell_direction": (
            "When a sensor cell has no robot, the environment writes the "
            f"one-hot for {Direction(0).name} in the direction fields. "
            "Those bits do not mean a robot is facing that way."
        ),
    }
    _require_empty_sensor_direction(observation, warehouse)
    return _json_ready(inspection), _json_ready(observation_report)


def analyse_actions(env):
    """Action metadata from the installed Action enum and action space."""
    rows = []
    descriptions = {
        "NOOP": "Stay in the current cell. Direction and load stay as they are.",
        "FORWARD": (
            "Request one step in the facing direction. Agent.req_location clamps "
            "the target inside the grid. The collision resolver can cancel the "
            "move and leave the robot where it is."
        ),
        "LEFT": (
            "Turn one quarter to the left along "
            "UP -> LEFT -> DOWN -> RIGHT. The cell does not change."
        ),
        "RIGHT": (
            "Turn one quarter to the right along "
            "UP -> RIGHT -> DOWN -> LEFT. The cell does not change."
        ),
        "TOGGLE_LOAD": (
            "Pick up the shelf in this cell when the robot is empty. "
            "Drop a carried shelf when the cell is not a highway. "
            "A carried shelf stays loaded when the cell is a highway."
        ),
    }
    for action in Action:
        if action.name not in descriptions:
            raise RuntimeError(f"no description is recorded for {action.name}")
        rows.append(
            {
                "action_id": int(action.value),
                "name": action.name,
                "effect": descriptions[action.name],
            }
        )
    space = env.action_space
    if not isinstance(space, gym.spaces.Tuple):
        raise TypeError(f"expected a Tuple action space, got {type(space).__name__}")
    for index, item in enumerate(space):
        if not isinstance(item, gym.spaces.Discrete):
            raise TypeError(
                f"robot {index + 1} action space is {type(item).__name__}, "
                "expected Discrete"
            )
        if int(item.n) != len(Action):
            raise RuntimeError(
                f"robot {index + 1} has Discrete({item.n}); Action has {len(Action)} values"
            )
    return {
        "space": str(space),
        "space_type": type(space).__name__,
        "n_robots": len(space),
        "actions_per_robot": int(space[0].n),
        "joint_actions": int(space[0].n) ** len(space),
        "individually_assigned": True,
        "actions": rows,
    }


def probe_actions(env):
    """Record a short scripted episode, including one rejected action id."""
    warehouse = env.unwrapped
    report = analyse_actions(env)
    examples = []

    env.reset(seed=SAMPLE_SEED)
    _record_step(env, _joint(warehouse, 0, Action.NOOP.value), "noop", examples)
    _record_step(env, _joint(warehouse, 0, Action.RIGHT.value), "right_turn", examples)
    _record_step(env, _joint(warehouse, 0, Action.LEFT.value), "left_turn", examples)

    boundary = _boundary_example(env, examples)
    pickup = _load_example(env, examples)
    invalid = _invalid_action_example(env)

    report["examples"] = examples
    report["boundary_example"] = boundary
    report["load_example"] = pickup
    report["invalid_action"] = invalid
    return _json_ready(report)


def run_random_baseline(env):
    """Play N_EPISODES with an independent Generator per episode seed."""
    warehouse = env.unwrapped
    agent_layer, shelf_layer = _LAYER_AGENTS, _LAYER_SHELFS
    episodes = []
    sample = None
    blocked = {key: 0 for key in _BLOCK_KEYS}
    obs_min = None
    obs_max = None
    all_integer = True
    reward_values = set()
    observed_count = 0
    safety = int(warehouse.max_steps or 0) + 5
    if safety < 5:
        raise RuntimeError("the warehouse did not expose max_steps")

    for episode_index in range(N_EPISODES):
        seed = BASE_SEED + episode_index
        observation, _info = env.reset(seed=seed)
        generator = np.random.default_rng(seed)
        robot_reward = np.zeros(warehouse.n_agents, dtype=np.float64)
        team_reward = 0.0
        positive_steps = 0
        length = 0
        terminated = False
        truncated = False
        snapshots = []
        paths = [[(int(agent.x), int(agent.y))] for agent in warehouse.agents]
        starts = [_robot_record(agent, warehouse) for agent in warehouse.agents]
        observed_count, obs_min, obs_max, all_integer = _absorb_observation(
            observation, observed_count, obs_min, obs_max, all_integer
        )
        shelf_cells = [(int(shelf.x), int(shelf.y)) for shelf in warehouse.shelfs]
        requests = [(int(shelf.x), int(shelf.y)) for shelf in warehouse.request_queue]
        if seed == SAMPLE_SEED:
            snapshots.append(capture_state(warehouse))

        for _step in range(safety):
            actions = tuple(
                int(generator.integers(0, space.n)) for space in env.action_space
            )
            before = [_robot_record(agent, warehouse) for agent in warehouse.agents]
            targets = []
            shelf_block = []
            for index, action in enumerate(actions):
                agent = warehouse.agents[index]
                if action == Action.FORWARD.value:
                    target = forward_target(agent, warehouse.grid_size)
                    targets.append(target)
                    shelf_block.append(
                        _carried_shelf_blocks(
                            warehouse, agent, target, agent_layer, shelf_layer
                        )
                    )
                else:
                    targets.append(None)
                    shelf_block.append(False)

            observation, rewards, terminated, truncated, _info = env.step(actions)
            rewards = [float(value) for value in rewards]
            if len(rewards) != warehouse.n_agents:
                raise RuntimeError(
                    f"expected {warehouse.n_agents} rewards, got {len(rewards)}"
                )
            reward_values.update(rewards)
            team = float(sum(rewards))
            team_reward += team
            robot_reward += np.asarray(rewards, dtype=np.float64)
            if team > 0:
                positive_steps += 1
            length += 1
            after = [_robot_record(agent, warehouse) for agent in warehouse.agents]
            _count_actions(blocked, actions, before, after, targets, shelf_block)
            for index, agent in enumerate(warehouse.agents):
                paths[index].append((int(agent.x), int(agent.y)))

            observed_count, obs_min, obs_max, all_integer = _absorb_observation(
                observation, observed_count, obs_min, obs_max, all_integer
            )
            if seed == SAMPLE_SEED:
                snapshots.append(capture_state(warehouse))

            if bool(terminated) or bool(truncated):
                break
        else:
            raise RuntimeError(
                f"episode seed {seed} did not terminate or truncate within {safety} steps"
            )

        record = {
            "episode": episode_index + 1,
            "seed": seed,
            "reward": team_reward,
            "length": length,
            "deliveries": int(round(team_reward)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "positive_reward_steps": positive_steps,
            "start_robots": starts,
            "start_shelves": shelf_cells,
            "start_requests": requests,
        }
        for index, value in enumerate(robot_reward.tolist()):
            record[f"robot_{index + 1}_reward"] = float(value)
        episodes.append(record)
        if seed == SAMPLE_SEED:
            sample = {
                "seed": seed,
                "snapshots": snapshots,
                "paths": paths,
                "highways": np.array(warehouse.highways, copy=True),
                "goals": [(int(x), int(y)) for x, y in warehouse.goals],
                "grid_rows": int(warehouse.grid_size[0]),
                "grid_columns": int(warehouse.grid_size[1]),
            }
        if (episode_index + 1) % 10 == 0:
            print(f"finished episode {episode_index + 1}/{N_EPISODES}", flush=True)

    # Min and max are reduced across robots, then across the running batch.
    ranges = {
        "min": np.min(obs_min, axis=0),
        "max": np.max(obs_max, axis=0),
        "all_integer": all_integer,
        "n_observations": observed_count,
    }
    return episodes, sample, blocked, ranges, reward_values


_BLOCK_KEYS = (
    "noop",
    "noop_changed",
    "left",
    "left_moved",
    "left_wrong_direction",
    "right",
    "right_moved",
    "right_wrong_direction",
    "forward_moved",
    "forward_boundary",
    "forward_blocked_collision",
    "forward_blocked_shelf",
    "forward_shelf_rule_but_moved",
    "toggle_pickup",
    "toggle_drop",
    "toggle_empty",
    "toggle_highway_kept",
    "toggle_other",
)


def summarise_baseline(episodes, blocked, ranges, reward_values):
    """Aggregate the random-policy episodes. Deliveries follow the reward sum."""
    rewards = np.asarray([episode["reward"] for episode in episodes], dtype=np.float64)
    lengths = np.asarray([episode["length"] for episode in episodes], dtype=np.float64)
    deliveries = np.asarray(
        [episode["deliveries"] for episode in episodes], dtype=np.float64
    )
    n_agents = len([key for key in episodes[0] if key.startswith("robot_")])
    per_robot = []
    for index in range(1, n_agents + 1):
        values = np.asarray(
            [episode[f"robot_{index}_reward"] for episode in episodes],
            dtype=np.float64,
        )
        per_robot.append(float(np.mean(values)))

    observed = {float(episode["reward"]) for episode in episodes}
    start_cells = []
    start_directions = []
    start_on_goal = 0
    start_on_highway = 0
    shelf_sets = []
    request_sets = []
    for episode in episodes:
        shelf_sets.append(tuple(sorted(tuple(cell) for cell in episode["start_shelves"])))
        request_sets.append(tuple(sorted(tuple(cell) for cell in episode["start_requests"])))
        for robot in episode["start_robots"]:
            start_cells.append((robot["x"], robot["y"]))
            start_directions.append(robot["direction"])
            if robot["on_goal"]:
                start_on_goal += 1
            if robot["on_highway"]:
                start_on_highway += 1

    direction_counts = {
        name: sum(1 for direction in start_directions if direction == name)
        for name in (item.name for item in Direction)
    }
    successes = int(np.sum(deliveries > 0))
    summary = {
        "episodes": len(episodes),
        "first_seed": int(episodes[0]["seed"]),
        "last_seed": int(episodes[-1]["seed"]),
        "mean_reward": float(np.mean(rewards)),
        "median_reward": float(np.median(rewards)),
        "std_reward": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "total_deliveries": int(np.sum(deliveries)),
        "mean_deliveries": float(np.mean(deliveries)),
        "episodes_with_delivery": successes,
        "delivery_success_rate": float(successes / len(episodes)),
        "positive_reward_steps": int(
            sum(episode["positive_reward_steps"] for episode in episodes)
        ),
        "total_steps": int(np.sum(lengths)),
        "robot_mean_reward": per_robot,
        "all_terminated": all(episode["terminated"] for episode in episodes),
        "any_truncated": any(episode["truncated"] for episode in episodes),
        "unique_start_cells": len(set(start_cells)),
        "start_samples": len(start_cells),
        "starts_on_highway": start_on_highway,
        "starts_on_goal": start_on_goal,
        "start_direction_counts": direction_counts,
        "unique_start_shelf_layouts": len(set(shelf_sets)),
        "unique_request_layouts": len(set(request_sets)),
        "observation_min": ranges["min"].round(6).tolist(),
        "observation_max": ranges["max"].round(6).tolist(),
        "observations_integer_valued": bool(ranges["all_integer"]),
        "n_observations": int(ranges["n_observations"]),
        "blocked": blocked,
        "observed_episode_rewards": sorted(observed),
        "observed_step_rewards": sorted(float(value) for value in reward_values),
        "reward_values_are_delivery_units": set(float(value) for value in reward_values) <= {0.0, 1.0},
    }
    summary["deliveries_measurable"] = summary["reward_values_are_delivery_units"]
    return _json_ready(summary)


def save_tables(inspection, observation, probe, episodes, summary, blocked):
    """Write the CSV and JSON outputs used by the report and the app."""
    baseline_rows = []
    for episode in episodes:
        row = {
            "episode": episode["episode"],
            "seed": episode["seed"],
            "reward": episode["reward"],
            "length": episode["length"],
            "deliveries": episode["deliveries"]
            if summary["reward_values_are_delivery_units"]
            else None,
            "terminated": episode["terminated"],
            "truncated": episode["truncated"],
            "positive_reward_steps": episode["positive_reward_steps"],
        }
        for key, value in episode.items():
            if key.startswith("robot_"):
                row[key] = value
        baseline_rows.append(row)
    pd.DataFrame(baseline_rows).to_csv(OUTPUT_DIR / "random_baseline.csv", index=False)

    stat_rows = [
        ("episodes", summary["episodes"]),
        ("first_seed", summary["first_seed"]),
        ("last_seed", summary["last_seed"]),
        ("mean_reward", summary["mean_reward"]),
        ("median_reward", summary["median_reward"]),
        ("std_reward_sample", summary["std_reward"]),
        ("min_reward", summary["min_reward"]),
        ("max_reward", summary["max_reward"]),
        ("mean_episode_length", summary["mean_length"]),
        ("median_episode_length", summary["median_length"]),
        ("min_episode_length", summary["min_length"]),
        ("max_episode_length", summary["max_length"]),
        ("total_deliveries", summary["total_deliveries"]),
        ("mean_deliveries", summary["mean_deliveries"]),
        ("episodes_with_delivery", summary["episodes_with_delivery"]),
        ("delivery_success_rate", summary["delivery_success_rate"]),
        ("positive_reward_steps", summary["positive_reward_steps"]),
        ("total_steps", summary["total_steps"]),
        ("deliveries_measurable", summary["reward_values_are_delivery_units"]),
    ]
    for index, value in enumerate(summary["robot_mean_reward"], start=1):
        stat_rows.append((f"mean_robot_{index}_reward", value))
    pd.DataFrame(stat_rows, columns=["metric", "value"]).to_csv(
        OUTPUT_DIR / "statistical_summary.csv", index=False
    )
    pd.DataFrame(probe["actions"]).to_csv(OUTPUT_DIR / "action_descriptions.csv", index=False)
    pd.DataFrame(observation["components"]).to_csv(
        OUTPUT_DIR / "observation_components.csv", index=False
    )

    representative = []
    for robot in observation["representative"]:
        for index, value in enumerate(robot["values"]):
            representative.append(
                {
                    "robot": robot["robot"],
                    "index": index,
                    "value": value,
                    "component": observation["components"][index]["component"],
                    "meaning": observation["components"][index]["meaning"],
                }
            )
    pd.DataFrame(representative).to_csv(
        OUTPUT_DIR / "representative_observations.csv", index=False
    )
    pd.DataFrame(
        [{"outcome": key, "count": int(value)} for key, value in blocked.items()]
    ).to_csv(OUTPUT_DIR / "blocked_movement_summary.csv", index=False)

    ranges = pd.DataFrame(
        {
            "index": np.arange(len(summary["observation_min"])),
            "observed_min": summary["observation_min"],
            "observed_max": summary["observation_max"],
        }
    )
    ranges.to_csv(OUTPUT_DIR / "observation_ranges.csv", index=False)
    write_json(OUTPUT_DIR / "environment_inspection.json", inspection)
    write_json(
        OUTPUT_DIR / "observation_action_analysis.json",
        {"observation": observation, "actions": probe},
    )


def save_figures(inspection, sample, episodes, summary):
    """Draw the warehouse, the sample episode, and the baseline charts."""
    _plot_layout(inspection, sample, OUTPUT_DIR / "warehouse_layout.png")
    _plot_paths(sample, OUTPUT_DIR / "sample_episode_paths.png")
    _plot_frames(sample, OUTPUT_DIR / "sample_episode_frames.png")
    _save_gif(sample, OUTPUT_DIR / "sample_episode.gif")
    frame = pd.DataFrame(episodes)
    _plot_reward_hist(frame, OUTPUT_DIR / "reward_distribution.png")
    _plot_reward_series(frame, OUTPUT_DIR / "episode_rewards.png")
    _plot_length_hist(frame, OUTPUT_DIR / "episode_length_distribution.png")
    _plot_deliveries(frame, summary, OUTPUT_DIR / "delivery_performance.png")


def build_report(inspection, observation, probe, summary, notes):
    """Markdown report. Every number comes from the objects saved beside it."""
    robots = ", ".join(
        f"robot {index + 1} at ({robot['x']}, {robot['y']}) facing {robot['direction']}"
        for index, robot in enumerate(inspection["seed0_robots"])
    )
    goals = ", ".join(f"({x}, {y})" for x, y in inspection["goals"])
    requests = ", ".join(f"({x}, {y})" for x, y in inspection["seed0_requests"])
    measurable = summary["reward_values_are_delivery_units"]
    delivery_line = (
        f"Each per-robot reward in the baseline was 0 or 1, so the team sum "
        f"equals deliveries. Total deliveries: {summary['total_deliveries']}. "
        f"Episodes with at least one delivery: {summary['episodes_with_delivery']} "
        f"of {summary['episodes']} "
        f"(success rate {summary['delivery_success_rate']:.2%}). "
        f"Mean deliveries per episode: {summary['mean_deliveries']:.4f}."
        if measurable
        else (
            "The step rewards were not only 0 and 1, so this report does not "
            "treat the reward sum as a delivery count. `info` from reset is empty, "
            "and the environment does not return a separate delivery counter."
        )
    )
    positive_rate = summary["positive_reward_steps"] / summary["total_steps"]
    action_table = "\n".join(
        f"| {row['action_id']} | {row['name']} | {row['effect']} |"
        for row in probe["actions"]
    )
    self_rows = [
        row for row in observation["components"] if row["group"] == "self"
    ]
    self_table = "\n".join(
        f"| {row['index']} | {row['component']} | {row['meaning']} |"
        for row in self_rows
    )
    blocked = summary["blocked"]
    notes_md = "\n\n".join(f"### {name}\n\n{text}" for name, text in notes.items())
    vector_preview = ", ".join(
        f"{value:.0f}" for value in observation["representative"][0]["values"][:8]
    )
    return f"""# Warehouse EDA: {inspection["env_id"]}

This report was produced by `eda/warehouse_eda.py` from the installed
Gymnasium environment. The reward function and the warehouse were not modified.

## Environment inspection

- Environment id: `{inspection["env_id"]}`.
- Wrapper returned by `gym.make`: `{inspection["wrapper"]}`.
- Registration kwargs: `{json.dumps(inspection["registration_kwargs"])}`.
- Grid: {inspection["grid_rows"]} rows by {inspection["grid_columns"]} columns.
  `grid_size` is `(rows, columns)`. Column height is {inspection["column_height"]}.
- Robots: {inspection["n_agents"]}.
- Shelves at reset: {inspection["n_shelves"]}.
- Storage cells, defined as cells with `highways == 0`: {len(inspection["storage_cells"])}.
  Shelf coordinates at the seed-{SAMPLE_SEED} reset match those storage cells:
  {inspection["shelves_match_storage"]}. Across the baseline, distinct shelf
  layouts at reset: {summary["unique_start_shelf_layouts"]}.
- Delivery goals `(x, y)`: {goals}. Every goal cell is a highway:
  {inspection["goals_are_highways"]}. Shelves on a goal at reset:
  {inspection["shelves_on_goals"]}.
- Seed {SAMPLE_SEED} robots: {robots}.
- Seed {SAMPLE_SEED} requested shelves: {requests}. Request queue size is
  {inspection["request_queue_size"]}. Distinct request layouts over the
  {summary["episodes"]} resets: {summary["unique_request_layouts"]}.
- Starting cells are resampled on every reset. The baseline recorded
  {summary["start_samples"]} robot starts on {summary["unique_start_cells"]}
  distinct cells. {summary["starts_on_highway"]} starts were on a highway and
  {summary["starts_on_goal"]} were on a goal. Direction counts at reset:
  {json.dumps(summary["start_direction_counts"])}.
- Sensor range: {inspection["sensor_range"]}. Message bits: {inspection["msg_bits"]}.
  Coordinates are normalised: {inspection["normalised_coordinates"]}.
  Flattened observations: {inspection["fast_obs"]}.

## Observation analysis

- Space: `{observation["space"]}`.
- Each robot receives its own vector. The reset observation is a tuple of
  length {observation["n_robots"]}, and robot `i` coordinates in that vector
  match that robot's `(x, y)`: {observation["individual_positions_match"]}.
  Robots do not receive one shared observation.
- Dtype: `{observation["dtype"]}`. Shape per robot: {observation["shape_per_robot"]}.
- Declared Box bounds: low {observation["box_low"]}, high {observation["box_high"]}.
  The space itself does not state the true numeric range.
- The vector is a flattened structured feature list: 8 self features, then a
  {inspection["sensor_range"] * 2 + 1} by {inspection["sensor_range"] * 2 + 1}
  sensor window. With message bits equal to {inspection["msg_bits"]}, each
  sensor cell contributes 7 values.
- Values written by the environment are integers stored as float32. Every
  baseline observation was integer-valued:
  {summary["observations_integer_valued"]}. Observed per-index minima and
  maxima are in `observation_ranges.csv` ({summary["n_observations"]} robot
  observations).
- {observation["empty_cell_direction"]}

Self features, verified by rebuilding the vector from the live agents and
matching it to `reset()`:

| Index | Component | Meaning |
| --- | --- | --- |
{self_table}

Seed-{SAMPLE_SEED} robot 1, first eight values: {vector_preview}. The full
vectors are in `representative_observations.csv`.

The observation is therefore a structured, discrete-valued float vector in a
continuous `Box`, one vector per robot. It is not a single shared image and
it is not a `Discrete` space.

## Action analysis

- Space: `{probe["space"]}`.
- One `Discrete({probe["actions_per_robot"]})` per robot. Actions are assigned
  individually. `step` requires one action id per robot.
- Joint combinations: {probe["joint_actions"]}.

| ID | Name | Effect in the installed environment |
| --- | --- | --- |
{action_table}

Invalid ids are not legal samples of the action space. The probe submitted
action id {probe["invalid_action"]["action_id"]} and the environment raised
`{probe["invalid_action"]["error"]}`: {probe["invalid_action"]["message"]}.

Blocked legal moves stay inside `step` and become a no-op. Over the random
baseline the forward outcomes were:

- Moved: {blocked["forward_moved"]}
- Stayed because the clamped target was the current cell: {blocked["forward_boundary"]}
- Stayed because a carried shelf faced a standing shelf: {blocked["forward_blocked_shelf"]}
- Stayed for another collision rejection: {blocked["forward_blocked_collision"]}
- Carried-shelf rule matched but the robot still moved: {blocked["forward_shelf_rule_but_moved"]}

Turns: left {blocked["left"]} (cell changed {blocked["left_moved"]}, unexpected
direction {blocked["left_wrong_direction"]}); right {blocked["right"]} (cell
changed {blocked["right_moved"]}, unexpected direction {blocked["right_wrong_direction"]}).
No-op count {blocked["noop"]}, of which the state changed {blocked["noop_changed"]}.

Load toggles: pickup {blocked["toggle_pickup"]}, drop on a non-highway cell
{blocked["toggle_drop"]}, toggle with no shelf {blocked["toggle_empty"]},
toggle while carrying on a highway {blocked["toggle_highway_kept"]}, other
{blocked["toggle_other"]}.

The scripted examples for seed {SAMPLE_SEED}, the boundary move, and the
load sequence are stored in `observation_action_analysis.json`.

## Reward analysis

- Reward type: `{inspection["reward_type"]}`.
- `reward_range` attribute: {inspection["reward_range"]}.
- {inspection["reward_mechanism"]}
- Reset `info`: `{json.dumps(inspection["reset_info"])}`.

The original reward was not modified. The baseline team reward is the sum of
the per-robot list returned by `step`.

- Mean episode reward: {summary["mean_reward"]:.4f}
- Median episode reward: {summary["median_reward"]:.4f}
- Sample standard deviation (ddof=1): {summary["std_reward"]:.4f}
- Minimum episode reward: {summary["min_reward"]:.4f}
- Maximum episode reward: {summary["max_reward"]:.4f}
- Mean per-robot return: {summary["robot_mean_reward"]}
- Steps with a positive team reward: {summary["positive_reward_steps"]} of
  {summary["total_steps"]} ({positive_rate:.4%})
- Distinct per-robot step rewards: {summary["observed_step_rewards"]}
- Distinct episode team rewards: {summary["observed_episode_rewards"]}

{delivery_line}

## Termination

{inspection["termination"]}

In the baseline, every episode had `terminated` True: {summary["all_terminated"]}.
Any episode had `truncated` True: {summary["any_truncated"]}. Episode length
ranged from {summary["min_length"]} to {summary["max_length"]}
(mean {summary["mean_length"]:.2f}).

## Random policy

{summary["episodes"]} episodes used seeds {summary["first_seed"]} through
{summary["last_seed"]}. `reset(seed)` used that seed. Actions were drawn
independently for each robot with `numpy.random.default_rng(seed)` from
`0 .. n_actions-1`. The same seed therefore repeats the same layout and the
same action sequence.

## Design consequences for SARSA, DQN, and PPO

{notes_md}

## Outputs

All files are in `results/eda/`.
"""


def algorithm_notes(inspection, observation, summary):
    """Design consequences that follow from the measured environment."""
    joint = len(Action) ** inspection["n_agents"]
    horizon = inspection["max_steps"]
    sensor = inspection["sensor_range"]
    window = 1 + 2 * sensor
    success = summary["delivery_success_rate"]
    positive = summary["positive_reward_steps"] / summary["total_steps"]
    measurable = summary["reward_values_are_delivery_units"]
    delivery_clause = (
        f"The random policy delivered on {success:.2%} of episodes "
        f"({summary['total_deliveries']} deliveries in {summary['episodes']} episodes)."
        if measurable
        else "The random-policy return is the measured learning signal; deliveries were not counted separately."
    )
    shared = (
        f"Each robot has its own {observation['shape_per_robot'][0]}-value observation "
        f"and its own {len(Action)} actions. A joint action is one of {joint} pairs. "
        f"The environment returns a list of individual rewards, not a shared scalar."
    )
    sparse = (
        f"A positive team reward occurred on {positive:.4%} of steps. "
        f"{delivery_clause} There is no negative reward on the "
        f"{inspection['reward_type']} path: rewards start at 0 and a delivery adds 1 "
        f"to the robot on that goal."
    )
    horizon_text = (
        f"Episodes run to max_steps={horizon}. A delivery does not end the episode. "
        f"The baseline finished with every episode terminated={summary['all_terminated']} "
        f"and any episode truncated={summary['any_truncated']}. "
        f"The last transition is marked terminated, so a standard bootstrap treats "
        f"it as a true ending even though the cause is the step limit."
    )
    partial = (
        f"The sensor covers a {window}x{window} window (sensor_range={sensor}). "
        f"Absolute x and y are in the vector, and goals are fixed at "
        f"{inspection['goals']}, but goal cells are not a separate observation channel. "
        f"Empty sensor cells still contain the {Direction(0).name} direction one-hot."
    )
    blocked = summary["blocked"]
    collisions = (
        f"Forward requests that did not change the cell: "
        f"{blocked['forward_boundary']} at the boundary, "
        f"{blocked['forward_blocked_shelf']} against a standing shelf while carrying, "
        f"and {blocked['forward_blocked_collision']} rejected by the collision resolver. "
        f"Those failures pay 0, the same as a successful step that does not deliver."
    )
    return {
        "SARSA": (
            f"{shared} Tabular SARSA can index the {len(Action)} actions, or the "
            f"{joint} joint actions, but it cannot use the raw "
            f"{observation['shape_per_robot'][0]}-value vector as a table key. "
            f"The grid is {inspection['grid_rows']}x{inspection['grid_columns']}, "
            f"and the shelves and the request queue change the sensor bits. "
            f"A practical SARSA state has to be a compact encoding or a function "
            f"approximator. {sparse} {horizon_text} {partial} {collisions} "
            f"Independent SARSA on each robot's own reward never credits the robot "
            f"that did not deliver. Summing the list matches the cooperative task "
            f"and gives both robots the same team target."
        ),
        "DQN": (
            f"A DQN head of {len(Action)} units matches one robot's action space. "
            f"The network can read the float32 vector directly; the declared Box "
            f"bounds are infinite, while the observed values are small integers "
            f"(integer-valued={summary['observations_integer_valued']}), so the "
            f"inputs should be scaled. {sparse} Replay from a random policy is "
            f"almost all zero targets, and a delivery is not a terminal success. "
            f"{horizon_text} {partial} Mask the sensor direction unless has_agent "
            f"is 1. {collisions} With {inspection['n_agents']} robots, independent "
            f"Q-learning on the individual reward and a joint network over "
            f"{joint} actions are different algorithms: only the joint network, "
            f"or a team reward, trains on the other robot's delivery."
        ),
        "PPO": (
            f"A categorical policy of {len(Action)} actions per robot matches the "
            f"Tuple action space, and the identical observation layout supports "
            f"shared actor weights. {shared} The trainer has to decide whether "
            f"the value target is one robot's reward or the sum. {sparse} "
            f"The value function therefore stays near the measured mean return "
            f"of {summary['mean_reward']:.4f} until a delivery appears, and the "
            f"advantage of that delivery is rare. A rollout has to cover the "
            f"{horizon}-step horizon because nothing else ends the episode. "
            f"{horizon_text} {partial} A critic that sees both observations has "
            f"information neither actor has alone. {collisions} Entropy is part "
            f"of the exploration the random baseline shows is otherwise unlikely "
            f"to deliver. Do not change the environment reward to create that signal."
        ),
    }


def reward_mechanism_text(warehouse):
    """Describe only the reward branch this environment was constructed with."""
    kind = warehouse.reward_type
    if kind is RewardType.INDIVIDUAL:
        return (
            "RewardType.INDIVIDUAL starts each step at 0 for every robot. "
            "When a requested shelf is on a goal, the robot standing on that "
            "goal receives +1 and the shelf is replaced in the request queue. "
            "The other robot's reward stays 0. This branch does not subtract."
        )
    if kind is RewardType.GLOBAL:
        return (
            "RewardType.GLOBAL starts each step at 0 and adds 1 to every robot "
            "when a requested shelf is on a goal."
        )
    if kind is RewardType.TWO_STAGE:
        return (
            "RewardType.TWO_STAGE adds 0.5 when a requested shelf reaches a goal "
            "and 0.5 when that shelf is later dropped off a highway."
        )
    raise RuntimeError(f"unrecognised reward type {kind!r}")


def termination_text(warehouse):
    """Describe the ending rule from the attributes on this instance."""
    inactivity = warehouse.max_inactivity_steps
    if inactivity is None:
        inactivity_text = (
            "max_inactivity_steps is None, so an idle stretch does not end the episode."
        )
    else:
        inactivity_text = (
            f"max_inactivity_steps is {int(inactivity)}. "
            "The episode also ends after that many steps without a delivery."
        )
    return (
        f"max_steps is {warehouse.max_steps}. {inactivity_text} "
        "When the limit is reached, step returns terminated True and truncated False. "
        "A delivery does not set the terminated flag."
    )


def encode_observation(warehouse, agent, agent_layer, shelf_layer):
    """Rebuild one flattened observation the way Warehouse._get_default_obs does."""
    del agent_layer  # Layer ids are checked by the sensor reader.
    parts = [
        float(agent.x),
        float(agent.y),
        float(agent.carrying_shelf is not None),
    ]
    direction = np.zeros(len(Direction), dtype=np.float32)
    direction[int(agent.dir.value)] = 1.0
    parts.extend(direction.tolist())
    parts.append(float(warehouse._is_highway(int(agent.x), int(agent.y))))

    agent_ids, shelf_ids = _sensor_ids(warehouse, agent, shelf_layer)
    for agent_id, shelf_id in zip(agent_ids, shelf_ids):
        if int(agent_id) == 0:
            parts.append(0.0)
            parts.extend([1.0, 0.0, 0.0, 0.0])
            parts.extend([0.0] * int(warehouse.msg_bits))
        else:
            parts.append(1.0)
            one_hot = np.zeros(len(Direction), dtype=np.float32)
            other = warehouse.agents[int(agent_id) - 1]
            one_hot[int(other.dir.value)] = 1.0
            parts.extend(one_hot.tolist())
            if int(warehouse.msg_bits) > 0:
                parts.extend(np.asarray(other.message, dtype=np.float32).tolist())
        if int(shelf_id) == 0:
            parts.extend([0.0, 0.0])
        else:
            shelf = warehouse.shelfs[int(shelf_id) - 1]
            parts.extend([1.0, float(shelf in warehouse.request_queue)])
    return np.asarray(parts, dtype=np.float32)


def component_rows(warehouse):
    """Name every observation index from the flattened feature layout."""
    rows = [
        (0, "self", "x", "Column of this robot. Not normalised in this environment."),
        (1, "self", "y", "Row of this robot. Not normalised in this environment."),
        (2, "self", "carrying_shelf", "1 when this robot is carrying a shelf, otherwise 0."),
    ]
    for direction in Direction:
        rows.append(
            (
                3 + int(direction.value),
                "self",
                f"direction_{direction.name.lower()}",
                f"1 when this robot faces {direction.name}.",
            )
        )
    rows.append(
        (
            7,
            "self",
            "on_highway",
            "1 when this robot's cell is a highway, including the delivery row.",
        )
    )
    width = 1 + 2 * int(warehouse.sensor_range)
    cell_size = 1 + len(Direction) + int(warehouse.msg_bits) + 2
    for cell in range(width * width):
        offset_x = (cell % width) - int(warehouse.sensor_range)
        offset_y = (cell // width) - int(warehouse.sensor_range)
        base = 8 + cell * cell_size
        where = f"sensor cell {cell} at offset ({offset_x}, {offset_y})"
        rows.append(
            (base, "sensor", f"cell_{cell}_has_agent", f"1 when a robot is in {where}.")
        )
        for direction in Direction:
            rows.append(
                (
                    base + 1 + int(direction.value),
                    "sensor",
                    f"cell_{cell}_direction_{direction.name.lower()}",
                    (
                        f"Direction one-hot in {where}. "
                        f"An empty cell is written as {Direction(0).name}."
                    ),
                )
            )
        bit = base + 1 + len(Direction)
        for message in range(int(warehouse.msg_bits)):
            rows.append(
                (
                    bit + message,
                    "sensor",
                    f"cell_{cell}_message_{message}",
                    f"Message bit {message} in {where}.",
                )
            )
        shelf_at = bit + int(warehouse.msg_bits)
        rows.append(
            (shelf_at, "sensor", f"cell_{cell}_has_shelf", f"1 when a shelf is in {where}.")
        )
        rows.append(
            (
                shelf_at + 1,
                "sensor",
                f"cell_{cell}_shelf_requested",
                f"1 when the shelf in {where} is in the request queue.",
            )
        )
    return [
        {"index": index, "group": group, "component": name, "meaning": meaning}
        for index, group, name, meaning in rows
    ]


def capture_state(warehouse):
    """Positions needed to draw one frame. Shelves include the request flag."""
    requested = set(warehouse.request_queue)
    return {
        "agents": [_robot_record(agent, warehouse) for agent in warehouse.agents],
        "shelves": [
            {
                "x": int(shelf.x),
                "y": int(shelf.y),
                "requested": shelf in requested,
            }
            for shelf in warehouse.shelfs
        ],
    }


def forward_target(agent, grid_size):
    """Target cell of FORWARD, using Agent.req_location, then restore the agent."""
    previous = agent.req_action
    try:
        agent.req_action = Action.FORWARD
        target = agent.req_location(grid_size)
    finally:
        agent.req_action = previous
    return int(target[0]), int(target[1])


def _carried_shelf_blocks(warehouse, agent, target, agent_layer, shelf_layer):
    """The early cancel in Warehouse.step for a loaded robot facing a shelf."""
    start = (int(agent.x), int(agent.y))
    if agent.carrying_shelf is None or start == target:
        return False
    shelf_id = int(warehouse.grid[shelf_layer, target[1], target[0]])
    if shelf_id == 0:
        return False
    other_id = int(warehouse.grid[agent_layer, target[1], target[0]])
    if other_id and warehouse.agents[other_id - 1].carrying_shelf is not None:
        return False
    return True


def _count_actions(blocked, actions, before, after, targets, shelf_block):
    for index, action in enumerate(actions):
        previous = before[index]
        current = after[index]
        moved = (previous["x"], previous["y"]) != (current["x"], current["y"])
        if action == Action.NOOP.value:
            blocked["noop"] += 1
            if moved or previous["direction"] != current["direction"] or previous["carrying"] != current["carrying"]:
                blocked["noop_changed"] += 1
        elif action == Action.LEFT.value:
            blocked["left"] += 1
            if moved:
                blocked["left_moved"] += 1
            if current["direction"] != _turned(previous["direction"], -1):
                blocked["left_wrong_direction"] += 1
        elif action == Action.RIGHT.value:
            blocked["right"] += 1
            if moved:
                blocked["right_moved"] += 1
            if current["direction"] != _turned(previous["direction"], 1):
                blocked["right_wrong_direction"] += 1
        elif action == Action.FORWARD.value:
            target = targets[index]
            if moved:
                blocked["forward_moved"] += 1
                if shelf_block[index]:
                    blocked["forward_shelf_rule_but_moved"] += 1
            elif target == (previous["x"], previous["y"]):
                blocked["forward_boundary"] += 1
            elif shelf_block[index]:
                blocked["forward_blocked_shelf"] += 1
            else:
                blocked["forward_blocked_collision"] += 1
        elif action == Action.TOGGLE_LOAD.value:
            _count_toggle(blocked, previous, current)
        else:
            raise RuntimeError(f"baseline sampled an unexpected action {action}")


def _count_toggle(blocked, previous, current):
    if (previous["x"], previous["y"]) != (current["x"], current["y"]):
        blocked["toggle_other"] += 1
        return
    if not previous["carrying"] and current["carrying"]:
        blocked["toggle_pickup"] += 1
    elif previous["carrying"] and not current["carrying"] and not previous["on_highway"]:
        blocked["toggle_drop"] += 1
    elif not previous["carrying"] and not current["carrying"]:
        blocked["toggle_empty"] += 1
    elif previous["carrying"] and current["carrying"] and previous["on_highway"]:
        blocked["toggle_highway_kept"] += 1
    else:
        blocked["toggle_other"] += 1


def _turned(direction_name, quarter_turns):
    current = Direction[direction_name]
    index = TURN_ORDER.index(current)
    return TURN_ORDER[(index + quarter_turns) % len(TURN_ORDER)].name


def _record_step(env, actions, label, examples):
    warehouse = env.unwrapped
    before = [_robot_record(agent, warehouse) for agent in warehouse.agents]
    _observation, rewards, terminated, truncated, info = env.step(actions)
    after = [_robot_record(agent, warehouse) for agent in warehouse.agents]
    examples.append(
        {
            "label": label,
            "actions": [int(action) for action in actions],
            "rewards": [float(value) for value in rewards],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
            "before": before,
            "after": after,
        }
    )
    return bool(terminated) or bool(truncated)


def _boundary_example(env, examples):
    """Turn one edge robot outward and step forward. Record the actual result."""
    warehouse = env.unwrapped
    env.reset(seed=SAMPLE_SEED)
    rows, columns = warehouse.grid_size
    chosen = None
    for index, agent in enumerate(warehouse.agents):
        direction = _outward_direction(int(agent.x), int(agent.y), int(rows), int(columns))
        if direction is not None:
            chosen = (index, direction)
            break
    if chosen is None:
        return {"found": False, "reason": "no robot started on the border at this seed"}
    index, direction = chosen
    for action in _turn_actions(warehouse.agents[index].dir, direction):
        if _record_step(env, _joint(warehouse, index, action), "boundary_turn", examples):
            return {"found": False, "reason": "the episode ended while turning"}
    before = _robot_record(warehouse.agents[index], warehouse)
    done = _record_step(
        env, _joint(warehouse, index, Action.FORWARD.value), "boundary_forward", examples
    )
    after = examples[-1]["after"][index]
    return {
        "found": True,
        "robot": index + 1,
        "before": before,
        "after": after,
        "stayed": (before["x"], before["y"]) == (after["x"], after["y"]),
        "episode_ended": done,
    }


def _load_example(env, examples):
    """Pick up a shelf, fail to drop it on a highway, then drop it in storage."""
    warehouse = env.unwrapped
    setup = _find_shelf_step(env)
    if setup is None:
        return {"found": False, "reason": "no adjacent shelf step was found"}
    seed, index, direction, destination = setup
    env.reset(seed=seed)
    for action in _turn_actions(warehouse.agents[index].dir, direction):
        if _record_step(env, _joint(warehouse, index, action), "load_turn", examples):
            return {"found": False, "reason": "the episode ended while turning toward a shelf"}
    if _record_step(env, _joint(warehouse, index, Action.FORWARD.value), "load_forward", examples):
        return {"found": False, "reason": "the episode ended on the way to a shelf"}
    arrived = examples[-1]["after"][index]
    if (arrived["x"], arrived["y"]) != destination:
        return {
            "found": False,
            "reason": "the forward step did not reach the shelf",
            "after": arrived,
            "destination": destination,
        }
    _record_step(env, _joint(warehouse, index, Action.TOGGLE_LOAD.value), "load_pickup", examples)
    if not examples[-1]["after"][index]["carrying"]:
        return {"found": False, "reason": "toggle did not pick up the shelf", "after": examples[-1]["after"][index]}

    origin = _step_back_to_highway(env, warehouse, index, examples)
    if origin is None:
        return {"found": False, "reason": "could not step back onto the highway while carrying"}
    _record_step(
        env, _joint(warehouse, index, Action.TOGGLE_LOAD.value), "load_toggle_on_highway", examples
    )
    kept = examples[-1]["after"][index]["carrying"]
    returned = _step_onto(env, warehouse, index, destination, examples, "load_return")
    if not returned:
        return {
            "found": True,
            "pickup": True,
            "kept_load_on_highway": kept,
            "dropped_in_storage": False,
            "seed": seed,
            "robot": index + 1,
        }
    _record_step(env, _joint(warehouse, index, Action.TOGGLE_LOAD.value), "load_drop", examples)
    return {
        "found": True,
        "seed": seed,
        "robot": index + 1,
        "destination": list(destination),
        "pickup": True,
        "kept_load_on_highway": kept,
        "dropped_in_storage": not examples[-1]["after"][index]["carrying"],
    }


def _find_shelf_step(env):
    warehouse = env.unwrapped
    for seed in range(BASE_SEED, BASE_SEED + 40):
        env.reset(seed=seed)
        for index, agent in enumerate(warehouse.agents):
            if agent.carrying_shelf is not None or not warehouse._is_highway(int(agent.x), int(agent.y)):
                continue
            for direction in Direction:
                target = _neighbor(int(agent.x), int(agent.y), direction, warehouse.grid_size)
                if target == (int(agent.x), int(agent.y)):
                    continue
                if warehouse._is_highway(target[0], target[1]):
                    continue
                if int(warehouse.grid[_LAYER_SHELFS, target[1], target[0]]) == 0:
                    continue
                other = [
                    (int(robot.x), int(robot.y))
                    for other_index, robot in enumerate(warehouse.agents)
                    if other_index != index
                ]
                if target in other:
                    continue
                return seed, index, direction, target
    return None


def _step_back_to_highway(env, warehouse, index, examples):
    agent = warehouse.agents[index]
    for direction in Direction:
        target = _neighbor(int(agent.x), int(agent.y), direction, warehouse.grid_size)
        if target == (int(agent.x), int(agent.y)) or not warehouse._is_highway(*target):
            continue
        for action in _turn_actions(agent.dir, direction):
            if _record_step(env, _joint(warehouse, index, action), "load_turn_back", examples):
                return None
        if _record_step(env, _joint(warehouse, index, Action.FORWARD.value), "load_to_highway", examples):
            return None
        arrived = examples[-1]["after"][index]
        if (arrived["x"], arrived["y"]) == target and arrived["on_highway"] and arrived["carrying"]:
            return target
    return None


def _step_onto(env, warehouse, index, destination, examples, label):
    agent = warehouse.agents[index]
    direction = _direction_between(
        (int(agent.x), int(agent.y)), destination
    )
    if direction is None:
        return False
    for action in _turn_actions(agent.dir, direction):
        if _record_step(env, _joint(warehouse, index, action), f"{label}_turn", examples):
            return False
    if _record_step(env, _joint(warehouse, index, Action.FORWARD.value), label, examples):
        return False
    arrived = examples[-1]["after"][index]
    return (arrived["x"], arrived["y"]) == destination


def _invalid_action_example(env):
    invalid_id = len(Action)
    try:
        env.step(_joint(env.unwrapped, 0, invalid_id))
    except Exception as exc:
        result = {
            "action_id": invalid_id,
            "error": type(exc).__name__,
            "message": str(exc),
        }
    else:
        result = {
            "action_id": invalid_id,
            "error": None,
            "message": "step accepted an action id outside the Action enum",
        }
    env.reset(seed=SAMPLE_SEED)
    return result


def _direction_between(start, destination):
    dx = destination[0] - start[0]
    dy = destination[1] - start[1]
    if (abs(dx), abs(dy)) != (1, 0) and (abs(dx), abs(dy)) != (0, 1):
        return None
    if dx == 1:
        return Direction.RIGHT
    if dx == -1:
        return Direction.LEFT
    if dy == 1:
        return Direction.DOWN
    return Direction.UP


def _neighbor(x, y, direction, grid_size):
    rows, columns = int(grid_size[0]), int(grid_size[1])
    if direction is Direction.UP:
        return x, max(0, y - 1)
    if direction is Direction.DOWN:
        return x, min(rows - 1, y + 1)
    if direction is Direction.LEFT:
        return max(0, x - 1), y
    if direction is Direction.RIGHT:
        return min(columns - 1, x + 1), y
    raise ValueError(f"unknown direction {direction}")


def _outward_direction(x, y, rows, columns):
    if y == 0:
        return Direction.UP
    if y == rows - 1:
        return Direction.DOWN
    if x == 0:
        return Direction.LEFT
    if x == columns - 1:
        return Direction.RIGHT
    return None


def _turn_actions(current, needed):
    right_turns = (TURN_ORDER.index(needed) - TURN_ORDER.index(current)) % len(TURN_ORDER)
    if right_turns == 0:
        return []
    if right_turns <= 2:
        return [Action.RIGHT.value] * right_turns
    return [Action.LEFT.value] * (len(TURN_ORDER) - right_turns)


def _joint(warehouse, index, action):
    actions = [Action.NOOP.value] * int(warehouse.n_agents)
    actions[index] = int(action)
    return tuple(actions)


def _robot_record(agent, warehouse):
    """Robot state, including highway and goal flags from the live grid."""
    x, y = int(agent.x), int(agent.y)
    goals = {(int(goal_x), int(goal_y)) for goal_x, goal_y in warehouse.goals}
    return {
        "x": x,
        "y": y,
        "direction": agent.dir.name,
        "carrying": agent.carrying_shelf is not None,
        "on_highway": bool(warehouse._is_highway(x, y)),
        "on_goal": (x, y) in goals,
    }


def _absorb_observation(observation, count, obs_min, obs_max, all_integer):
    """Update running observation bounds. Returns the new running state."""
    stacked = np.stack([np.asarray(obs, dtype=np.float32) for obs in observation])
    obs_min = stacked.copy() if obs_min is None else np.minimum(obs_min, stacked)
    obs_max = stacked.copy() if obs_max is None else np.maximum(obs_max, stacked)
    if not np.all(stacked == np.round(stacked)):
        all_integer = False
    return count + int(stacked.shape[0]), obs_min, obs_max, all_integer


def _require_reset_contract(env, observation, info):
    warehouse = env.unwrapped
    if not isinstance(observation, tuple) or len(observation) != warehouse.n_agents:
        raise RuntimeError("reset did not return one observation per robot")
    for index, values in enumerate(observation):
        array = np.asarray(values)
        expected = env.observation_space[index].shape
        if array.shape != expected:
            raise RuntimeError(
                f"robot {index + 1} observation shape {array.shape} != {expected}"
            )
    if info != {}:
        raise RuntimeError(f"reset info was {info!r}, expected an empty dict")


def _require_layers(warehouse):
    agent = warehouse.agents[0]
    shelf = warehouse.shelfs[0]
    if int(warehouse.grid[_LAYER_AGENTS, agent.y, agent.x]) != int(agent.id):
        raise RuntimeError("agent layer does not contain the robot id")
    if int(warehouse.grid[_LAYER_SHELFS, shelf.y, shelf.x]) != int(shelf.id):
        raise RuntimeError("shelf layer does not contain the shelf id")
    return _LAYER_AGENTS, _LAYER_SHELFS


def _require_empty_sensor_direction(observation, warehouse):
    """Confirm an empty sensor cell uses the Direction(0) one-hot."""
    width = 1 + 2 * int(warehouse.sensor_range)
    center = int(warehouse.sensor_range) * width + int(warehouse.sensor_range)
    cell_size = 1 + len(Direction) + int(warehouse.msg_bits) + 2
    found = False
    for values in observation:
        for cell in range(width * width):
            if cell == center:
                continue
            base = 8 + cell * cell_size
            if float(values[base]) == 0.0:
                direction = np.asarray(values[base + 1 : base + 1 + len(Direction)])
                expected = np.zeros(len(Direction), dtype=np.float32)
                expected[0] = 1.0
                if not np.array_equal(direction, expected):
                    raise RuntimeError(
                        f"empty sensor cell {cell} direction was {direction.tolist()}"
                    )
                found = True
                break
        if found:
            break
    if not found:
        raise RuntimeError("no empty sensor cell was available to check")


def _sensor_ids(warehouse, agent, shelf_layer):
    del shelf_layer
    sensor_range = int(warehouse.sensor_range)
    min_x = int(agent.x) - sensor_range
    max_x = int(agent.x) + sensor_range + 1
    min_y = int(agent.y) - sensor_range
    max_y = int(agent.y) + sensor_range + 1
    rows, columns = warehouse.grid_size
    if min_x < 0 or min_y < 0 or max_x > int(columns) or max_y > int(rows):
        agents = np.pad(warehouse.grid[_LAYER_AGENTS], sensor_range, mode="constant")
        shelves = np.pad(warehouse.grid[_LAYER_SHELFS], sensor_range, mode="constant")
        min_x += sensor_range
        max_x += sensor_range
        min_y += sensor_range
        max_y += sensor_range
    else:
        agents = warehouse.grid[_LAYER_AGENTS]
        shelves = warehouse.grid[_LAYER_SHELFS]
    return (
        agents[min_y:max_y, min_x:max_x].reshape(-1),
        shelves[min_y:max_y, min_x:max_x].reshape(-1),
    )


def _plot_layout(inspection, sample, path):
    image = draw_frame(
        sample["highways"],
        sample["goals"],
        sample["snapshots"][0],
    )
    _save_image_figure(
        image,
        path,
        f"{inspection['env_id']} at seed {SAMPLE_SEED}",
        "Column",
        "Row",
        sample["grid_columns"],
        sample["grid_rows"],
    )


def _plot_paths(sample, path):
    image = draw_frame(sample["highways"], sample["goals"], sample["snapshots"][0])
    figure, axis = plt.subplots(figsize=(8.2, 7.2))
    columns, rows = sample["grid_columns"], sample["grid_rows"]
    axis.imshow(image, origin="upper", extent=(0, columns, rows, 0), interpolation="nearest")
    for index, cells in enumerate(sample["paths"]):
        color = np.array(ROBOT_COLORS[index % len(ROBOT_COLORS)]) / 255
        xs = [cell[0] + 0.5 for cell in cells]
        ys = [cell[1] + 0.5 for cell in cells]
        axis.plot(xs, ys, color=color, linewidth=1.4, label=f"Robot {index + 1} path")
        axis.scatter(xs[0], ys[0], s=40, color=color, marker="o", zorder=3)
        axis.scatter(xs[-1], ys[-1], s=40, color=color, marker="s", zorder=3)
    axis.set_title(f"Random-policy paths, seed {sample['seed']}")
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _plot_frames(sample, path):
    indexes = _frame_indexes(len(sample["snapshots"]))
    figure, axes = plt.subplots(2, 2, figsize=(9, 8))
    for axis, index in zip(axes.ravel(), indexes):
        image = draw_frame(sample["highways"], sample["goals"], sample["snapshots"][index])
        axis.imshow(image, origin="upper", interpolation="nearest")
        axis.set_title(f"Step {index}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"Sample episode frames, seed {sample['seed']}")
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _save_gif(sample, path):
    indexes = list(range(0, len(sample["snapshots"]), 10))
    if indexes[-1] != len(sample["snapshots"]) - 1:
        indexes.append(len(sample["snapshots"]) - 1)
    frames = [
        Image.fromarray(draw_frame(sample["highways"], sample["goals"], sample["snapshots"][index]))
        for index in indexes
    ]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=180, loop=0)


def _frame_indexes(n_frames):
    if n_frames <= 4:
        return list(range(n_frames))
    raw = [0, n_frames // 3, (2 * n_frames) // 3, n_frames - 1]
    return list(dict.fromkeys(raw))


def _plot_reward_hist(frame, path):
    figure, axis = plt.subplots(figsize=(7, 4.2))
    values = frame["reward"].to_numpy()
    bins = max(int(values.max() - values.min()) + 1, 1)
    axis.hist(values, bins=bins, color="#1E88E5", edgecolor="white")
    axis.set_title("Episode reward distribution")
    axis.set_xlabel("Episode reward (sum of both robots)")
    axis.set_ylabel("Episodes")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_reward_series(frame, path):
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    axis.plot(frame["episode"], frame["reward"], color="#FF8C00", marker="o", markersize=3)
    axis.set_title("Episode reward across episodes")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Episode reward")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_length_hist(frame, path):
    figure, axis = plt.subplots(figsize=(7, 4.2))
    values = frame["length"].to_numpy()
    bins = max(int(values.max() - values.min()) + 1, 1)
    axis.hist(values, bins=bins, color="#008080", edgecolor="white")
    axis.set_title("Episode length distribution")
    axis.set_xlabel("Episode length (steps)")
    axis.set_ylabel("Episodes")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_deliveries(frame, summary, path):
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    if summary["reward_values_are_delivery_units"]:
        labels = ["No delivery", "At least one delivery"]
        successes = int((frame["deliveries"] > 0).sum())
        values = [len(frame) - successes, successes]
        axis.bar(labels, values, color=["#9E9E9E", "#008080"])
        axis.set_ylabel("Episodes")
        axis.set_title("Delivery performance of the random policy")
    else:
        axis.plot(frame["episode"], frame["reward"], color="#008080")
        axis.set_title("Episode reward (deliveries not measured separately)")
        axis.set_xlabel("Episode")
        axis.set_ylabel("Episode reward")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _save_image_figure(image, path, title, xlabel, ylabel, columns, rows):
    figure, axis = plt.subplots(figsize=(8.2, 7.2))
    axis.imshow(image, origin="upper", extent=(0, columns, rows, 0), interpolation="nearest")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _legend_handles():
    return [
        Patch(facecolor=np.array(AISLE) / 255, edgecolor="0.75", label="Highway"),
        Patch(facecolor=np.array(SHELF) / 255, label="Shelf"),
        Patch(facecolor=np.array(REQUESTED) / 255, label="Requested shelf"),
        Patch(facecolor=np.array(GOAL) / 255, label="Delivery goal"),
        Patch(facecolor=np.array(ROBOT_COLORS[0]) / 255, label="Robot 1"),
        Patch(facecolor=np.array(ROBOT_COLORS[1]) / 255, label="Robot 2"),
    ]


def draw_frame(highways, goals, state):
    """Raster of one warehouse state. Row 0 is the top of the image."""
    rows, columns = highways.shape
    image = np.full((rows * CELL, columns * CELL, 3), FLOOR, dtype=np.uint8)
    for y in range(rows):
        for x in range(columns):
            if int(highways[y, x]) == 1:
                _fill(image, x, y, AISLE, 1)
            image[y * CELL : y * CELL + 1, x * CELL : (x + 1) * CELL] = GRID
            image[y * CELL : (y + 1) * CELL, x * CELL : x * CELL + 1] = GRID
    for x, y in goals:
        _fill(image, x, y, GOAL, 4)
    for shelf in state["shelves"]:
        color = REQUESTED if shelf["requested"] else SHELF
        _fill(image, shelf["x"], shelf["y"], color, 7)
    for index, agent in enumerate(state["agents"]):
        color = ROBOT_COLORS[index % len(ROBOT_COLORS)]
        _fill(image, agent["x"], agent["y"], color, 8)
        _paint_facing(image, agent["x"], agent["y"], agent["direction"])
        if agent["carrying"]:
            _fill(image, agent["x"], agent["y"], CARRY, 15)
    return image


def _fill(image, x, y, color, pad):
    y0 = y * CELL + pad
    x0 = x * CELL + pad
    y1 = (y + 1) * CELL - pad
    x1 = (x + 1) * CELL - pad
    image[y0:y1, x0:x1] = color


def _paint_facing(image, x, y, direction):
    y0 = y * CELL + 7
    x0 = x * CELL + 7
    y1 = (y + 1) * CELL - 7
    x1 = (x + 1) * CELL - 7
    cx0 = x * CELL + CELL // 2 - 3
    cx1 = x * CELL + CELL // 2 + 3
    cy0 = y * CELL + CELL // 2 - 3
    cy1 = y * CELL + CELL // 2 + 3
    if direction == Direction.UP.name:
        image[y0 : y0 + 6, cx0:cx1] = FACING
    elif direction == Direction.DOWN.name:
        image[y1 - 6 : y1, cx0:cx1] = FACING
    elif direction == Direction.LEFT.name:
        image[cy0:cy1, x0 : x0 + 6] = FACING
    else:
        image[cy0:cy1, x1 - 6 : x1] = FACING


def _plain_kwargs(kwargs):
    plain = {}
    for key, value in kwargs.items():
        if isinstance(value, RewardType):
            plain[key] = value.name
        else:
            plain[key] = _json_ready(value)
    return plain


def _bound_summary(array):
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    low = float(np.min(values))
    high = float(np.max(values))
    return {"min": _finite_or_text(low), "max": _finite_or_text(high)}


def _finite_or_text(value):
    if np.isneginf(value):
        return "-inf"
    if np.isposinf(value):
        return "inf"
    return value


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return _finite_or_text(value)
    return value


def write_json(path, payload):
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
