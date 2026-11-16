"""One episode of the shared warehouse, stored before the environment moves on."""

from dataclasses import dataclass

import numpy as np

from environment.config import MAX_STEPS, N_ACTIONS, N_AGENTS, OBS_SIZE
from shared.reward_manager import robot_rewards, team_reward


def as_actions(actions):
    """Return one legal action int per robot."""
    actions = tuple(int(action) for action in actions)
    if len(actions) != N_AGENTS:
        raise ValueError(f"expected {N_AGENTS} actions, got {len(actions)}")
    for action in actions:
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action {action} is outside 0..{N_ACTIONS - 1}")
    return actions


def copy_observations(observations):
    """A detached copy of each robot's observation."""
    copied = tuple(np.array(obs, dtype=np.float32, copy=True) for obs in observations)
    if len(copied) != N_AGENTS or any(obs.shape != (OBS_SIZE,) for obs in copied):
        shapes = tuple(obs.shape for obs in copied)
        raise ValueError(f"expected {N_AGENTS} observations of shape {(OBS_SIZE,)}, got {shapes}")
    return copied


@dataclass(frozen=True)
class Transition:
    """One shared time step."""

    observations: tuple
    actions: tuple
    rewards: tuple
    next_observations: tuple
    terminated: bool
    truncated: bool

    @property
    def team_reward(self):
        return team_reward(self.rewards)

    @property
    def done(self):
        return self.terminated or self.truncated


@dataclass(frozen=True)
class Episode:
    """One episode from the shared warehouse."""

    transitions: tuple
    seed: int | None

    @property
    def length(self):
        return len(self.transitions)

    @property
    def team_return(self):
        """Undiscounted sum of both robots' rewards."""
        return float(sum(step.team_reward for step in self.transitions))

    @property
    def robot_returns(self):
        """Undiscounted return for robot 1, then robot 2."""
        totals = [0.0] * N_AGENTS
        for step in self.transitions:
            for index, reward in enumerate(step.rewards):
                totals[index] += reward
        return tuple(totals)

    @property
    def deliveries(self):
        """Team return rounded to an int. Valid when each reward is 0 or 1."""
        return int(round(self.team_return))


def run_episode(env, choose_actions, seed=None, on_episode_start=None):
    """Play one episode and record every shared transition.

    choose_actions(observations) returns one int per robot.
    on_episode_start(observations) runs after reset, before the first action.
    The episode stops when the environment sets terminated or truncated,
    and also stops at MAX_STEPS if the environment does not.
    """
    observations, _info = env.reset(seed=seed)
    observations = copy_observations(observations)
    if on_episode_start is not None:
        on_episode_start(observations)

    transitions = []
    for _step in range(MAX_STEPS):
        actions = as_actions(choose_actions(observations))
        next_observations, rewards, terminated, truncated, _info = env.step(actions)
        next_observations = copy_observations(next_observations)
        rewards = robot_rewards(rewards)
        terminated = bool(terminated)
        truncated = bool(truncated)
        transitions.append(
            Transition(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                terminated=terminated,
                truncated=truncated,
            )
        )
        observations = next_observations
        if terminated or truncated:
            break

    return Episode(transitions=tuple(transitions), seed=None if seed is None else int(seed))
