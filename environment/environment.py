"""Create the shared warehouse. Do not change the installed reward or transitions."""

import gymnasium as gym
import rware

from environment.config import (
    ENV_ID,
    MAX_STEPS,
    N_ACTIONS,
    N_AGENTS,
    NOOP,
    OBS_SIZE,
)

# Importing rware registers rware-tiny-2ag-v2.
del rware


def make_warehouse(render_mode=None):
    """Return a new copy of the shared warehouse.

    reset() returns a tuple of 2 observations, each shape (71,) and
    dtype float32, plus an empty info dict. step() takes one action per
    robot. Each action is an int from 0 to 4: no-op, forward, left,
    right, toggle load. step() returns the next observations, a list of
    one reward per robot, terminated, truncated, and info.

    A delivery pays 1 to the robot standing on that goal with the
    requested shelf. The other robot's reward stays 0. The episode ends
    after 1000 steps with terminated True and truncated False.
    """
    if render_mode is None:
        return gym.make(ENV_ID, max_steps=MAX_STEPS)
    return gym.make(ENV_ID, max_steps=MAX_STEPS, render_mode=render_mode)


def check_contract():
    """Raise if this configuration no longer matches the installed warehouse."""
    env = make_warehouse()
    try:
        warehouse = env.unwrapped
        if warehouse.n_agents != N_AGENTS:
            raise AssertionError(f"n_agents is {warehouse.n_agents}, expected {N_AGENTS}")
        if warehouse.max_steps != MAX_STEPS:
            raise AssertionError(f"max_steps is {warehouse.max_steps}, expected {MAX_STEPS}")
        if len(env.action_space) != N_AGENTS:
            raise AssertionError("action space does not have one Discrete per robot")
        for space in env.action_space:
            if space.n != N_ACTIONS:
                raise AssertionError(f"action space is Discrete({space.n}), expected {N_ACTIONS}")
        observations, info = env.reset(seed=0)
        if info != {}:
            raise AssertionError(f"reset info is {info}, expected an empty dict")
        if len(observations) != N_AGENTS:
            raise AssertionError(f"reset returned {len(observations)} observations")
        for observation in observations:
            if tuple(observation.shape) != (OBS_SIZE,):
                raise AssertionError(f"observation shape is {observation.shape}")
        _observations, rewards, terminated, truncated, _info = env.step((NOOP, NOOP))
        if len(rewards) != N_AGENTS:
            raise AssertionError(f"reward length is {len(rewards)}")
        if terminated or truncated:
            raise AssertionError("the first step ended the episode")
    finally:
        env.close()
    return True
