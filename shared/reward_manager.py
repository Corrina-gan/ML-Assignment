"""Turn the environment's reward list into the team score. Do not change the reward.

step() returns one number per robot. Under RewardType.INDIVIDUAL the robot
that delivers receives 1 and the other receives 0. This module only sums
that list. It does not add a penalty or a bonus.
"""

from environment.config import DELIVERY_REWARD, N_AGENTS


def robot_rewards(rewards):
    """One float per robot, in robot order."""
    rewards = tuple(float(reward) for reward in rewards)
    if len(rewards) != N_AGENTS:
        raise ValueError(f"expected {N_AGENTS} rewards, got {len(rewards)}")
    return rewards


def team_reward(rewards):
    """Sum of the per-robot rewards. This is the shared step score.

    A delivery contributes 1. Do not average the two robots.
    """
    return float(sum(robot_rewards(rewards)))


def reward_is_delivery_unit(rewards):
    """True when every robot reward is 0 or the delivery value of 1."""
    return all(reward in (0.0, float(DELIVERY_REWARD)) for reward in robot_rewards(rewards))
