"""Constants for the shared warehouse. Do not change these in an algorithm file.

Every algorithm imports this configuration so the warehouse, the actions,
and the reported score stay the same. The numbers below match the installed
``rware-tiny-2ag-v2`` environment, with the shared episode length set to 1000.
"""

ENV_ID = "rware-tiny-2ag-v2"

N_AGENTS = 2
OBS_SIZE = 71
N_ACTIONS = 5
# Shared episode length. The registered warehouse defaults to 500 steps;
# make_warehouse passes this value so every algorithm ends at 1000.
# A delivery does not end the episode.
MAX_STEPS = 1000

# Same integers as rware.warehouse.Action.
NOOP = 0
FORWARD = 1
LEFT = 2
RIGHT = 3
TOGGLE_LOAD = 4
ACTION_NAMES = ("No-op", "Forward", "Left", "Right", "Toggle load")

# RewardType.INDIVIDUAL: the delivering robot gets 1, the other gets 0.
DELIVERY_REWARD = 1.0

# Discount used when an algorithm learns from these rewards. Not an environment attribute.
GAMMA = 0.99

# evaluate() plays these episodes. Report that result for every algorithm.
EVAL_EPISODES = 10
EVAL_SEEDS = tuple(range(EVAL_EPISODES))
