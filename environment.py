import gymnasium as gym
import rware
env = gym.make("rware-tiny-2ag-v2")

# env.n_agents  # 2
# env.action_space  # Tuple(Discrete(5), Discrete(5))
# env.observation_space  # Tuple(Box(XX,), Box(XX,))

# obs = env.reset()  # a tuple of observations

# actions = env.action_space.sample()  # the action space can be sampled
# print(actions)  # (1, 0)
# n_obs, reward, done, info = env.step(actions)
# print(done)    # False
# print(reward)  # [0.0, 0.0]

# env.render()
# env.close()