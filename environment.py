import gymnasium as gym   # use `import gym` if you installed an older gym-based version
import rware

env = gym.make("rware-tiny-2ag-v2")   # small warehouse, 2 robots
# env = gym.make("rware-medium-8ag-hard-v2")
obs, info = env.reset()

for step in range(500):
    env.render()                          # opens the visual window
    actions = env.action_space.sample()   # random actions for all agents
    obs, reward, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        obs, info = env.reset()

env.close()

