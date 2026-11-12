import gymnasium as gym   # use `import gym` if you installed an older gym-based version
import rware

env = gym.make("rware-tiny-4ag-v2")   # small warehouse, 4 robots
# env = gym.make("rware-medium-8ag-hard-v2")
obs, info = env.reset()

for step in range(500):
    env.render()                          # opens the visual window
    actions = env.action_space.sample()   # random actions for all agents
    obs, reward, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        obs, info = env.reset()

env.close()

# import multiprocessing as mp
# import numpy as np
# try:
#     import gymnasium as gym
# except ImportError:
#     import gym
# import rware

# def run(name, steps=500):
#     env = gym.make(name)
#     env.reset()
#     for _ in range(steps):
#         env.render()
#         out = env.step(env.action_space.sample())
#         done = out[2] if len(out) == 4 else (np.all(out[2]) or np.all(out[3]))
#         if np.all(done):
#             env.reset()
#     env.close()

# if __name__ == "__main__":   # required on Windows
#     names = ["rware-tiny-2ag-v2", "rware-small-4ag-v2"]
#     procs = [mp.Process(target=run, args=(n,)) for n in names]
#     for p in procs: p.start()
#     for p in procs: p.join()