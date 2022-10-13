import gym
from stable_baselines3 import PPO
from src import *
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v1')
agent = PPOAgent(PPO("MlpPolicy", env),
                 hook = Reward_Per_Episode_Hook()
                 )
run(env, agent, 500_000, show_progress_frequency=100_000)

# see results
agent.hook.plot()
run(env, agent, 1_000, render=True, train=False)
