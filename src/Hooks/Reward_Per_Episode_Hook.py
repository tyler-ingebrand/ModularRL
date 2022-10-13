import matplotlib.pyplot as plt
from src.Hooks.Abstract_Hook import Abstract_Hook


# A hook which keeps track of reward per episode in training
# This is in training mode only, no evaluation is performed
# Also keeps track of number of steps and episodes observed
# Outputs a dict containing "episode_rewards", "n_steps", "n_episodes"
class Reward_Per_Episode_Hook(Abstract_Hook):
    def __init__(self):
        self.current_episode_reward = 0
        self.rewards = []
        self.number_steps = 0
        self.number_episodes = 0

    def observe(self, agent, obs, action, reward, done, info):
        self.current_episode_reward += reward
        self.number_steps += 1
        if done:
            self.number_episodes += 1
            self.rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0

    def get_output(self):
        return {"episode_rewards" : self.rewards,
                "n_steps" :         self.number_steps,
                "n_episodes":       self.number_episodes}

    def plot(self):
        plt.plot(self.rewards)
        plt.show()
