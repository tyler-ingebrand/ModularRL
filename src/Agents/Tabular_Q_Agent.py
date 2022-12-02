import numpy as np

from src.Agents.Abstract_Agent import Abstract_Agent
from src.Hooks.Abstract_Hook import Abstract_Hook
from src.Hooks.Do_Nothing_Hook import Do_Nothing_Hook

# Agent interface
class Tabular_Q_Agent(Abstract_Agent):

    def __init__(self,
                 hook:Abstract_Hook = Do_Nothing_Hook()):
        self.hook = hook
        # TODO


    # returns  an action for the given state.
    # Must also return extras, None is ok if the alg does not use them.
    def act(self, state):
        return np.random.randint(low=0, high=4), None
        # raise Exception("Unimplemented") # TODO

    # The main function to learn from data. At a high level, takes in a transition (SARS) and should update the function
    # ocassionally updates the policy, but always stores transition
    def learn(self, state, action, reward, next_state, done, info, extras, tag = "1"):
        return None
        # raise Exception("Unimplemented")# TODO

    def plot(self):
        self.hook.plot()
