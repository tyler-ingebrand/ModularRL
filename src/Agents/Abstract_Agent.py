from src.Hooks.Abstract_Hook import Abstract_Hook
from src.Hooks.Do_Nothing_Hook import Do_Nothing_Hook

# Agent interface
class Abstract_Agent:
    def __init__(self, hook : Abstract_Hook = Do_Nothing_Hook()):
        self.hook = hook
    # returns  an action for the given state.
    # Must also return action probabilties, None is ok if the alg does not use them.
    def act(self, state):
        pass

    # The main function to learn from data. At a high level, takes in a transition (SARS) and should update the function
    # ocassionally updates the policy, but always stores transition
    def learn(self, state, action, value, action_probabilities, reward, next_state, done, info):
        self.hook.observe(self, state, action, reward, done, info)
