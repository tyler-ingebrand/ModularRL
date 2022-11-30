from typing import Dict

from src.Agents.Abstract_Agent import Abstract_Agent
from src.Hooks.Abstract_Hook import Abstract_Hook
from src.Hooks.Do_Nothing_Hook import Do_Nothing_Hook

# Agent interface
class Multi_Agent(Abstract_Agent):

    def __init__(self, agents:Dict,
                 hook : Abstract_Hook = Do_Nothing_Hook() # The hook to observe this process
                 ):
        self.hook = hook
        self.agents = agents


    # returns  an action for the given state.
    # Must also return extras, None is ok if the alg does not use them.
    def act(self, state):
        actions = {}
        extras = {}
        for key in state:
            agent_action, agent_extras = self.agents[key].act(state[key])
            actions[key] = agent_action
            extras[key] = agent_extras
        return actions

    # The main function to learn from data. At a high level, takes in a transition (SARS) and should update the function
    # ocassionally updates the policy, but always stores transition
    def learn(self, state, action, reward, next_state, done, info, extras, tag = "1"):
        # update hook
        self.hook.observe(self, state, action, reward, done, info, tag)

        # Update all agents
        for agent in state:
            self.agents[agent].learn(state[agent], action[agent], reward[agent], next_state[agent], done[agent], info[agent], extras[agent])

    def plot(self):
        self.hook.plot()
        for a in self.agents:
            a.plot()

