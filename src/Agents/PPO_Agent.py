from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor, configure_logger
from .Abstract_Agent import Abstract_Agent
from src.Hooks.Abstract_Hook import *
from src.Hooks.Do_Nothing_Hook import *


class PPOAgent(Abstract_Agent):
    def __init__(self, PPOAlg:PPO, hook : Abstract_Hook = Do_Nothing_Hook()):
        super().__init__(hook)

        if not PPOAlg._custom_logger:
            PPOAlg._logger = configure_logger(PPOAlg.verbose, PPOAlg.tensorboard_log, "Test", 10000)
        self.alg = PPOAlg

    def act(self, state):
        state_on_device = obs_as_tensor(state.reshape(1, len(state)), self.alg.device)
        action, value, log_action_probs = self.alg.policy(state_on_device)
        return action.detach().cpu().numpy(), value.detach(), log_action_probs.detach()

    def learn(self, state, action, value, action_probabilities, reward, next_state, done, info):
        # Allow the standard learn method for an agent to happen, which is just to pass info into hook
        super().learn(state, action, value, action_probabilities, reward, next_state, done, info)

        # always add to rollout buffer
        self.alg.rollout_buffer.add(state.flatten(), action, reward, False, value, action_probabilities)

        # if it has been some number of steps, train policy.
        if self.alg.rollout_buffer.full:
            self.alg.rollout_buffer.compute_returns_and_advantage(last_values=value, dones=done)
            self.alg.train()
            self.alg.rollout_buffer.reset()
            # print(self.alg.policy.value_net.weight)
