import numpy as np

from src.Agents.Abstract_Agent import Abstract_Agent
from src.Hooks.Abstract_Hook import Abstract_Hook
from src.Hooks.Do_Nothing_Hook import Do_Nothing_Hook

# Agent interface
class Tabular_Q_Agent(Abstract_Agent):

    def __init__(self, game_size, 
                 hook:Abstract_Hook = Do_Nothing_Hook()):

        self.hook = hook
        self.set_learning_params() # set the learning params for the q-learning
        # question: how do I know what the possible states are?
        self.max_x_pos = game_size
        self.max_y_pos = game_size
        self.actions= [0, 1, 2, 3] # TODO: HACK come back to
        self.q_function = np.zeros([self.max_x_pos, self.max_y_pos, len(self.actions)])

    # returns  an action for the given state.
    # Must also return extras, None is ok if the alg does not use them.
    
    def set_learning_params(self):
        '''
        sets hardcoded learning parameters for q-learning
        stores in dictionary, keys: 
            'alpha', 'gamma', 'T', 'initial_epsilon', 
            initial_epsilon', 'max_timesteps_per_task'
        '''
        learning_params = {}
        learning_params['alpha'] = .8 # learning rate
        learning_params['gamma'] = .9 # MDP discount factor
        learning_params['T'] = 50 # WHAT DOES THIS DO?????? TODO: check
        learning_params['initial_epsilon'] = .1 # set epsilon to zero to  turn off epsilon-greedy exploration
        # learning_params['max_timesteps_per_task'] = 1000 # DO I NEED THIS TODO: check 
        self.learning_params = learning_params 

    def act(self, state):
        '''
        I think this is currently epsilon greedy exploration. probably could do better? 
        return: action, agent extras 
        '''
        x_state = state[0]
        y_state = state[1]

        T = self.learning_params['T']
        boltzman_exponential = np.exp((self.q_function[x_state, y_state :]) * T)[0]
        pr_sum = np.sum(boltzman_exponential) 
        pr = boltzman_exponential / pr_sum # pr[a] is probability of taking action a.
        
        if np.isnan(pr).any():
            print(' BOLTZMANN CONSTANT TOO LARGE IN ACTION-SELECTION SOFTMAX.')
            temp = np.array(np.isnan(pr), dtype = float)
            pr = temp / np.sum(temp) 

        pr_select = np.zeros(len(self.actions) + 1)
        pr_select[0] = 0
        for i in range(len(self.actions)):
            pr_select[i+1] = pr_select[i] + pr[i]
            
        randn = np.random.random_sample() 

        for a in self.actions:
            if randn >= pr_select[a] and randn <= pr_select[a+1]:
                a_selected = a
                break
        return a_selected, None # np.random.randint(low=0, high=4), None # this is randomly generating steps
        
    # The main function to learn from data. At a high level, takes in a transition (SARS) and should update the function
    # ocassionally updates the policy, but always stores transition

    # TODO: check learn return arguments and update frequency also what are these inputs
    def learn(self, state, action, reward, next_state, done, info, extras, tag = "1"):
        
        ''' 
        What is the 'tag'
        what is extras 
        what is info
        what is done
        what is this "ocassionally updates the policy, but always stores transition"
        '''
        self.hook.observe(state, action, reward, done, info, tag)

        alpha = self.learning_params['alpha']
        gamma = self.learning_params['gamma']
        current_x = state[0]
        current_y = state[1]
        next_x = next_state[0]
        next_y = next_state[1]

    
        rhs = (1 - alpha) * self.q_function[current_x][current_y][action] + alpha * (reward + gamma * np.amax(self.q_function[next_x][next_y]))
        self.q_function[next_x][next_y][action] = rhs
        
        return None 


    def plot(self):
        print("plotting q")
        print("rewards", self.hook.rewards)
        print("current_episode_reward", self.hook.current_episode_reward)
        print("number_steps", self.hook.number_steps)
        print("number_episodes", self.hook.number_episodes)

        self.hook.plot()
