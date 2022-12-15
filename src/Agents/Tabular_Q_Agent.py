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
        # print("this worked, I have a q_function")
        # print(self.q_function) 

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
        # print("state is", state) result: [x, y]
        # exploartion 
        # max(min_exploration_prob, np.exp(-exploration_decreasing_decay*e))
        # print("state is", state)
        x_state = state[0]
        y_state = state[1]

        T = self.learning_params['T']
        boltzman_exponential = np.exp((self.q_function[x_state, y_state :]) * T)[0]
        pr_sum = np.sum(boltzman_exponential) 
        pr = boltzman_exponential / pr_sum # pr[a] is probability of taking action a.
        # print(" pr ", pr)
        # pr = pr[0]
        # print(np.isnan(pr).any())
        
        if np.isnan(pr).any():
            print(' BOLTZMANN CONSTANT TOO LARGE IN ACTION-SELECTION SOFTMAX.')
            temp = np.array(np.isnan(pr), dtype = float)
            pr = temp / np.sum(temp) 

        pr_select = np.zeros(len(self.actions) + 1)
        pr_select[0] = 0
        for i in range(len(self.actions)):
            pr_select[i+1] = pr_select[i] + pr[i]
            
        # print(" PR select", pr_select)
        randn = np.random.random_sample() 
        # print(" randn", randn)
        for a in self.actions:
            if randn >= pr_select[a] and randn <= pr_select[a+1]:
                a_selected = a
                break
        # a_selected = input("what move 0,1,2,3")
        # return int(a_selected), None
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
        # print("done is", done)
        self.hook.observe(state, action, reward, done, info, tag)

        alpha = self.learning_params['alpha']
        gamma = self.learning_params['gamma']
        current_x = state[0]
        current_y = state[1]
        next_x = next_state[0]
        next_y = next_state[1]

        # Bellman update of q-table
        # print(" pre bellman update")
        # print(" breaking it down ")
        # print(" current x ", current_x)
        # print(" current y ", current_y)
        # print(" actin ", action)
        # print(" self.q_function[current_x][current_y][action] ", self.q_function[current_x][current_y][action])
        
        rhs = (1 - alpha) * self.q_function[current_x][current_y][action] + alpha * (reward + gamma * np.amax(self.q_function[next_x][next_y]))
        # if reward != 0:
            # print( "NON ZERO REWARD ", reward)
        # print('rhs', rhs)
        self.q_function[next_x][next_y][action] = rhs
        
        # print("   ", self.q_function[next_x][next_y][action])
        # print(' post bellman update ')
        return None # WHAT SHOULD THIS BE DOING? does it need a return? 


    def plot(self):
        print("plotting q")
        print("rewards", self.hook.rewards)
        print("current_episode_reward", self.hook.current_episode_reward)
        print("number_steps", self.hook.number_steps)
        print("number_episodes", self.hook.number_episodes)

        self.hook.plot()
