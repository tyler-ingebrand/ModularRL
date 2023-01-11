from gymnasium.spaces import MultiDiscrete, Discrete

from src.Agents.Abstract_Agent import Abstract_Agent
from src.Hooks.Abstract_Hook import Abstract_Hook
from src.Hooks.Do_Nothing_Hook import Do_Nothing_Hook
import torch

# Create a replay buffer to store experiences. This assumes state and actions are discrete (ints)
class Replay_Buffer:
    def __init__(self, state_dims, action_dims, buffer_size=100_000):
        self.buffer_size = buffer_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        state_dim_len = len(state_dims)
        action_dim_len = len(action_dims)

        # Create tables
        self.states = torch.zeros((self.buffer_size, state_dim_len), device=self.device, dtype=torch.long) # int type
        self.actions = torch.zeros((self.buffer_size, action_dim_len), device=self.device, dtype=torch.long) # int type
        self.rewards = torch.zeros((self.buffer_size,), device=self.device)
        self.dones = torch.zeros((self.buffer_size,), device=self.device)
        self.next_states = torch.zeros((self.buffer_size, state_dim_len), device=self.device, dtype=torch.long) # int type

        # create counter of where we are. Need to know if we are full or not for sampling
        self.pos = 0
        self.full = False

    # add data to our buffer
    # wrap around if needed
    def add(self, state, action, reward, done, next_state):
        self.states[self.pos] = torch.tensor(state, dtype=torch.long, device=self.device)
        self.actions[self.pos] = torch.tensor(action, dtype=torch.long, device=self.device)
        self.rewards[self.pos] = torch.tensor(reward, device=self.device)
        self.dones[self.pos] = torch.tensor(done, device=self.device)
        self.next_states[self.pos] = torch.tensor(next_state, dtype=torch.long, device=self.device) if next_state is not None else torch.zeros_like(self.states[self.pos])
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0


    # sample a random batch of data from the dataset
    def sample(self, batch_size):
        batch_inds = torch.randint(low=0,
                                   high=self.pos if not self.full else self.buffer_size, # exclusive
                                   size=(batch_size,))
        data = (self.states[batch_inds],
                self.actions[batch_inds],
                self.rewards[batch_inds],
                self.dones[batch_inds],
                self.next_states[batch_inds])
        return data





# Agent interface
# this class provides a tabular Q based algorithm for solving discrete state/action space MDPs
class Tabular_Q_Agent(Abstract_Agent):

    # for now assume state_space is multi-discrete and action space is discrete
    def __init__(self,
                 state_space,
                 action_space,
                 table_learning_rate=0.8,
                 gamma=0.9,
                 buffer_size=100_000,
                 update_frequency = 1000,
                 batch_size=100,
                 reward_scale = 1.0,

                 hook:Abstract_Hook = Do_Nothing_Hook()):

        self.hook = hook

        # detirmine state dims. Must be a tuple
        if type(state_space) is MultiDiscrete:
            self.state_dims = tuple(state_space.nvec)
        elif type(state_space) is Discrete:
            self.state_dims=(state_space.n,)
        else:
            raise Exception("Unknown state space type. Expected MultiDiscrete or Discrete, got {}".format(type(state_space)))

        # detirmine action dims. Must be a tuple
        if type(action_space) is MultiDiscrete:
            self.action_dims = tuple(action_space.nvec)
        elif type(action_space) is Discrete:
            self.action_dims = (action_space.n,)
        else:
            raise Exception("Unknown action space type. Expected MultiDiscrete or Discrete, got {}".format(type(action_space)))

        # learning hyper parameters
        self.lr = table_learning_rate
        self.gamma = gamma
        self.update_frequency = update_frequency
        self.current_step = 0
        self.batch_size = batch_size
        self.reward_scale = reward_scale

        # store memories in table
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.buffer = Replay_Buffer(self.state_dims, self.action_dims, buffer_size=buffer_size)

        # self.set_learning_params() # set the learning params for the q-learning
        # question: how do I know what the possible states are?
        #self.max_x_pos = game_size
        #self.max_y_pos = game_size
        #self.actions= [0, 1, 2, 3] # TODO: HACK come back to
        #self.q_function = np.zeros([self.max_x_pos, self.max_y_pos, len(self.actions)])

        # q table and optimizer. Can use gradient descent, ADAM, etc
        self.q_function = torch.zeros(self.state_dims + self.action_dims, device=self.device, requires_grad=True)
        self.q_optimizer = torch.optim.Adam((self.q_function,))


        # whether or not to act stochasitcally
        self.stochastic = True
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
        #x_state = state[0]
        #y_state = state[1]

        # This is softmax
        # T scales how much we care about a small difference in values
        # equivalent to scaling reward
        # T = self.learning_params['T']
        # boltzman_exponential = np.exp((self.q_function[x_state, y_state :]) * T)[0]
        # pr_sum = np.sum(boltzman_exponential)
        # pr = boltzman_exponential / pr_sum # pr[a] is probability of taking action a.

        state = tuple(state)
        values = self.q_function[state]
        if self.stochastic:
            probabilities = torch.nn.Softmax(dim=0)(values) # assumes 1 dimensional action space, todo
            random_index =  torch.multinomial(probabilities, num_samples=1)
            return random_index.item(), None
        else:
            return torch.max(values, dim=0).indices.item(), None

        # if np.isnan(pr).any():
        #     print(' BOLTZMANN CONSTANT TOO LARGE IN ACTION-SELECTION SOFTMAX.')
        #     temp = np.array(np.isnan(pr), dtype = float)
        #     pr = temp / np.sum(temp)
        #
        # pr_select = np.zeros(len(self.actions) + 1)
        # pr_select[0] = 0
        # for i in range(len(self.actions)):
        #     pr_select[i+1] = pr_select[i] + pr[i]
        #
        # randn = np.random.random_sample()
        #
        # for a in self.actions:
        #     if randn >= pr_select[a] and randn <= pr_select[a+1]:
        #         a_selected = a
        #         break

        # note item converts it from a tensor containing an int to an int
        # np.random.randint(low=0, high=4), None # this is randomly generating steps
        
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
        # allow hook to observe learning, mostly to save reward
        # note the agent needs to view the agent maybe, which is self
        self.hook.observe(self, state, action, reward, done, info, tag)

        # Save transition to buffer for future use
        self.buffer.add(state, action, reward, done, next_state)

        # increment step counter, every <update_frequency> steps, also update the table
        self.current_step += 1
        if self.current_step % self.update_frequency == 0:
            self.update_table()




    def update_table(self):
        # fetch data from table
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)

        # delete old gradients. Necessary for torch syntax.
        self.q_optimizer.zero_grad()

        # alpha = self.learning_params['alpha']
        # gamma = self.learning_params['gamma']
        # current_x = state[0]
        # current_y = state[1]
        # next_x = next_state[0]
        # next_y = next_state[1]
        #
        #
        # rhs = (1 - alpha) * self.q_function[current_x][current_y][action] + alpha * (reward + gamma * np.amax(self.q_function[next_x][next_y]))
        # self.q_function[next_x][next_y][action] = rhs

        # compute q value according to reward + value at next state

        next_state_values = multi_dimensional_index_select(self.q_function, next_states)
        target_q_values = rewards + (1-dones) * self.gamma * torch.max(next_state_values, dim=1).values

        # current estimate for q value
        dims = torch.concat((states, actions), dim=1)
        current_q_values = multi_dimensional_index_select(self.q_function, dims)

        # loss is MSE of the difference
        loss = torch.nn.MSELoss()(current_q_values, target_q_values)

        # compute gradients and update table
        loss.backward()
        self.q_optimizer.step()


    def plot(self):
        # print("plotting q")
        # print("rewards", self.hook.rewards)
        # print("current_episode_reward", self.hook.current_episode_reward)
        # print("number_steps", self.hook.number_steps)
        # print("number_episodes", self.hook.number_episodes)
        self.hook.plot()

def multi_dimensional_index_select(source, indices):
    output_dims_to_keep = len(source.shape) - indices.shape[1]
    lengths = source.shape[len(source.shape)-output_dims_to_keep:]
    ret = torch.empty((indices.shape[0],) + lengths, device=source.device)

    for batch in range(indices.shape[0]):
        out = source[tuple(indices[batch])]
        ret[batch] = out
    return ret




    # number_dims = indices.shape[1]
    # values = source
    # for i in range(number_dims):
    #     values = torch.index_select(values, i, indices[:, i])
    #     if i > 0:
    #         values = values[0]
    # return values
