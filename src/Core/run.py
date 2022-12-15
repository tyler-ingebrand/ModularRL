import gym
import pettingzoo.utils.env
from tqdm import trange # progress bar
from src.Agents.Abstract_Agent import Abstract_Agent
import numpy

# slightly different loops depending on the type of environment
def run(  env,
          agent : Abstract_Agent,
          steps : int,
          train : bool = True,
          render : bool = False,
          show_progress : bool = True
          ):
    if isinstance(env, gym.Env):
        gym_run(env, agent, steps, train, render, show_progress)
    elif isinstance(env, pettingzoo.utils.env.ParallelEnv):
        multi_agent_run(env, agent, steps, train, render, show_progress)
    else:
        raise Exception("Unknown environment type: {}".format(type(env)))


# Runs a classic gym environment
def gym_run(  env : gym.Env,
          agent : Abstract_Agent,
          steps : int,
          train : bool = True,
          render : bool = False,
          show_progress : bool = True
          ):
    assert env is not None, " Env must exists. Got None instead of a gym.Env object"
    assert agent is not None, "Agent must exists. Got None instead of a Agent object"
    assert steps > 0, "Must run for some number positive number of steps. Got {} steps".format(steps)

    # This iterable shows progress, or is a normal range depending
    r = range(steps) if not show_progress else trange(steps)

    obs = env.reset()
    for i in r:
        action, extras = agent.act(obs)
        nobs, reward, done, info = env.step(action)

        if train:
            agent.learn(obs, action, reward, nobs, done, info, extras)
        if render:
            env.render()

        # handle reset. Env may be a vector or a single env.
        # If single, done = bool. If vector, done = numpy array
        if type(done) is bool and done:
            nobs = env.reset()
        elif type(done) is numpy.ndarray:
            for env_index, env_done in enumerate(done):
                if env_done:
                    nobs[env_index] = env.envs[env_index].reset()

        obs = nobs

    # wrap up
    env.close()


# Multi agent environment run loop
def multi_agent_run(  env : pettingzoo.utils.env.ParallelEnv,
          agent : Abstract_Agent,
          steps : int,
          train : bool = True,
          render : bool = False,
          show_progress : bool = True
          ):
    assert env is not None, " Env must exists. Got None instead of a gym.Env object"
    assert agent is not None, "Agent must exists. Got None instead of a Agent object"
    assert steps > 0, "Must run for some number positive number of steps. Got {} steps".format(steps)
    
    # This iterable shows progress, or is a normal range depending
    r = range(steps) if not show_progress else trange(steps)
   
    obs = env.reset()
    for i in r:
        action, extras = agent.act(obs)

        nobs, reward, terminations, truncations, info = env.step(action)
        if train:
            agent.learn(obs, action, reward, nobs, terminations, info, extras)
        if render:
            env.render()

        # handle reset. Env may be a vector or a single env.
        # If single, done = bool. If vector, done = numpy array
        if all(terminations.values()) or all(truncations.values()):
            nobs = env.reset()
        obs = nobs

    # wrap up
    env.close()