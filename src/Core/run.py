import gym

from src.Agents.Abstract_Agent import Abstract_Agent


def run(  env : gym.Env,
          agent : Abstract_Agent,
          steps : int,
          train : bool = True,
          render : bool = False,
          show_progress_frequency = 0
          ):
    assert env is not None, " Env must exists. Got None instead of a gym.Env object"
    assert agent is not None, "Agent must exists. Got None instead of a Agent object"
    assert steps > 0, "Must run for some number positive number of steps. Got {} steps".format(steps)
    assert show_progress_frequency >= 0, "Must display step count at some positive number of steps, or 0 for not at all. Got {}.".format(show_progress_frequency)


    obs = env.reset()

    for i in range(steps):
        action, value, log_action_probs = agent.act(obs)
        nobs, reward, done, info = env.step(action)

        if train:
            agent.learn(obs, action, value, log_action_probs, reward, nobs, done, info)
        if render:
            env.render()
        if show_progress_frequency and i % show_progress_frequency == 0:
            print("Step ", i)
        if done:
            nobs = env.reset()
        obs = nobs

    # wrap up
    env.close()
