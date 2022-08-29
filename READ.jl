### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 820d54e9-ebf4-4d90-9b7f-e2065f8f728d
md"""
# Modular RL

## Introduction

This project seeks to solve [Markov Decision Processes (MDP)](https://en.wikipedia.org/wiki/Markov_decision_process) in a modular way. To do so, this project will offer one framework to support classical RL, classical control, multi-agent RL, hierarchical RL, and compositional RL. Doing so allows the end user to 1) understand RL in a modular way and 2) solve complex MDPs that would normally be unsolvable with classical RL approaches.

Please read the following descriptions of each type of RL and how it fits into the framework.

"""

# ╔═╡ ec42f40a-3013-4438-8790-d0e31094c298
md"""
## Classical RL

For the purposes of this project, I am defining classical RL to be typical end-to-end approaches to solving MDPs. These approaches work well for many simple problems but eventually break down as problem complexity increases. Example algorithms are [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), [DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html), and [PPO](https://arxiv.org/abs/1707.06347). Explaining the details of these algorithms is beyond the scope of this document, but generally speaking they are algorithms designed to solve MDPs via trial and error learning approaches. This means the algorithm tries different actions and uses the results to improve the policy. The policy is just a function which returns an action based on the state of the MDP.

Most if not all classical RL algorithms have the following structure. They consists of only 2 functions: 
* An act function, which takes in the state and returns the action. 
* An update function, which takes in an experience of interacting with the MDP and updates the policy. Typically, this function call records the experience into a buffer, but only updates the policy every so many steps.

A classical RL algorithm can be represented by the diagram below. The flow from State > Policy > Action represents how states are fed into the policy and an action is returned. The dotted line from Experience > Policy represents how an experience is used to update the policy via whatever algorithm is being used. 


![Internet required to see images](https://github.com/tyler-ingebrand/ModularRL/blob/master/docs/images/Classical%20RL%20Diagram.jpg?raw=true")

I am going to call an object with the above 2 functions (act, update) an Agent. As we will see, an Agent object can represent many different types of MDP-solving actions from all types of RL and control.

"""

# ╔═╡ ce420d87-4c25-49f1-a08c-e9066c0a6c10
md"""
## Classical Control

This framework also supports classical control. In control, an optimal policy is calculated based on the transition dynamics of a given problem. In the case with a known dynamics function, the control function takes the place of the policy, where it uses the state to calculate an action in some way. Since the transition dynamics are known, the update function does nothing. If the transition dynamics are unknown, the update function may update a model of the transition. As a result, the diagram for a control agent looks almost identical to a classical RL agent.

![Internet required to see images](https://github.com/tyler-ingebrand/ModularRL/blob/master/docs/images/Classical%20RL%20Diagram.jpg?raw=true")




"""

# ╔═╡ 742af376-5985-4dc4-a558-bb187b0e6e5a
md"""
# Explainable AI
"""

# ╔═╡ 94841fd3-c37a-4f85-90f8-aa44ab371bc8
md"""

![Internet required to see images](https://github.com/tyler-ingebrand/ModularRL/blob/master/docs/images/Cat.jpeg?raw=true")

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[deps]
"""

# ╔═╡ Cell order:
# ╟─820d54e9-ebf4-4d90-9b7f-e2065f8f728d
# ╟─ec42f40a-3013-4438-8790-d0e31094c298
# ╠═ce420d87-4c25-49f1-a08c-e9066c0a6c10
# ╠═742af376-5985-4dc4-a558-bb187b0e6e5a
# ╠═94841fd3-c37a-4f85-90f8-aa44ab371bc8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
