## CartPole_v2 Environment in Gym

CartPole_v2 is a modified version of the CartPole-v1 environment in Gym. In this environment, a pole is attached to a cart that moves along a track. The goal is to balance the pole by moving the cart left, right, or not at all (i.e., "do nothing"). The episode ends when the pole falls or the cart moves outside the track boundaries. The state of the environment consists of the cart position, cart velocity, pole angle, and pole velocity. The action space consists of three actions, which are moving the cart left, moving the cart right, and staying in place.

## Adding a New Action

To implement the CartPole-v2 environment, we create a new class `CartPole_V2` that inherits from the CartPoleEnv class and overrides the `__init__` and `step` functions to include the "stay" action. 

## Deep Q-Learning Agent

To implement a DQL agent that learns to play CartPole-v2, we create a class `DQL_agent` that uses a deep Q-learning network to estimate the Q-values of the actions in each state and select the best action according to an epsilon-greedy policy. 

## Simu class

The `Simu` class takes two arguments: the mode of rendering to use (render_mode), and the type of agent to use (agent). In the initialization, it sets the rendering type. If the agent type is DQL, it also initializes a deep Q-learning agent (DQLAgent) with the state and action space sizes of the environment and trains it.

In the `run_simu` method, the state variable is initialized to the initial state of the environment, which is returned by the reset method. The score variable is set to 0 to keep track of the total reward of the episode.

The while loop runs until the episode is done. In each iteration of the loop, the DQL agent selects an action using the `step` method. The environment is then stepped forward using the selected action, the state variable is updated to the next state, and the score variable is increased by the observed reward.

After the episode ends, the score and the actions are printed to the console.


