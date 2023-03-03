import gym
import numpy as np
import torch
import math
import time
import random
import copy
from typing import Optional
from gym import logger 
from gym.utils.env_checker import check_env
from gym.envs.classic_control import CartPoleEnv
from collections import deque




class Random_agent:
    def __init__(self, env):
        self.env = env
        self.env.reset()
        pass

    def policy(self) :
        '''
        Define the policy of the agent, as a random agent it selects a random action given a uniform probability distribution
        Ouptut : an int corresponding to the selected action : 0 left , 1 right, 2 do nothing
	    '''
        return self.env.action_space.sample()
    

class DQL_agent :
    def __init__(self, env, hidden_dim=64, lr=0.05):
        self.env = env
        self.env.reset()
        # parameters and hyperparameters
        self.state_size = env.observation_space.shape[0] # this is for input of neural network node size
        self.action_size = env.action_space.n # this is for out of neural network node size
        
        self.loss_fn = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
                            torch.nn.Linear(self.state_size, hidden_dim),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_dim, hidden_dim*2),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_dim*2,self.action_size)
                    )
        self.target = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

        self.train(epochs = 5000, mem_size = 1000 , batch_size = 200, sync_freq = 500)



    
    def update(self, state, y):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(np.array(state)))
        loss = self.loss_fn(y_pred, (torch.Tensor(np.array(y))))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))
        
    def target_predict(self, s):
        ''' Use target network to make predicitons.'''
        with torch.no_grad(): 
            return self.target(torch.Tensor(s))
        
    def target_update(self):
        ''' Update target network with the model weights.'''
        self.target.load_state_dict(self.model.state_dict())
        
    def replay(self, memory, size, gamma=1.0):
        ''' Add experience replay to the DQL network class.'''
        if len(memory) > size:
            # Sample experiences from the agent's memory
            data = random.sample(memory, size)
            states = []
            targets = []
            # Extract datapoints from the data
            for state, action, reward, next_state, done in data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if done:
                    q_values[action] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    q_values_next = self.target_predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                targets.append(q_values)
            self.update(states, targets)

    def train(self, epochs = 1000, mem_size = 1000, batch_size = 200, sync_freq = 500 ) :
        ''' '''
        print("Training phase with "+str(epochs)+" epochs. please wait :) ")
        counter = 0
        epsilon = 0.3
        eps_decay=0.99
        memory = deque([],maxlen=mem_size)
        for i in range(epochs):
            counter += 1
            current_state = self.env.reset()[0]
            current_state_ = torch.from_numpy(current_state).float()
            while True :
                qval = self.predict(current_state_)
                qval_ = qval.data.numpy()
                action = 0 
                if (random.random() < epsilon):
                    action = np.random.randint(0,self.action_size)
                else:
                    action = np.argmax(qval_)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_state_ = torch.from_numpy(next_state).float()
                exp =  (current_state, action, reward, next_state, terminated or truncated)
                memory.append(exp)
                self.replay(memory, batch_size, gamma=1.0)
                current_state = next_state
                current_state_ = next_state_
                if sync_freq % counter == 0 :
                    self.target_update()
                if terminated or truncated :
                    self.env.close()
                    break
        print("Training is done. You can test :)")


    def policy(self, state) :
            '''
            Define the policy of the agent, as a DQL agent it selects "the best action ".
            Ouptut : an int corresponding to the selected action : 0 left , 1 right, 2 do nothing
            '''
            return np.argmax(self.predict(state).data.numpy())




    

class Simu:
    def __init__(self, render_mode="human", agent=None, max_steps=None):
        self.agent = Random_agent(CartPole_V2()) if agent is None else agent(CartPole_V2())
        self.max_steps = 500 if max_steps is None or max_steps > 500 else max_steps
        self.step_count = 0 

        self.env = CartPole_V2(render_mode)
        self.obs = self.env.reset()[0]

    def step(self,state):
        action = self.agent.policy(state)
        print(action, end="--") 
        return self.env.step(action)
    
    def run_simu(self):
        while self.step_count < self.max_steps :
            time.sleep(0.2)
            self.env.render()
            self.obs, reward, terminated, truncated, info = self.step(self.obs)
            self.step_count += 1
            if terminated or truncated :
                self.env.close()
                return
        

class CartPole_V2(CartPoleEnv):
    def __init__(self,render_mode: Optional[str] = None):
        super().__init__(render_mode)
        self.action_space = gym.spaces.Discrete(3)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag if action == 0 else 0
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
        

#env = CartPole_v2(render_mode="human")
#env = gym.make('CartPole-v1',render_mode="human")



#sm = Simu(agent=DQL_agent)

#sm.run_simu()
#print(sm.step_count)