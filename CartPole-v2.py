import gym
import numpy as np
import math
import time
from typing import Optional
from gym import logger 
from gym.utils.env_checker import check_env
from gym.envs.classic_control import CartPoleEnv




class Random_agent:
    def __init__(self, env):
        self.env = env
        pass

    def policy(self) :
        '''
        Define the policy of the agent, as a random agent it selects a random action given a uniform probability distribution
        Ouptut : an int corresponding to the selected action : 0 left , 1 right, 2 do nothing
	    '''
        return self.env.action_space.sample()
    

class Simu:
    def __init__(self, render_mode="human", agent=None, max_steps=None):
        self.env = CartPole_v2(render_mode)
        self.obs = self.env.reset()
        self.agent = Random_agent(self.env) if agent is None else agent
        self.max_steps = 500 if max_steps is None or max_steps > 500 else max_steps
        self.step_count = 0 

    def step(self):
        action = self.agent.policy() 
        return self.env.step(action)
    
    def run_simu(self):
        while self.step_count < self.max_steps :
            time.sleep(0.2)
            self.env.render()
            self.obs, reward, terminated, truncated, info = self.step()
            self.step_count += 1
            if terminated :
                self.env.close()
                return
        

class CartPole_v2(CartPoleEnv):
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



sm = Simu()
sm.run_simu()
print(sm.step_count)