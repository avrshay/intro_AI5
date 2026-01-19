from typing import List, Tuple

import gymnasium as gym
import numpy as np
import random

SEED = 63

# Set the seed
rng = np.random.default_rng(SEED)


class Qlearning:
    def __init__(
        self,
        learning_rate: float,
        gamma: float,
        state_size: int,
        action_size: int,
        epsilon: float,
    ):
        self.state_size = state_size
        self.action_space_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.qtable = np.zeros((state_size, action_size))

    def update(self, state: int, action: int, reward: float, new_state: int):
        """In this function you need to implement the update of the Q-table.

        Args:
            state (int): Current state
            action (int): Action taken in the current state
            reward (float): Reward received after taking the action
            new_state (int): New state reached after taking the action.
        """
        max_val=np.max(self.qtable[new_state])
        self.qtable[state,action]=(1-self.learning_rate)*self.qtable[state,action]+self.learning_rate*(reward+self.gamma*max_val)

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_space_size))

    def select_epsilon_greedy_action(self, state: int) -> int:
        """Select an action from the Q-table."""
        # exploration
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.action_space_size))
        # exploitation
        max_q = np.max(self.qtable[state]) # mav val
        best_actions = np.where(self.qtable[state] == max_q)[0] # actions of max val
        return int(rng.choice(best_actions))
    def train_episode(self, env: gym.Env) -> Tuple[float, int]:
        """Train the agent for a single episode.

        Notice an episode is a single run of the environment until the agent reaches a terminal state
        (the return value of env.step() is True for the third and fourth elements)


        :param env: The environment to train the agent on.
        :return: the cumulative reward obtained during the episode and the number of steps executed in the episode.
        """
        state_cur, info = env.reset()
        all_reward = 0.0
        num_steps = 0
        terminated=False
        truncated=False
        while not (terminated or truncated):
            action = self.select_epsilon_greedy_action(state_cur)
            num_steps=num_steps+1
            new_state, reward, terminated, truncated, info = env.step(action)
            all_reward=all_reward+reward
            self.update(state_cur,action,reward,new_state) # update val
            state_cur=new_state
        return all_reward,num_steps


    def run_environment(
        self, env: gym.Env, num_episodes: int
    ) -> Tuple[List[float], List[int]]:
        """
        Run the environment with the given policy.

        Args:
            env (gym.Env): The environment to train the agent on.
            num_episodes (int): The number of episodes to run the environment.

        Returns:
            A tuple (total_rewards, total_steps).
        """
        self.reset_qtable() #init
        list_reward=[]
        list_steps=[]
        reward=0.0
        steps=0
        env.reset(seed=SEED)
        for i in range(num_episodes):
            reward,steps= self.train_episode(env)  #episode
            list_steps.append(steps)
            list_reward.append(reward)
        return list_reward,list_steps



