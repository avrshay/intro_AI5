from typing import Tuple

import numpy as np
import math


class ValueIteration:

    def __init__(
            self,
            theta=0.0001,
            discount_factor=1.0,
    ):
        self.theta = theta
        self.discount_factor = discount_factor

    def calculate_q_values(
            self, current_capital: int, value_function: np.ndarray, rewards: np.ndarray
    ) -> np.ndarray:
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            current_capital: The gambler’s capital. Integer. (state)
            value_function: The vector that contains values at each state. (the recursive value function)
            rewards: The reward vector. (the immediate reward according to the gambler's problem definition)

        Returns:
            A vector containing the expected value of each action in THIS state.
            Its length equals to the number of actions.
        """
        can_bet = min(current_capital, 100 - current_capital)

        all_actions = np.arange(0, can_bet + 1)  # function that creates an array of numbers in a specific range

        res_values = np.zeros(
            len(all_actions))  # a value for every possible action (for every bet amount) in the current situation, not just the maximum.

        p1 = 5 / 12  # more or less than 7 - loss
        p2 = 1 / 6  # 7 - win

        # calculate for each action the next steps:
        for i, action in enumerate(all_actions):
            # current_capital= current money

            win_state = current_capital + action
            loss_state = current_capital - action
            half_loss_state = current_capital - math.ceil(action / 2)  # round it up

            # given gama=1
            # val= rewards(current state) + gama * sum(p * value_function)
            win_val = p2 * (rewards[win_state] + value_function[win_state])
            loss_val = p1 * (rewards[loss_state] + value_function[loss_state])
            half_loss_val = p1 * (rewards[half_loss_state] + value_function[half_loss_state])

            res_values[i] = win_val + loss_val + half_loss_val

        return res_values

    def value_iteration_for_gamblers(self) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        pass

        # Define the values for each state
        V = np.zeros(101)

        # rewards vectors
        rewards = np.zeros(101)
        rewards[-1] = 1  # win

        # The policy will keep for each situation what is the optimal bet
        policy = np.zeros(101)

        while True:

            delta=0
            for s in range(1, 100): # 1,...99
                old_v = V[s]
                q_values = self.calculate_q_values(s, V, rewards)
                V[s] = np.max(q_values) # update the max val that we get
                delta = max(delta, abs(old_v - V[s]))

                policy[s] = np.argmax(q_values)

            # the epsilon
            if delta <= self.theta:
                break

        # Calculate the final optimal policy after convergence
        for s in range(1, 100):
            q_values = self.calculate_q_values(s, V, rewards)
            policy[s] = np.argmax(q_values)

        return policy, V
