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
        pass

    def value_iteration_for_gamblers(self) -> Tuple[np.ndarray, np.ndarray]:
        """ """
        pass
        # return policy, V
