import numpy as np
from collections import defaultdict
import random


class Agent:

    def __init__(self, nA=6, alpha=0.1, gamma=0.9, epsilon=0.2):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

        # Defining hyper params
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        print("Hyper parameters: alpha {}, gamma: {}, epsilon: {}"
              .format(alpha, gamma, epsilon))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        action = np.argmax(self.Q[state])

        return action

    def epsilon_greedy(self, state, nA, eps):

        if random.random() > eps:
            return np.argmax(self.Q[state])
        else:
            return random.choice(np.arange(self.nA))

    def update_Q_sarsa(self, alpha, gamma, Q, state, action, reward, next_state=None, next_action=None):
        """Returns updated Q-value for the most recent experience."""
        current = Q[state][action]  # estimate in Q-table (for current state, action pair)
        # get value of state, action pair at next time step
        Qsa_next = Q[next_state][next_action] if next_state is not None else 0
        target = reward + (gamma * Qsa_next)  # construct TD target
        new_value = current + (alpha * (target - current))  # get updated value
        return new_value

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if not done:
            next_action = self.epsilon_greedy(state, self.nA, self.epsilon)
            self.Q[state][action] = self.update_Q_sarsa(self.alpha, self.gamma, self.Q, state, action, reward, next_state,
                                                   next_action)
            state = next_state
            action = next_action

        if done:
            self.Q[state][action] = self.update_Q_sarsa(self.alpha, self.gamma, self.Q, state, action, reward)

