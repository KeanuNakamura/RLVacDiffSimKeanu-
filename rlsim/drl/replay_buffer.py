from typing import List, Tuple

import numpy as np

from rlsim.memory import Memory


class PrioritizedReplayBuffer:
    """Prioritized experience replay with fixed-size transition sampling.

    Transitions are prioritized by TD error, updated after each training step.
    """

    def __init__(
        self,
        sample_size: int = 64,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        self.sample_size = sample_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

        self._transitions: List[dict] = []
        self._priorities = np.array([], dtype=np.float64)

    def __len__(self) -> int:
        return len(self._transitions)

    def add_dqn_episode(self, memory: Memory) -> None:
        aspace = memory.act_space
        actions = memory.actions
        for i in range(len(aspace) - 1):
            self._transitions.append(
                {
                    "state": memory.states[i],
                    "next_state": memory.next_states[i],
                    "reward": memory.rewards[i],
                    "taken_action": [aspace[i][actions[i]]],
                    "next_aspace": aspace[i + 1],
                }
            )
            self._priorities = np.append(self._priorities, self.max_priority)

    def add_context_bandit_episode(self, memory: Memory) -> None:
        aspace = memory.act_space
        actions = memory.actions
        for i in range(len(aspace)):
            self._transitions.append(
                {
                    "state": memory.states[i],
                    "taken_action": [aspace[i][actions[i]]],
                    "barrier": memory.barrier[i],
                    "freq": memory.freq[i],
                }
            )
            self._priorities = np.append(self._priorities, self.max_priority)

    def sample(self) -> Tuple[List[dict], np.ndarray, np.ndarray]:
        n = len(self._transitions)
        if n == 0:
            return [], np.array([], dtype=np.int64), np.array([])

        priorities = self._priorities[:n] ** self.alpha
        probs = priorities / priorities.sum()

        indices = np.random.choice(
            n, size=self.sample_size, replace=True, p=probs
        )

        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        transitions = [self._transitions[i] for i in indices]
        self.beta = min(1.0, self.beta + self.beta_increment)

        return transitions, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + self.epsilon
            self._priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
