import numpy as np
import random

from .ringbuffer import RingBuffer
from .memoryitem import MemoryItem

class Memory(RingBuffer):
    def get_state(self, index=None, n=4):
        if index is None:
            index = self.end

        retrieved_num = n + 1

        if len(self) >= retrieved_num:
            start = index - retrieved_num

            if start >= 0:
                observations = self.concat_observations(self.data[start:index])
            else:
                start = start % len(self.data)
                observations = self.concat_observations(self.data[start:]+self.data[:index])

            return np.stack(observations, axis=2)

        return None

    def get_recent_state(self, current_observation, n=3):
        current_memory_item = MemoryItem(current_observation, None, None, None, None)
        index = self.end

        retrieved_num = n + 1

        if len(self) >= retrieved_num:
            start = index - retrieved_num

            if start >= 0:
                observations = self.concat_observations(self.data[start:index]+[current_memory_item])
            else:
                start = start % len(self.data)
                observations = self.concat_observations(self.data[start:]+self.data[:index]+[current_memory_item])

            return np.stack(observations, axis=2)

        return None

    def get_batch(self, k=32, include_last=False):
        if include_last:
            ind = random.sample(range(5, len(self)), k - 1) + [self.end]
        else:
            ind = random.sample(range(5, len(self)), k)
        states = [self.get_state(index=i) for i in ind]
        next_states = [self.get_state(index=i + 1) for i in ind]

        actions, rewards, dones, infos = zip(*[(self.data[i].action,
                                                self.data[i].reward,
                                                self.data[i].done,
                                                self.data[i].info)
                                               for i in ind])

        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        next_states = np.stack(next_states, axis=0)
        rewards = np.stack(rewards, axis=0)
        dones = np.stack(dones, axis=0)

        return states, actions, next_states, rewards, dones, infos

    def concat_observations(self, data):
        observations = []
        zero_out = False

        # keep the last observation
        last_memory_item = data[-1]
        observations.append(last_memory_item.observation)

        for current_idx in reversed(range(1, len(data[:-1]))):
            if data[current_idx-1].done: # currnt terminal signal was stored in data[current_idx-1]
                zero_out = True

            if zero_out:
                observations.insert(0, self.zeroed_observation(data[current_idx].observation))
            else:
                observations.insert(0, data[current_idx].observation)

        return observations

    def zeroed_observation(self, observation):
        """Return an array of zeros with same shape as given observation

        # Argument
            observation (list): List of observation
        
        # Return
            A np.ndarray of zeros with observation.shape
        """
        if hasattr(observation, 'shape'):
            return np.zeros(observation.shape)
        elif hasattr(observation, '__iter__'):
            out = []
            for x in observation:
                out.append(zeroed_observation(x))
            return out
        else:
            return 0.
