import numpy as np
import cv2
import random


class ReplayBuffer(object):
    """
    TODO:
     1. Add unit tests
    """

    def __init__(self, buffer_size, obs_shape, obs_history_len, batch_size):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_history_len = obs_history_len
        self.batch_size = batch_size # batch_size of states

        self.obs_type = np.uint8
        self.state_shape = (self.obs_history_len,) + self.obs_shape
        self.state_type = self.obs_type
        self.num_in_buffer = 0 # stores the number of frames
        self.empty_idx = 0
        self.obs_type = np.uint8

        self.obs = np.empty((self.buffer_size,) + self.obs_shape, dtype=self.obs_type)
        self.a = np.empty(self.buffer_size, dtype=np.uint8)
        self.r = np.empty(self.buffer_size, dtype=np.float32)
        self.done = np.empty(self.buffer_size, dtype=np.bool)

    def store_observation(self, obs):
        """stores new observation, overwrites the old ones"""
        self.obs[self.empty_idx] = obs
        current_idx = self.empty_idx
        self.empty_idx += 1
        self.empty_idx %= self.buffer_size
        self.num_in_buffer += 1
        self.num_in_buffer = min(self.buffer_size, self.num_in_buffer)
        return current_idx

    def store_a_r_d(self, idx, a, r, done):
        self.a[idx] = a
        self.r[idx] = r
        self.done[idx] = done

    def _retrieve_state(self, last_obs_idx):
        """Note: the format is in CHW"""
        state = np.zeros(self.state_shape, dtype=self.state_type)
        first_obs_idx = last_obs_idx - self.obs_history_len + 1
        # check if the buffer is not filled
        # if not then the tail of the buffer is empty and start from 0
        if first_obs_idx < 0 and self.num_in_buffer < self.buffer_size:
            first_obs_idx = 0

        # keep the last observation, since it can not be terminal
        # since at least it should come from env.reset()
        state_offset = self.obs_history_len - 1
        state[state_offset] = self.obs[last_obs_idx]
        state_offset -= 1
        for idx in reversed(range(first_obs_idx, last_obs_idx)):
            # if obs is terminal, fill context with zeros
            if self.done[idx]:
                break
            # if obs is not terminal, continue
            else:
                state[state_offset] = self.obs[idx]
                state_offset -= 1
        return state

    def retrieve_last_state(self):
        last_obs_idx = (self.empty_idx - 1) % self.buffer_size
        return self._retrieve_state(last_obs_idx)

    def can_sample(self):
        """
        make sure we have  enough observations for states and next_states
        Note: next state is with an offset of 1 observation, hence we need
        buffer_size + 1
        """
        return self.batch_size + 1 <= self.num_in_buffer

    def sample(self):
        """Note: returns batch in the format NCHW, which is suitable for PyTorch"""
        assert self.can_sample(), "Not enough observations in the buffer to sample"
        # sample unique idxes
        idxes = []
        while not len(idxes) == self.batch_size:
            idx_candidate = random.randint(0, self.num_in_buffer - 2)
            if idx_candidate not in idxes:
                idxes.append(idx_candidate)
        idxes.sort()
        s_batch = np.stack([self._retrieve_state(idx) for idx in idxes], axis=0)
        a_batch = self.a[idxes]
        r_batch = self.r[idxes]
        done_batch = np.array([float(self.done[idx]) for idx in idxes], dtype=np.float32)
        sp_batch = np.stack([self._retrieve_state(idx + 1) for idx in idxes], axis=0)
        return s_batch, a_batch, r_batch, done_batch, sp_batch

