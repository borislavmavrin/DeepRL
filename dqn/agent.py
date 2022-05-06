import sys
import os
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from config import Config
from common.replay_buffer import ReplayBuffer
import numpy as np
import time
from common.atari_wrappers import make_atari, wrap_deepmind
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


config = Config()


class DQN(nn.Module):

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(64 * 7 * 7, 512)
        self.head = nn.Linear(512, self.num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear1(x.view(x.shape[0], -1)))
        return self.head(x)


def experience_generator(config, Q, experience_queue):
    env = make_atari(config.game, noop_reset=False, frame_skip=4, max_pool=True)
    env = wrap_deepmind(env, episode_life=True, clip_rewards=False, frame_stack=False, scale=False)
    # local mini replay buffer
    local_replay_buffer = ReplayBuffer(
        buffer_size=config.local_buffer_size,
        obs_shape=(config.height, config.width),
        obs_history_len=config.agent_history_length,
        batch_size=config.batch_size)
    total_steps_counter = 0
    step_counter_from_buffer_filled = 0
    replay_buffer_filled = False
    epsilon = config.epsilon_start
    max_number_of_steps_reached = False
    rewards_lst = []
    while not max_number_of_steps_reached:
        step_in_episode_counter = 0
        frame_counter = 0
        reward_in_episode = 0.
        done = False
        obs = env.reset()
        while not done:
            # replay buffer
            idx = local_replay_buffer.store_observation(obs)

            # print(step_in_episode)
            # print(frame_counter)

            # epsilon schedule
            if replay_buffer_filled and \
                    (step_counter_from_buffer_filled < config.eps_nsteps):
                # start updating schedule after finished populating replay buffer and continue
                # for config.eps_nsteps
                epsilon = config.epsilon_start + \
                          step_counter_from_buffer_filled * config.epsilon_step_size

            # take action
            random_action = np.random.choice([True, False], p=[epsilon, 1. - epsilon])
            if random_action or not replay_buffer_filled:
                a = env.action_space.sample()
            else:
                current_state = local_replay_buffer.retrieve_last_state()
                current_state = torch.from_numpy(current_state).to(config.device)
                current_state = current_state.unsqueeze(dim=0)
                # current_state = current_state.permute(0, 3, 1, 2)
                current_state = current_state.type(torch.float32) / 255.
                value = Q(current_state).detach()
                value = value.cpu().numpy()
                best_action = np.argmax(value)
                a = int(best_action)
            obs_next, raw_r, done, info = env.step(a)
            was_real_done = env.env.was_real_done
            r = np.sign(raw_r)
            # write to replay buffer
            local_replay_buffer.store_a_r_d(idx, a, r, done)
            # register reward
            # reward_in_episode += raw_r
            # update counters
            frame_counter += 4
            step_in_episode_counter += 1
            total_steps_counter += 1
            if replay_buffer_filled:
                step_counter_from_buffer_filled += 1

            # check if buffer is filled up to the desired level
            if total_steps_counter == config.replay_start_size:
                replay_buffer_filled = True

            if step_counter_from_buffer_filled > config.max_number_of_steps:
                max_number_of_steps_reached = True
                break

            experience_queue.put((obs, a, raw_r, done, was_real_done))

            obs = obs_next

        # update global stats
        rewards_lst.append(reward_in_episode)

    experience_queue.put(None)


class DQNAgentMP(object):
    """DQN
    TODO:
     0. Refactor config and DQN (i.e. number of actions, may be conv output)
     1. Add unit tests
     2. Add tensorboard (loss, gradients, q outputs, q weight norms)
     3. Add save/restore model method
     4. tune performance
    """
    def __init__(self, config):
        self.config = config
        self.pbar = tqdm(total=self.config.max_number_of_steps)
        env = make_atari(self.config.game, noop_reset=False, frame_skip=4, max_pool=True)
        self.config.num_actions = env.action_space.n
        self.Q = DQN(self.config.num_actions).to(self.config.device)
        self.target_Q = DQN(self.config.num_actions).to(self.config.device)
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.Q.eval()
        self.optimizer = torch.optim.Adam(self.Q.parameters(),
                                          self.config.lr_start, eps=self.config.eps)
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.config.buffer_size,
            obs_shape=(self.config.height, self.config.width),
            obs_history_len=self.config.agent_history_length,
            batch_size=self.config.batch_size)
        self.experience_queue = mp.Queue(maxsize=self.config.experience_queue_size)
        self.experience_gen_proc = mp.Process(target=experience_generator,
                                              args=(self.config, self.Q, self.experience_queue))
        self.tf_logger = SummaryWriter(log_dir=self.config.log_path)

    def do_sgd_update(self):
        s_batch, a_batch, r_batch, done_batch, sp_batch = \
            self.replay_buffer.sample()

        s_batch_tr = torch.from_numpy(s_batch).to(self.config.device)
        a_batch_tr = torch.from_numpy(a_batch).to(self.config.device)
        r_batch_tr = torch.from_numpy(r_batch).to(self.config.device)
        sp_batch_tr = torch.from_numpy(sp_batch).to(self.config.device)
        done_batch_tr = torch.tensor(done_batch).to(self.config.device)

        # NHWC -> NCHW
        # s_batch_tr = s_batch_tr.permute(0, 3, 1, 2)
        s_batch_tr = s_batch_tr.type(torch.float32) / 255.
        # sp_batch_tr = sp_batch_tr.permute(0, 3, 1, 2)
        sp_batch_tr = sp_batch_tr.type(torch.float32) / 255.
        a_batch_tr = a_batch_tr.type(torch.int64)
        done_batch_tr = done_batch_tr.type(torch.float32)

        q = self.Q(s_batch_tr).gather(1, a_batch_tr.unsqueeze(-1)).squeeze(-1)

        target_q = self.target_Q(sp_batch_tr).detach().max(dim=1)[0]

        target = r_batch_tr + (1. - done_batch_tr) * self.config.gamma * target_q

        criterion = nn.SmoothL1Loss()
        loss = criterion(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def best_action(self, current_state):
        current_state = torch.from_numpy(current_state).to(self.config.device)
        current_state = current_state.unsqueeze(dim=0)
        # current_state = current_state.permute(0, 3, 1, 2)
        current_state = current_state.type(torch.float32) / 255.
        value = self.Q(current_state).detach()
        value = value.cpu().numpy()
        best_action = np.argmax(value)
        best_action = int(best_action)
        return best_action

    def train(self):
        self.pbar.set_description("Filling replay buffer...")
        total_steps_counter = 0
        step_counter_from_buffer_filled = 0
        replay_buffer_filled = False
        rewards_lst = []
        step_in_episode_counter = 0
        reward_in_episode = 0.
        was_real_done = False
        max_number_of_steps_reached = False
        self.experience_gen_proc.start()

        while self.experience_gen_proc.is_alive() and not max_number_of_steps_reached:
            frame_counter = 0
            if was_real_done:
                rewards_lst.append(reward_in_episode)
                reward_in_episode = 0.
                step_in_episode_counter = 0
                was_real_done = False
            done = False
            while not done:
                transition = self.experience_queue.get()
                if transition is None:
                    self.experience_gen_proc.join()
                    break
                obs, a, raw_r, done, was_real_done = transition
                r = np.sign(raw_r)
                # replay buffer
                idx = self.replay_buffer.store_observation(obs)
                self.replay_buffer.store_a_r_d(idx, a, r, done)
                # register reward
                reward_in_episode += raw_r
                # update counters
                frame_counter += 4
                step_in_episode_counter += 1
                total_steps_counter += 1
                if replay_buffer_filled:
                    step_counter_from_buffer_filled += 1

                # check if buffer is filled up to the desired level
                if self.replay_buffer.num_in_buffer == self.config.replay_start_size:
                    replay_buffer_filled = True
                    self.pbar.set_description("Training...")
                    # start timer
                    timer_start = time.time()

                # do SGD
                if step_counter_from_buffer_filled % self.config.sgd_update_frequency == 0 and \
                        self.replay_buffer.can_sample() and \
                        replay_buffer_filled:
                    # print("SGD")
                    self.do_sgd_update()

                # update target Q
                if replay_buffer_filled and \
                        (step_counter_from_buffer_filled % self.config.target_network_update_frequency == 0):
                    # print("Update target Q")
                    self.target_Q.load_state_dict(self.Q.state_dict())

                # print some stats
                if step_counter_from_buffer_filled and step_counter_from_buffer_filled % self.config.result_print_freq == 0 \
                        and rewards_lst:
                    time_elapsed = time.time() - timer_start
                    timer_start = time.time()
                    steps_per_sec = self.config.result_print_freq / float(time_elapsed)
                    avg_100_return = np.mean(rewards_lst[-100:])
                    avg_100_return = round(avg_100_return, 2)
                    self.tf_logger.add_scalar("average_reward_100_episodes", avg_100_return,
                                              step_counter_from_buffer_filled)
                    self.tf_logger.flush()
                    self.pbar.update(self.config.result_print_freq)
                    self.pbar.set_description("Average reward %.2f" % avg_100_return)

                if step_counter_from_buffer_filled >= config.max_number_of_steps:
                    max_number_of_steps_reached = True
                    break

        self.tf_logger.close()
        self.pbar.close()


if __name__ == '__main__':
    config = Config()

    config.number_of_runs = 3
    config.max_number_of_steps = 5000000
    config.result_print_freq = 250000
    config.game = "PongNoFrameskip-v4"
    config.device = 'cuda'
    mp.set_start_method('spawn', force=True)
    # print(game)
    for run in range(config.number_of_runs):
        print("\n")
        print('=' * 36)
        print("run " + str(run))
        home = '/tmp/tf_logs'
        config.base_path = os.path.join(home, 'dqn', config.game, config.model)
        config.log_path = os.path.join(config.base_path, str(run))
        dqn = DQNAgentMP(config)
        dqn.train()
