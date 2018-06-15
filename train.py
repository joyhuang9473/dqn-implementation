from __future__ import division

import os
import gym
import numpy as np
import logging
import argparse
import time
import tensorflow as tf

from agent import DQNAgent
from util import preprocess_observation
from memory import MemoryItem

screen_log = logging.getLogger(__name__)
screen_log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
fh = logging.FileHandler(str('./screen_train_output.log'))
fh.setFormatter(formatter)
screen_log.addHandler(ch)
screen_log.addHandler(fh)

config = {
    'network': [
        ('input', {}),
        ('conv1', {'W_size': 8, 'stride': 4, 'in': 4, 'out': 16}),
        ('conv2', {'W_size': 4, 'stride': 2, 'in': 16, 'out': 32}),
        ('fc1', {'num_relus': 256}),
        ('output', {}),
    ],
    'input_size': [84, 84], # height, width
    'num_actions': 4,
    'var_init_mean': 0.0,
    'var_init_stddev': 0.01,
    'minibatch_size': 32,
    'replay_memory_size': 10 ** 6,
    'agent_history_length': 4,
    'discount_factor': 0.95,
    #'action_repeat': 4,
    #'update_frequency': 4,
    'learning_rate': 0.00025,
    'rms_prop_decay': 0.95,
    'gradient_momentum': 0.0,
    'min_squared_gradient': 0.01,
    #'initial_exploration': 1,
    'final_exploration': 0.1,
    'final_exploration_frame': 10 ** 6,
    'replay_start_size': 5 * (10 ** 4),
    #'no-op_max': 30,
    'validation_size': 500,
    'evaluation_exploration': 0.05,
}
game_name = 'BreakoutDeterministic-v4'
env = gym.make(game_name)

def get_epsilon(config, step):
    rss = config['replay_start_size']
    n_steps = config['final_exploration_frame']
    min_epsilon = config['final_exploration']

    if step < rss:
        epsilon = 1
    elif (step < n_steps):
        epsilon = 1 - (1 - min_epsilon) / n_steps * step
    else:
        epsilon = min_epsilon

    return epsilon

def reset_random_env(random_steps=30):
    _ = env.reset()

    for i in range(random_steps):
        action = env.action_space.sample()
        next_observation, reward, episode_done, info = env.step(action)

    return next_observation

def main(config, max_num_of_steps, max_num_of_episodes, load_model, save_model, load_memory, save_memory, log_path):
    agent = DQNAgent(config)

    init_frame_nums = config['minibatch_size']+config['agent_history_length']

    with agent.graph.as_default():
        if load_model:
            step = agent.load_model(load_model)
            screen_log.info("Load model: {}".format(load_model))
            screen_log.info("Start from step {}".format(step))
        else:
            step = 0

        if load_memory:
            agent.load_memory(load_memory)
            n_frames = len(agent.memory)
            screen_log.info("Load memory: {}".format(load_memory))
            screen_log.info("Memory size: {}".format(n_frames))

        log_name = ('{:02}{:02}{:02}{:02}{:02}'
                    .format(*time.localtime()[1:6]))
        summary_writer = tf.summary.FileWriter(
            logdir=os.path.join(log_path, '{}'.format(log_name)),
            graph=agent.graph
        )

        episode = 0
        rewards_per_episode = []
        sum_Qs = .0
        sum_losses = .0

        try:
            while (step < max_num_of_steps
                and episode < max_num_of_episodes
            ):
                episode += 1
                episode_done = False

                next_observation = reset_random_env()
                next_observation = preprocess_observation(next_observation)

                rewards_per_episode.append(0)

                while not episode_done:
                    observation = next_observation

                    if len(agent.memory) < init_frame_nums:
                        # init replay memory
                        action = env.action_space.sample()

                        next_observation, reward, episode_done, info = env.step(action)
                        next_observation = preprocess_observation(next_observation)
                        agent.memory.append(MemoryItem(observation, action, reward, episode_done, info))

                        continue

                    state = agent.get_recent_state(observation)
                    Qs = agent.get_Q_values(state)
                    Qs = Qs[0]

                    # epsilon-greedy action selection
                    epsilon = get_epsilon(config, step)
                    if np.random.RandomState().rand() < epsilon:
                        action = env.action_space.sample()
                    else:
                        action = agent.get_action_from_Q(Qs)

                    next_observation, reward, episode_done, info = env.step(action)
                    next_observation = preprocess_observation(next_observation)
                    agent.memory.append(MemoryItem(observation, action, reward, episode_done, info))

                    step += 1
                    rewards_per_episode[-1] += reward
                    sum_Qs += Qs[action]

                    # train step
                    loss, loss_summary_str = agent.optimize_Q()
                    summary_writer.add_summary(loss_summary_str, step)
                    sum_losses += loss

                    if step % 1000 == 0:
                        ave_loss = sum_losses / step
                        ave_reward = np.mean(rewards_per_episode)
                        ave_Q = sum_Qs / step

                        [Q_summary_str, reward_summary_str] = agent.evaluate(ave_reward, ave_Q)

                        summary_writer.add_summary(Q_summary_str, step)
                        summary_writer.add_summary(reward_summary_str, step)

                        screen_log.info(
                            'step: {}, ave. loss: {:g}, '
                            'ave. reward: {:g}, ave. Q: {:g}'
                            .format(
                                step,
                                ave_loss,
                                ave_reward,
                                ave_Q,
                            )
                        )
                    if step % 10000 == 0:
                        agent.save_model(save_model, step)
                    if step % 1000000 == 0:
                        agent.save_memory(save_memory, step)

        except KeyboardInterrupt:
            print("\nUser interrupted training...")
        finally:
            summary_writer.close()

            agent.save_model(save_model, step)
            agent.save_memory(save_memory, step)

        screen_log.info(
            'Finished: the number of steps {}, the number of episodes {}.'
            .format(step, episode)
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps",
                        dest="max_steps",
                        type=int,
                        default=1000000,
                        help="The maximum number of steps to train for")
    parser.add_argument("--max-episodes",
                        dest="max_episodes",
                        type=int,
                        default=1000000,
                        help="The maximum number of episodes to train for")
    parser.add_argument("--load-model",
                        dest="load_model",
                        help="Pre-trained model to load")
    parser.add_argument("--save-model",
                        dest="save_model",
                        default="./checkpoints/breakout-v4",
                        help="Save model to file")
    parser.add_argument("--load-memory",
                        dest="load_memory",
                        help="Pre-filled memory to load")
    parser.add_argument("--save-memory",
                        dest="save_memory",
                        default="./data/memory.pkl",
                        help="Save memory to file")
    parser.add_argument("--log-path",
                        dest="log_path",
                        default="./log/",
                        help="Save log to the path")
    args = parser.parse_args()

    main(config,
        args.max_steps,
        args.max_episodes,
        args.load_model,
        args.save_model,
        args.load_memory,
        args.save_memory,
        args.log_path,
    )
