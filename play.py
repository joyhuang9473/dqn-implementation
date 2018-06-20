from __future__ import division

import tensorflow as tf
import numpy as np
import gym
import logging
import argparse
from time import sleep
from PIL import Image

from agent import DQNAgent
from util import preprocess_observation
from memory import MemoryItem

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

def main(config, screen_log, frame_output, max_episodes, load_model):
    game_name = 'BreakoutDeterministic-v4'
    env = gym.make(game_name)
    agent = DQNAgent(config)

    with agent.graph.as_default():
        if load_model:
            _ = agent.load_model(load_model)
            screen_log.info("Load model: {}".format(load_model))

        rewards_per_episode = []
        play_images = []

        try:
            for episode in range(max_episodes):
                init_frame = env.reset()
                play_images.append(Image.fromarray(init_frame))

                next_observation = preprocess_observation(init_frame)
                env.render()

                episode_done = False
                rewards_per_episode.append(0)

                while not episode_done:
                    # sleep for the duration of the frame so we can see what happens
                    sleep(1. / 30)

                    observation = next_observation

                    if len(agent.memory) < config['agent_history_length']:
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
                    if np.random.RandomState().rand() < config['evaluation_exploration']:
                        action = env.action_space.sample()
                    else:
                        action = agent.get_action_from_Q(Qs)

                    next_observation, reward, episode_done, info = env.step(action)
                    play_images.append(Image.fromarray(next_observation))

                    next_observation = preprocess_observation(next_observation)
                    agent.memory.append(MemoryItem(observation, action, reward, episode_done, info))

                    rewards_per_episode[-1] += reward

                    env.render()

                screen_log.info(
                    'episode: {}, reward: {:g}, ave. reward: {:g}, '
                    .format(
                        episode+1,
                        rewards_per_episode[-1],
                        np.mean(rewards_per_episode),
                    )
                )

            play_images[0].save(
                frame_output,
                save_all=True,
                append_images=play_images[1:],
                duration=30,
            )
        except KeyboardInterrupt:
            print("\nUser interrupted playinging...")
        finally:
            env.close()

        screen_log.info(
            'Finished: the best reward {:g}, the ave. reward {:g}.'
            .format(
                np.max(rewards_per_episode),
                np.mean(rewards_per_episode),
            )
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-output",
                        dest="screen_output",
                        default="screen_play_output.log",
                        help="Output path of screen log")
    parser.add_argument("--frame-output",
                        dest="frame_output",
                        default="play_frame_output.gif",
                        help="Output path of play frames")
    parser.add_argument("--max-episodes",
                        dest="max_episodes",
                        type=int,
                        default=10,
                        help="The maximum number of episodes to test for")
    parser.add_argument("--load-model",
                        dest="load_model",
                        help="Pre-trained model to load")
    args = parser.parse_args()

    screen_log = logging.getLogger(__name__)
    screen_log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    fh = logging.FileHandler(str(args.screen_output))
    fh.setFormatter(formatter)
    screen_log.addHandler(ch)
    screen_log.addHandler(fh)

    main(config,
        screen_log,
        args.frame_output,
        args.max_episodes,
        args.load_model,
    )
