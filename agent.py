import tensorflow as tf
import numpy as np
from memory import Memory
from util import normalized

class DQNAgent:
    def __init__(
        self,
        config,
    ):
        self._config = config
        self._network_layers = []
        self._train_op_input = {}
        self._train_op = None
        self._loss_op = None
        self._validation_op_input = {} # TODO
        self._ave_Q_op = None
        self._ave_reward = None
        self._tf_summary = {}
        self._tf_session = None

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_graph()
            self._build_train_op()
            self._build_validation_op()

            self._tf_session = tf.Session()
            self._tf_session.run(tf.global_variables_initializer())

            self._saver = tf.train.Saver()

        self.memory = Memory(self._config['replay_memory_size'])

    def _build_graph(self):
        bottom_layer = None

        for layer_name, layer_conf in self._config['network']:
            with tf.variable_scope(layer_name):
                if 'input' in layer_name:
                    [h, w] = self._config['input_size']
                    c = self._config['agent_history_length']

                    # The size of the minibatch can be either 32 when training
                    # or 1 when evaluating Q for a given state,
                    # therefore set to None.
                    top_layer = tf.placeholder(
                        tf.float32,
                        shape=(None, h, w, c),
                        name='input_layer',
                    )
                elif 'conv' in layer_name:
                    W_s = layer_conf['W_size']
                    s = layer_conf['stride']
                    i = layer_conf['in']
                    o = layer_conf['out']

                    W, b = self._get_filter_and_bias(
                        W_shape=(W_s, W_s, i, o),
                        b_shape=o,
                    )
                    a_tensor = tf.nn.conv2d(
                        bottom_layer,
                        W,
                        strides=(1, s, s, 1),
                        padding='VALID',
                    )
                    a_tensor = tf.nn.bias_add(a_tensor, b)
                    top_layer = tf.nn.relu(a_tensor)
                elif 'fc' in layer_name:
                    n = layer_conf['num_relus']
                    _, h, w, c = bottom_layer.shape.as_list()
                    in_dim = h * w * c
                    conv_out_flattened = tf.reshape(bottom_layer, (-1, in_dim))
                    W, b = self._get_filter_and_bias(
                        W_shape=(in_dim, n),
                        b_shape=n,
                    )
                    top_layer = tf.nn.relu(
                        tf.nn.bias_add(
                            tf.matmul(conv_out_flattened, W), b
                        )
                    )
                elif 'output' in layer_name:
                    in_dim = bottom_layer.shape.as_list()[-1]
                    num_actions = self._config['num_actions']
                    W, b = self._get_filter_and_bias(
                        W_shape=(in_dim, num_actions),
                        b_shape=(num_actions),
                    )
                    top_layer = tf.nn.bias_add(
                        tf.matmul(bottom_layer, W), b
                    )

                self._network_layers.append(top_layer)
                bottom_layer = top_layer

    def _build_train_op(self):
        with tf.variable_scope('train_op'):
            ys = tf.placeholder(
                dtype=tf.float32,
                shape=(self._config['minibatch_size']),
                name='ys',
            )
            actions = tf.placeholder(
                dtype=tf.uint8,
                shape=(self._config['minibatch_size']),
                name='actions',
            )
            Q_input = self._network_layers[0]

            self._train_op_input = {
                'states': Q_input,
                'actions': actions,
                'ys': ys,
            }

            Q_output = self._network_layers[-1]
            one_hot_actions = tf.one_hot(
                actions,
                self._config['num_actions'],
            )
            Qs_of_action = tf.reduce_sum(
                tf.multiply(Q_output, one_hot_actions),
                axis=1,
            )
            self._loss_op = tf.reduce_mean(tf.square(ys - Qs_of_action))
            self._train_op = tf.train.RMSPropOptimizer(
                learning_rate=self._config['learning_rate'],
                decay=self._config['rms_prop_decay'],
                momentum=self._config['gradient_momentum'],
                epsilon=self._config['min_squared_gradient'],
                centered=True,
            ).minimize(self._loss_op)
            self._tf_summary['loss'] = tf.summary.scalar('loss', self._loss_op)

    def _build_validation_op(self): # TODO
        with tf.variable_scope('validation_op'):
            self._ave_Q_op = tf.Variable(
                0.0,
                name='ave_Q',
            )
            self._tf_summary['average_Q'] = tf.summary.scalar(
                'average_Q',
                self._ave_Q_op,
            )

            self._ave_reward = tf.Variable(
                0.0,
                name='ave_reward',
            )
            self._tf_summary['reward_per_episode'] = tf.summary.scalar(
                'reward_per_episode',
                self._ave_reward,
            )

    def _get_filter_and_bias(self, W_shape, b_shape):
        W = tf.get_variable(
            'W',
            shape=W_shape,
            initializer=self._get_variable_initializer()
        )
        b = tf.get_variable(
            'b',
            shape=b_shape,
            initializer=self._get_variable_initializer()
        )
        return (W, b)

    def _get_variable_initializer(self):
        # TODO: Check the original initialization
        return tf.truncated_normal_initializer(
            mean=self._config['var_init_mean'],
            stddev=self._config['var_init_stddev'],
        )

    def get_recent_state(self, current_observation, n=3):
        # append current observation with last n observations
        state = self.memory.get_recent_state(current_observation, n=3)
        state = normalized(state)
        # (n,h,w,c)
        state= np.stack([state], axis=0)

        return state

    def get_action_from_Q(self, Qs):
        return np.argmax(Qs)

    def get_Q_values(self, states):
        assert(states.ndim == 4)
        assert(states.dtype == np.float32)
        return self._tf_session.run(
            self._network_layers[-1],
            feed_dict={self._network_layers[0]: states},
        )

    def optimize_Q(self):
        batch_size = self._config['minibatch_size']
        gamma = self._config['discount_factor']
        num_actions = self._config['num_actions']

        states, actions, next_states, rewards, dones, infos = self.memory.get_batch(batch_size)
        states = normalized(states)
        next_states = normalized(next_states)

        mask = np.zeros((batch_size, num_actions))
        mask[np.arange(batch_size), actions] = 1

        Qs_next = self.get_Q_values(next_states)
        # The Q values of the terminal states is 0 by definition, so override them
        Qs_next[dones] = 0

        ys = rewards + gamma * (
            np.amax(
                np.multiply(Qs_next, mask),
                axis=1,
            )
        )

        _, loss, loss_summary_str = self._tf_session.run(
            [self._train_op, self._loss_op, self._tf_summary['loss']],
            feed_dict={
               self._train_op_input['states']: states,
               self._train_op_input['actions']: actions,
               self._train_op_input['ys']: ys,
            }
        )

        return (loss, loss_summary_str)

    def evaluate(self, ave_Q, ave_reward): # TODO
        _, Q_summary_str = self._tf_session.run(
            [tf.assign(self._ave_Q_op, ave_Q),
            self._tf_summary['average_Q']],
        )

        _, reward_summary_str = self._tf_session.run(
            [tf.assign(self._ave_reward, ave_reward),
            self._tf_summary['reward_per_episode']]
        )

        return (Q_summary_str, reward_summary_str)

    def load_model(self, filepath):
        self._saver.restore(self._tf_session, filepath)
        step = int(filepath.split('-')[-1])

        return step

    def save_model(self, filepath, step):
        save_path = self._saver.save(
            self._tf_session,
            filepath,
            global_step=step,
        )
        print('checkpoint saved at {}'.format(save_path))

    def load_memory(self, filepath):
        self.memory.load(filepath)

    def save_memory(self, filepath, step):
        save_path = filepath + '.step-%d' % step
        self.memory.save(save_path)
        print('memory saved at {}'.format(save_path))
