import numpy as np
import sys
import pickle
import tensorflow as tf
from utils import greedy, ep_greedy

def clipped_error(x):
    # Huber loss
    try:
      return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
      return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

class nn_model:
    def __init__(self, shape, a_list, name, lrate, load = False):
        self.lrate = lrate
        self._name = name
        self._path = 'trained_agents/'
        self.shape = shape
        self.a_list = a_list
        self.w = {}
        self.w_t = {}
        self.assign_op = {}
        self._graph = tf.Graph()
        config = tf.ConfigProto(device_count = {'GPU': 0})
        self._sess = tf.Session(graph=self._graph,config=config)
        try:
            self._build()
        except Exception as e:
            print('Model Initialization Fails! Because {}'.format(e))
        else:
            print('Model Generated!')
        if load == True:
            try:
                self.saver.restore(self._sess, self._path + name)
            except:
                print('Restoring Variables Fails!')
            else:
                print('Model Restored!')
        else:
            self._sess.run(self.init_op)
            print('Model Initialized!')

    def get_loss(self, states, actions, targets):
        if hasattr(self, 'loss'):
            feed_dict = {
                'states:0':states.reshape(list(np.shape(states)) + [1]),
                'actions:0':[self.a_list.index(a) for a in actions],
                'targets:0':targets,
            }
            return self._sess.run(self.loss, feed_dict=feed_dict)
        else:
            print('loss has not been calculated.')

    def __del__(self):
        self._sess.close()

    def _build(self):
        with self._graph.as_default():
            global_step = tf.Variable(0, trainable=False, name='global')
            s = tf.placeholder(dtype=tf.float32, shape=[None] + self.shape + [1], name='states')
            a = tf.placeholder(dtype=tf.uint8, shape=[None], name='actions')
            y = tf.placeholder(dtype=tf.float32, shape=[None], name='targets')
            onehot = tf.one_hot(a, depth=len(self.a_list))
            output_dim = len(self.a_list)
            with tf.variable_scope('train'):
                # conv layer 1
                self.w['l1c1w'] = l1c1w = tf.Variable(
                    tf.truncated_normal([2, 1, 1, 10]) / 100)
                self.w['l1c1b'] = l1c1b = tf.Variable(
                    tf.constant(0.01, shape=[10]))
                conv1 = tf.nn.conv2d(s, l1c1w, [1,1,1,1], 'SAME')
                l1c1 = tf.nn.relu(conv1 + l1c1b)
                self.w['l1c2w'] = l1c2w = tf.Variable(
                    tf.truncated_normal([1, 2, 1, 10]) / 100)
                self.w['l1c2b'] = l1c2b = tf.Variable(
                    tf.constant(0.01, shape=[10]))
                conv2 = tf.nn.conv2d(s, l1c2w, [1,1,1,1], 'SAME')
                l1c2 = tf.nn.relu(conv2 + l1c2b)
                # conv layer 2
                self.w['l2c1w'] = l2c1w = tf.Variable(
                    tf.truncated_normal([1, 2, 10, 10]) / 100)
                self.w['l2c1b'] = l2c1b = tf.Variable(
                    tf.constant(0.01, shape=[10]))
                conv3 = tf.nn.conv2d(l1c1, l2c1w, [1,1,1,1], 'VALID')
                l2c1 = tf.nn.relu(conv3 + l2c1b)
                self.w['l2c2w'] = l2c2w = tf.Variable(
                    tf.truncated_normal([2, 1, 10, 10]) / 100)
                self.w['l2c2b'] = l2c2b = tf.Variable(
                    tf.constant(0.01, shape=[10]))
                conv4 = tf.nn.conv2d(l1c2, l2c2w, [1,1,1,1], 'VALID')
                l2c2 = tf.nn.relu(conv4 + l2c2b)
                # fully connected layer
                L3 = tf.concat([tf.reshape(l2c1, [-1, 120]),
                    tf.reshape(l2c2, [-1, 120])], 1)
                self.w['l3w'] = l3w = tf.Variable(
                    tf.truncated_normal([240, 100]) / 100
                )
                self.w['l3b'] = l3b = tf.Variable(
                    tf.constant(0.01, shape=[100]))
                L4 = tf.nn.relu(tf.matmul(L3,l3w) + l3b)
                self.w['l4w'] = l4w = tf.Variable(
                    tf.truncated_normal([100, output_dim]) / 100
                )
                self.w['l4b'] = l4b = tf.Variable(
                    tf.constant(0.01, shape=[output_dim]))
                self.pred = tf.matmul(L4, l4w) + l4b
                q_act = tf.reduce_sum(self.pred * onehot, reduction_indices=1)
                self.loss = tf.reduce_mean(
                    (q_act  - y)**2) \
                    # + self.lambda_ * (tf.norm(l1w) + tf.norm(l4w))
            with tf.variable_scope('target'):
                # conv layer 1
                self.w_t['l1c1w'] = l1c1w = tf.Variable(
                    tf.truncated_normal([2, 1, 1, 10]) / 100)
                self.w_t['l1c1b'] = l1c1b = tf.Variable(
                    tf.constant(0.01, shape=[10]))
                conv1 = tf.nn.conv2d(s, l1c1w, [1,1,1,1], 'SAME')
                l1c1 = tf.nn.relu(conv1 + l1c1b)
                self.w_t['l1c2w'] = l1c2w = tf.Variable(
                    tf.truncated_normal([1, 2, 1, 10]) / 100)
                self.w_t['l1c2b'] = l1c2b = tf.Variable(
                    tf.constant(0.01, shape=[10]))
                conv2 = tf.nn.conv2d(s, l1c2w, [1,1,1,1], 'SAME')
                l1c2 = tf.nn.relu(conv2 + l1c2b)
                # conv layer 2
                self.w_t['l2c1w'] = l2c1w = tf.Variable(
                    tf.truncated_normal([1, 2, 10, 10]) / 100)
                self.w_t['l2c1b'] = l2c1b = tf.Variable(
                    tf.constant(0.01, shape=[10]))
                conv3 = tf.nn.conv2d(l1c1, l2c1w, [1,1,1,1], 'VALID')
                l2c1 = tf.nn.relu(conv3 + l2c1b)
                self.w_t['l2c2w'] = l2c2w = tf.Variable(
                    tf.truncated_normal([2, 1, 10, 10]) / 100)
                self.w_t['l2c2b'] = l2c2b = tf.Variable(
                    tf.constant(0.01, shape=[10]))
                conv4 = tf.nn.conv2d(l1c2, l2c2w, [1,1,1,1], 'VALID')
                l2c2 = tf.nn.relu(conv4 + l2c2b)
                # fully connected layer
                L3 = tf.concat([tf.reshape(l2c1, [-1, 120]),
                    tf.reshape(l2c2, [-1, 120])], 1)
                self.w_t['l3w'] = l3w = tf.Variable(
                    tf.truncated_normal([240, 100]) / 100
                )
                self.w_t['l3b'] = l3b = tf.Variable(
                    tf.constant(0.01, shape=[100]))
                L4 = tf.nn.relu(tf.matmul(L3,l3w) + l3b)
                self.w_t['l4w'] = l4w = tf.Variable(
                    tf.truncated_normal([100, output_dim]) / 100
                )
                self.w_t['l4b'] = l4b = tf.Variable(
                    tf.constant(0.01, shape=[output_dim]))
                self.pred_t = tf.matmul(L4, l4w) + l4b

            with tf.variable_scope('optimize'):
                optimizer = tf.train.AdamOptimizer(learning_rate=
                    self.lrate)
                # Clip gradient
                # gvs = optimizer.compute_gradients(self.loss, self.w)
                # capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
                # self.train_op = optimizer.apply_gradients(capped_gvs)

                # optimizer = tf.train.GradientDescentOptimizer(
                #     learning_rate=self.lrate
                # )

                # optimizer = tf.train.RMSPropOptimizer(learning_rate = self.lrate)
                self.train_op = optimizer.minimize(loss=self.loss,
                    global_step=global_step)
                self.init_op = tf.global_variables_initializer()
            with tf.variable_scope('assign'):
                for key in self.w.keys():
                    self.assign_op[key] = tf.assign(self.w_t[key], self.w[key])
            self.saver = tf.train.Saver()
        self._graph.finalize()

    def fit(self, states, actions, targets):
        with self._graph.as_default():
            feed_dict = {
                'states:0':states.reshape(list(np.shape(states)) + [1]),
                'actions:0':[self.a_list.index(a) for a in actions],
                'targets:0':targets,
            }
            self._sess.run(self.train_op, feed_dict)

    def update(self):
        for key in self.w.keys():
            self._sess.run(self.assign_op[key])

    # evaluate the trained network
    def Q(self, state):
        feed_dict = {
            'states:0': np.array(state).reshape([1] + list(np.shape(state)) + [1]),
        }
        return self._sess.run(self.pred, feed_dict)[0]

    # evaluate the target network
    def Qhat(self, state):
        feed_dict = {
            'states:0': np.array(state).reshape([1] + list(np.shape(state)) + [1]),
        }
        return self._sess.run(self.pred_t, feed_dict)[0]

    def save(self):
        self.saver.save(self._sess, self._path + self._name)

class memory(list):
    def __init__(self,length):
        self.length = length
        super().__init__()
    def add(self,x):
        if len(self) < self.length:
            super().append(x)
        elif len(self) == self.length:
            super().pop(0)
            super().append(x)
    def sample(self,size):
        output = np.random.choice(self,size)
        return output

class dqn_agent:
    '''
        The agent can play the game and train it self using DQN.
    '''
    def __init__(self, params, load = False):
        self._path = 'trained_agents/'
        if load:
            self.load(**params)
        else:
            self.new(**params)
        self.Q = self.model.Q
        self.Qhat = self.model.Qhat
        self.ep = lambda x: self.ep_start - min(self.ep_rate *
            (max(x - self.learn_starts, 0)), self.ep_start - self.ep_end)
        self.game_para = {'size': 4, 'odd_2': 0.9}

    def new(self, name, N, shape, ep_start, ep_end, ep_rate, batch_size,
        a_list, C, lrate, gamma = 1, learn_starts = 5):
        self.name = name
        self.shape = shape
        self.ep_start = ep_start
        self.ep_end = ep_end
        self.ep_rate = ep_rate
        self.batch_size = batch_size
        self.a_list = a_list
        self.C = C
        self.lrate = lrate
        self.gamma = gamma
        self.learn_starts = learn_starts

        self.D = memory(N)
        self.nA = len(a_list)
        self.model = nn_model(shape, a_list, name, lrate) # implement two networks in one model with an update method.
        self.iter = 0
        self.episode = 0

    def load(self, name):
        with open(self._path + name + '-params','rb') as f:
            params = pickle.load(f)
            self.name = name
            self.shape = params['shape']
            self.ep_start = params['ep_start']
            self.ep_end = params['ep_end']
            self.ep_rate = params['ep_rate']
            self.batch_size = params['batch_size']
            self.a_list = params['a_list']
            self.C = params['C']
            self.lrate = params['lrate']
            self.gamma = params['gamma']
            self.learn_starts = params['learn_starts']

            self.nA = params['nA']
            self.iter = params['iter']
            self.episode = params['episode']
        with open(self._path + name + '-memory','rb') as f:
            self.D = pickle.load(f)
        self.model = nn_model(self.shape, self.a_list, name, self.lrate, load=True)

    def save(self):
        with open(self._path + self.name + '-memory','wb') as f:
            pickle.dump(self.D, f)
        with open(self._path + self.name + '-params', 'wb') as f:
            params = {
                'shape': self.shape,
                'ep_start': self.ep_start,
                'ep_end': self.ep_end,
                'ep_rate': self.ep_rate,
                'batch_size': self.batch_size,
                'a_list': self.a_list,
                'C': self.C,
                'lrate': self.lrate,
                'gamma': self.gamma,
                'learn_starts': self.learn_starts,
                'nA': self.nA,
                'iter': self.iter,
                'episode': self.episode,
            }
            pickle.dump(params, f)
        self.model.save()

    def perceive(self, observation):
        self.iter += 1
        if observation['done']:
            self.episode += 1
        self.D.add(observation)

    def train_action(self, state):
        return self.a_list[ep_greedy(self.Q, state, self.ep(self.episode))]

    def play_action(self, state):
        return self.a_list[greedy(self.Q, state)]

    def train(self):
        if self.episode < self.learn_starts:
            return 0
        batch = self.D.sample(self.batch_size)
        y = np.empty(self.batch_size)
        s_batch = []
        a_batch = []
        for idx in range(self.batch_size):
            if batch[idx]['done']:
                y[idx] = batch[idx]['r']
            else:
                y[idx] = batch[idx]['r'] + \
                    self.gamma * max(self.Qhat(batch[idx]['ss']))
            s_batch += [batch[idx]['s']]
            a_batch += [batch[idx]['a']]
        self.model.fit(np.array(s_batch), np.array(a_batch), y)
        if self.iter % 100 == 0:
            loss = self.model.get_loss(np.array(s_batch), np.array(a_batch), y)
            print('episode: {}, loss: {:<.4}  '.format(
                        self.episode, loss))
        if self.iter % self.C == self.C - 1:
            self.model.update()
