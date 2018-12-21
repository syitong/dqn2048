import numpy as np
import tensorflow as tf
from utils import greedy, ep_greedy

def clipped_error(x):
    # Huber loss
    try:
      return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
      return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

class nn_model:
    def __init__(self, nS, a_list, name, lrate, load = False):
        self.lrate = lrate
        self._name = name
        self._path = 'trained_agents/'
        self.nS = nS
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
                'states:0':states,
                'actions:0':actions,
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
            s = tf.placeholder(dtype=tf.float32, shape=[None, self.nS], name='states')
            a = tf.placeholder(dtype=tf.uint8, shape=[None], name='actions')
            y = tf.placeholder(dtype=tf.float32, shape=[None], name='targets')
            onehot = tf.one_hot(a, depth=len(self.a_list))
            output_dim = len(self.a_list)
            with tf.variable_scope('train'):
                self.w['l1w'] = l1w = tf.Variable(
                    tf.truncated_normal((self.nS,64)))
                self.w['l1b'] = l1b = tf.Variable(
                    tf.constant(0.01, shape=[64]))
                L1 = tf.nn.relu(tf.matmul(s, l1w) + l1b)
                self.w['l2w'] = l2w = tf.Variable(
                    tf.truncated_normal((64,output_dim)))
                self.w['l2b'] = l2b = tf.Variable(
                    tf.constant(0.01, shape=[output_dim]))
                self.pred = tf.matmul(L1,l2w) + l2b
                q_act = tf.reduce_sum(self.pred * onehot, reduction_indices=1)
                self.loss = tf.reduce_mean(
                    (q_act  - y)**2) \
                    # + self.lambda_ * (tf.norm(l1w) + tf.norm(l4w))
            with tf.variable_scope('target'):
                self.w_t['l1w'] = l1w = tf.Variable(
                    tf.constant(0., shape=[self.nS,64]))
                self.w_t['l1b'] = l1b = tf.Variable(
                    tf.constant(0., shape=[64]))
                L1 = tf.nn.relu(tf.matmul(s, l1w) + l1b)
                self.w_t['l2w'] = l2w = tf.Variable(
                    tf.constant(0., shape=[64,output_dim]))
                self.w_t['l2b'] = l2b = tf.Variable(
                    tf.constant(0., shape=[output_dim]))
                self.pred_t = tf.matmul(L1,l2w) + l2b
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
                'states:0':states,
                'actions:0':actions,
                'targets:0':targets,
            }
            self._sess.run(self.train_op, feed_dict)

    def update(self):
        for key in self.w.keys():
            self._sess.run(self.assign_op[key])

    # evaluate the trained network
    def Q(self, state):
        feed_dict = {
            'states:0': np.array(state).reshape((1,-1)),
        }
        return self._sess.run(self.pred, feed_dict)[0]

    # evaluate the target network
    def Qhat(self, state):
        feed_dict = {
            'states:0': np.array(state).reshape((1,-1)),
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

    def new(self, name, N, nS, ep_start, ep_end, ep_rate, batch_size,
        a_list, C, lrate, gamma = 1, learn_starts = 5):
        self.name = name
        self.nS = nS
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
        self.model = nn_model(nS, a_list, name, lrate) # implement two networks in one model with an update method.
        self.iter = 0
        self.episode = 0

    def load(self, name):
        with open(self._path + name + '-params','rb') as f:
            params = pickle.load(f)
            self.name = name
            self.nS = params['nS']
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
        self.model = nn_model(self.nS, self.a_list, name, self.lrate, load=True)

    def save(self):
        with open(self._path + self.name + '-memory','wb') as f:
            pickle.dump(self.D, f)
        with open(self._path + self.name + '-params', 'wb') as f:
            params = {
                'nS': self.nS,
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
        for idx in range(batch_size):
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
            print('\repisode: {}, loss: {:<.4}  '.format(
                        self.episode, loss),end='')
            sys.stdout.flush()
        if self.iter % self.C == self.C - 1:
            self.model.update()
        self.iter += 1
