import sys,time
# from tiles3 import IHT, tiles
import gym
from gym.envs.registration import register
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from model import nn_model
from utils import ep_greedy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
MAX_EP = 2000

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

def dqn(importance_sample, frame_repeats, M, N, env, ep_start, ep_end, ep_rate, batch_size,
        gamma, a_list, C, lrate, criteria, test_episodes=100,
        learn_starts = 5):
    nA = len(a_list)
    D = memory(N)
    S = memory(M)
    nS = env.observation_space.shape[0]
    model = nn_model(nS, a_list, 'test1', lrate) # implement two networks in one model with an update method.
    Q = model.Q
    Qhat = model.Qhat
    score_list = []
    iter = 0
    episode = 0
    while True:
        s = env.reset()
        done = False
        ep = ep_start - min(ep_rate * (max(episode - learn_starts, 0)), ep_start - ep_end)
        while not done:
            if iter % frame_repeats == 0:
                a = ep_greedy(Q, s, ep)
            ss, r, done, _ = env.step(a)
            if importance_sample and r >= 0:
                S.add({'s':s,'a':a,'r':r,'ss':ss,'done':done})
            else:
                D.add({'s':s,'a':a,'r':r,'ss':ss,'done':done})
            s = ss
            if episode < learn_starts:
                iter += 1
                continue
            if importance_sample and len(S) > 0:
                batch = D.sample(batch_size - 1)
                batch += S.sample(1)
            else:
                batch = D.sample(batch_size)
            y = np.empty(batch_size)
            s_batch = []
            a_batch = []
            for idx in range(batch_size):
                if batch[idx]['done']:
                    y[idx] = batch[idx]['r']
                else:
                    y[idx] = batch[idx]['r'] + \
                        gamma * max(Qhat(batch[idx]['ss']))
                s_batch += [batch[idx]['s']]
                a_batch += [batch[idx]['a']]
            model.fit(np.array(s_batch), np.array(a_batch), y)
            if iter % 100 == 0:
                loss = model.get_loss(np.array(s_batch), np.array(a_batch), y)
                print('\rsuccess: {}, episode: {}, loss: {:<.4}  '.format(
                            len(S), episode, loss),end='')
                sys.stdout.flush()
            if iter % C == C - 1:
                model.update()
            iter += 1
        if episode > learn_starts and episode % 50 == 0:
            print('')
            score = eval_perform(model, env, test_episodes)
            score_list += [score]
            if score > criteria:
                break
        episode += 1
        if episode > MAX_EP:
            print('exceed max episode')
            break
    model.save()
    return score_list, model

def eval_perform(agent, env, episodes):
    avg_score = 0
    for idx in range(episodes):
        score = 0
        done = False
        s = env.reset()
        while not done:
            # env.render()
            a = ep_greedy(agent.Q, s, 0.)
            ss, r, done, _ = env.step(a)
            score += r
            s = ss
        avg_score = (avg_score * idx + score) / (idx + 1)
    print('avg_score: {}'.format(avg_score))
    return avg_score

def plot_dqn(num_steps, name, suffix):
    fig = plt.figure()
    plt.plot(np.array(num_steps))
    plt.title('Performance of DQN on '+name)
    plt.xlabel('per 50 episodes')
    plt.ylabel('score in testing')
    plt.savefig(name+'-'+suffix+'C_1.eps')
    plt.close()

if __name__ == '__main__':
    frame_repeats = int(sys.argv[1])
    importance_sample = bool(int(sys.argv[2]))
    suffix = sys.argv[3]
    np.random.seed(3)
    name = 'MountainCar-v0'
    env = gym.make(name)
    N = 10000
    M = 100
    criteria = -110
    ep_start = 1.
    ep_end = 0.1
    ep_rate = 0.005
    batch_size = 32
    gamma = 1.
    a_list = [0,1,2]
    C = 1
    lrate = 0.001

    t1 = time.process_time()
    score_list, agent = dqn(importance_sample, frame_repeats, M, N, env,
        ep_start, ep_end, ep_rate, batch_size, gamma, a_list, C, lrate, criteria)
    t2 = time.process_time()
    print('training time:', t2-t1)
    plot_dqn(score_list, name, suffix)
    # Call the trained agent to test.
    # agent = nn_model(2, a_list, 'test1', lrate, load=True)
    # eval_perform(agent, env, 10)
