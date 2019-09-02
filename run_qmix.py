import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import QMix_Agent
from metric import TravelTimeMetric
import argparse
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.activations import elu

# Q-mix network, computing mixed Q
class QMIXNet():
    def __init__(self, num_agents, state_size, hidden_size):
        super(QMIXNet, self).__init__()

        self.num_agents = num_agents
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.learning_rate = 0.001
        self.sample_batch_size  = 32
        self.memory = deque(maxlen=2000)
        self.state = [[]]

        self.hyper_net1 = Sequential()
        self.hyper_net1.add(Dense(self.num_agents*self.hidden_size, input_dim=self.num_agents*self.state_size, activation='linear'))
        self.hyper_net1.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.hyper_net2 = Sequential()
        self.hyper_net2.add(Dense(self.hidden_size, input_dim=self.num_agents*self.state_size, activation='linear'))
        self.hyper_net2.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def get_qtot(self, q_n, global_state):
        w1 = np.abs(self.hyper_net1.predict(global_state))
        w2 = np.abs(self.hyper_net2.predict(global_state))

        self.state = global_state
        self.w1 = w1
        self.w2 = w2

        w1 = w1.reshape(self.num_agents, self.hidden_size)
        w2 = w2.reshape(self.hidden_size, 1)

        q_tot = np.dot(q_n,w1)
        q_tot = np.dot(q_tot,w2)

        return q_tot

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=2, help='number of threads')
parser.add_argument('--steps', type=int, default=20, help='number of steps')
args = parser.parse_args()

# create world
world = World(args.config_file, thread_num=args.thread)

# get state size
max_state_size = 0
for iid in world.intersection_ids:
    in_road_num = len(world.intersection_roads[iid]["in_roads"])
    if in_road_num > max_state_size:
        max_state_size = in_road_num

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i["trafficLight"]["lightphases"]))
    action_size = len(i["trafficLight"]["lightphases"])
    agents.append(QMix_Agent(
        action_space,
        action_size,
        max_state_size,
        LaneVehicleGenerator(world, i["id"], ["lane_count"], in_only=True, average="road"),
        LaneVehicleGenerator(world, i["id"], ["lane_waiting_count"], in_only=True, average="all", negative=True)
    ))

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)

# simulate
obs = env.reset()
qmixnet = QMIXNet(len(agents), max_state_size, max_state_size)
for i in range(args.steps):
    if i % 5 == 0:
        # Get ob, q and actions from agents
        obs = [[]]
        qns = []
        actions = []
        for agent in agents:
            ob = agent.get_ob()
            obs = np.concatenate([obs,ob],axis=1)
            qns.append(np.max(agent.get_value(ob)))
            actions.append(agent.get_action(ob))
        qns = np.reshape(qns, [1, len(agents)])
        q_tot = qmixnet.get_qtot(qns, obs)
        print('action ',actions)
        #print('qtot: ',q_tot)

    # step
    obs, rewards, dones, info = env.step(actions)

    # Update network weights
    index = 0
    for agent in agents:
        next_ob = obs[index]
        next_ob = np.reshape(next_ob, [1, agent.state_size])
        reward = rewards[index]
        reward += q_tot[0]
        done = dones[index]
        agent.remember(agent.ob, agent.action, reward, next_ob, done)
        agent.replay(agent.sample_batch_size)
        index += 1

    qmixnet.hyper_net1.fit(qmixnet.state, qmixnet.w1, epochs=1, verbose=0)
    qmixnet.hyper_net2.fit(qmixnet.state, qmixnet.w2, epochs=1, verbose=0)
    
    #print(obs)
    #print(rewards)
    #print(info["metric"])

print("Final Travel Time is %.4f" % env.metric.update(done=True))