import gym
import os
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import QMix_Agent
from metric import TravelTimeMetric
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Q-mix network, computing mixed Q
class QMIXNet():
    def __init__(self, num_agents, state_size, hidden_size):
        super(QMIXNet, self).__init__()

        self.num_agents = num_agents
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.learning_rate = 0.005

        self.hyper_net_w1 = Sequential()
        self.hyper_net_w1.add(Dense(self.num_agents*self.hidden_size, input_dim=self.num_agents*self.state_size, activation='linear'))
        self.hyper_net_w1.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.hyper_net_b1 = Sequential()
        self.hyper_net_b1.add(Dense(self.hidden_size, input_dim=self.num_agents*self.state_size, activation='linear'))
        self.hyper_net_b1.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.hyper_net_w2 = Sequential()
        self.hyper_net_w2.add(Dense(self.hidden_size, input_dim=self.num_agents*self.state_size, activation='linear'))
        self.hyper_net_w2.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.hyper_net_b2 = Sequential()
        self.hyper_net_b2.add(Dense(1, input_dim=self.num_agents*self.state_size, activation='relu'))
        self.hyper_net_b2.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def get_qtot(self, q_n, global_state):
        state = []
        for i in range(self.num_agents):
            for j in range(self.state_size):
                state.append(global_state[i][j])
        state = np.reshape(np.array(state),[1,-1])

        w1 = np.abs(self.hyper_net_w1.predict(state))
        w1 = w1.reshape(self.num_agents, self.hidden_size)
        b1 = self.hyper_net_b1.predict(state)
        w2 = np.abs(self.hyper_net_w2.predict(state))
        w2 = w2.reshape(self.hidden_size, 1)
        b2 = self.hyper_net_b2.predict(state)

        self.state = state
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

        q_tot = np.dot(q_n,w1)+b1
        q_tot = np.maximum(0,q_tot)
        q_tot = np.dot(q_tot,w2)+b2

        return q_tot

    def update(self):
        raise Exception("Not implemented yet.")

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=5, help='number of threads')
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
        max_state_size,
        LaneVehicleGenerator(world, i["id"], ["lane_count"], in_only=True, average="road"),
        LaneVehicleGenerator(world, i["id"], ["lane_waiting_count"], in_only=True, average="all", negative=True)
    ))

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)

# create qmixnet
qmixnet = QMIXNet(len(agents), max_state_size, max_state_size)

# utils for ob reform
def reform(obs, state_size):
    for index, ob in enumerate(obs):
        formed_ob = ob.tolist()
        for i in range(state_size - len(ob)):
            formed_ob.append(0)
        formed_ob = np.array(formed_ob)
        obs[index] = formed_ob

# train
def train():
    last_obs = env.reset()
    reform(last_obs, max_state_size)
    for i in range(args.steps):
        if i % 5 == 0:
            actions = []
            qs = []
            for agent_id, agent in enumerate(agents):
                actions.append(agent.get_action(last_obs[agent_id]))
                qs.append(np.max(agent.get_value(last_obs[agent_id])))
            qs = np.reshape(qs, [1, -1])
            qtot = qmixnet.get_qtot(qs, last_obs)
            print('action ',actions)

        # step
        obs, rewards, dones, info = env.step(actions)
        reform(obs, max_state_size)
        for agent_id, agent in enumerate(agents):
            agent.remember(last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id])
        last_obs = obs
        if all(dones):
            break

        # Update agents net
        for agent_id, agent in enumerate(agents):
            if i > agent.learning_start:
                agent.replay()
                agent.update_target_network()
        # Update qmixnet
        qmixnet.update()

        # Note: Qmix net should be an end-to-end networks, current structure does not train qmix weights.
        # Need re-write in tf, not keras model.

    print("Final Travel Time is %.4f" % env.metric.update(done=True))
    # Save weights
    if not os.path.exists("examples/qmix_weights"):
        os.mkdir("examples/qmix_weights")
    for agent in agents:
        agent.save_model("examples/qmix_weights")

def test():
    obs = env.reset()
    reform(obs, max_state_size)
    for agent in agents:
        agent.load_model("examples/qmix_weights")
    for i in range(args.steps):
        if i % 5 == 0:
            actions = []
            for agent_id, agent in enumerate(agents):
                actions.append(agent.get_action(obs[agent_id]))
            obs, rewards, dones, info = env.step(actions)
            reform(obs, max_state_size)
        if all(dones):
            break
    print("Final Travel Time is %.4f" % env.metric.update(done=True))

if __name__ == '__main__':
    train()
    #test()