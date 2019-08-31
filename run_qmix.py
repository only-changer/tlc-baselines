import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import QMix_Agent
from metric import TravelTimeMetric
import argparse

def MixingNetwork(QList):
    QMix = []
    return QMix

def get_actions(QMix):
    actions = []
    return actions

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=100, help='number of steps')
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
for i in range(args.steps):
    if i % 5 == 0:
        '''
        QList = []
        for agent in agents:
            QList.append(agent.get_Qfunc())
        QMix = MixingNetwork(QList)
        actions = get_actions(QMix)
        '''
        actions = []
        for agent in agents:
            actions.append(agent.get_action(agent.get_ob()))
        #print(actions)

    obs, rewards, dones, info = env.step(actions)

    for agent in agents:
        agent.remember(agent.ob, agent.action, agent.get_reward())
        agent.replay(agent.sample_batch_size)

    #print(obs)
    #print(rewards)
    #print(info["metric"])
    
for agent in agents:
    agent.save_model()

print("Final Travel Time is %.4f" % env.metric.update(done=True))