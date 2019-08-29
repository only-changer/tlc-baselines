import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import SOTLAgent
from metric import TravelTimeMetric
import argparse

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=100, help='number of steps')
args = parser.parse_args()

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i["trafficLight"]["lightphases"]))
    agents.append(SOTLAgent(
        action_space,
        LaneVehicleGenerator(world, i["id"], ["lane_waiting_count"], in_only=True, average=None),
    ))

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)

# simulate
obs = env.reset()
actions = []
for i in range(args.steps):
    actions = []
    for i, agent in enumerate(agents):
        actions.append(agent.get_action(obs[i]))
    obs, rewards, dones, info = env.step(actions)
    #print(obs)
    #print(rewards)
    # print(info["metric"])

print("Final Travel Time is %.4f" % env.metric.update(done=True))