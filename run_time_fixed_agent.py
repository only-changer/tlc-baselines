import gym
from environment import TSCEnv
from world import World
from agent import Time_Fixed_Agent
from metric import TravelTimeMetric
import argparse

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, help='number of threads')
parser.add_argument('--steps', type=int, default=100, help='number of steps')
args = parser.parse_args()

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i["trafficLight"]["lightphases"]))
    agents.append(Time_Fixed_Agent(action_space))

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)

# simulate
obs = env.reset()
for i in range(args.steps):
    if i % 5 == 0:
        actions = env.action_space.sample()
        print('actions',actions)
    obs, rewards, dones, info = env.step(actions)
    print(info["metric"])

print("Final Travel Time is %.4f" % env.metric.update(done=True))