import json
import gym
from environment import TSCEnv
from world import World
from agent import Fixedtime_Agent
from metric import TravelTimeMetric
import argparse

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('phases_config_file', type=str, help='path of phases config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=100, help='number of steps')
args = parser.parse_args()

# create world
world = World(args.config_file, thread_num=args.thread)

# parse intersections config
config_valid = True
try:
    with open(args.phases_config_file) as f:
        phases_config = json.load(f)
except:
    config_valid = False
    raise Exception("phases config file load failed, will use default config")

# create agents
agents = []
if config_valid:
    for i in world.intersections:
        action_space = gym.spaces.Discrete(len(i["trafficLight"]["lightphases"]))
        agents.append(Fixedtime_Agent(action_space, phases_config[i['id']]))
else:
    raise Exception("default config not implemented error")
    

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)

# simulate
obs = env.reset()
for i in range(args.steps):
    if i % 5 == 0:
        actions = []
        for agent in agents:
            actions.append(agent.get_action(world))
        print('actions',actions)
    obs, rewards, dones, info = env.step(actions)
    print(info["metric"])

print("Final Travel Time is %.4f" % env.metric.update(done=True))