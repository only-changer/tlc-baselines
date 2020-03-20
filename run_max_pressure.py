import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import MaxPressureAgent
from metric import TravelTimeMetric
import argparse
import logging
from datetime import datetime
import os
import numpy as np

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--delta_t', type=int, default=1, help='how often agent make decisions')
parser.add_argument('--log_dir', type=str, default="log/maxpressure", help='directory in which logs should be saved')

args = parser.parse_args()

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(args.log_dir, datetime.now().strftime('%Y%m%d-%H%M%S') + ".log"))
fh.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)

def test(path):
    # create world
    world = World(path, thread_num=args.thread)

    # create agents
    agents = []
    for i in world.intersections:
        action_space = gym.spaces.Discrete(len(i.phases))
        agents.append(MaxPressureAgent(
            action_space, i, world,
            LaneVehicleGenerator(world, i, ["lane_count"], in_only=True)
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
        for agent_id, agent in enumerate(agents):
            actions.append(agent.get_action(obs[agent_id]))
        obs, rewards, dones, info = env.step(actions)
        #print(world.intersections[0]._current_phase, end=",")
        # print(obs, actions)
        # print(env.eng.get_average_travel_time())
        #print(obs)
        #print(rewards)
        # print(info["metric"])

    logger.info("Final Travel Time is %.4f" % env.eng.get_average_travel_time())
    return env.eng.get_average_travel_time()

if __name__ == '__main__':
    # meta_train(args)
    # train(args)
    real_flow_path = []
    real_flow_floder = '/mnt/c/users/onlyc/desktop/work/RRL_TLC/real_flow_config/'
    for root, dirs, files in os.walk(real_flow_floder):
        for file in files:
            real_flow_path.append(real_flow_floder + file)
    logger.info("Meta Test Real")
    result = []
    for n in range(len(real_flow_path)):
        logger.info("Meta Test Env: %d" % n)
        t1 = test(real_flow_path[n])
        result.append(t1)
    logger.info(
        "Meta Test Result, Max: {}, Min: {}, Mean: {}".format(np.max(result), np.min(result), np.mean(result)))
    fake_flow_floder = '/mnt/c/users/onlyc/desktop/work/RRL_TLC/fake_flow_config/'
    w_dis = [0.005, 0.01, 0.05, 0.1]
    for w in w_dis:
        logger.info("Meta Test Fake with W Distance: %.4f" % w)
        fake_flow_path = []
        result = []
        for root, dirs, files in os.walk(fake_flow_floder + str(w) + '/'):
            for file in files:
                fake_flow_path.append(fake_flow_floder + str(w) + '/' + file)
        for n in range(len(fake_flow_path)):
            logger.info("Meta Test Env: %d" % n)
            t1 = test(fake_flow_path[n])
            result.append(t1)
        logger.info("Meta Test Result, Max: {}, Min: {}, Mean: {}".format(np.max(result), np.min(result), np.mean(result)))
