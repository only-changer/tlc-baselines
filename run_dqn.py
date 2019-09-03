import gym
from environment import TSCEnv
from world import World
from generator import LaneVehicleGenerator
from agent import DQNAgent
from metric import TravelTimeMetric
import argparse
import os

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('config_file', type=str, help='path of config file')
parser.add_argument('--thread', type=int, default=1, help='number of threads')
parser.add_argument('--steps', type=int, default=100, help='number of steps')
parser.add_argument('--delta_t', type=int, default=1, help='how often agent make decisions')
parser.add_argument('--episodes', type=int, default=500, help='training episodes')
parser.add_argument('--save_model', action="store_true", default=False)
parser.add_argument("--save_rate", type=int, default=10, help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir', type=str, default="model/dqn", help='directory in which model should be saved')
args = parser.parse_args()

# create world
world = World(args.config_file, thread_num=args.thread)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i["trafficLight"]["lightphases"]))
    agents.append(DQNAgent(
        action_space,
        LaneVehicleGenerator(world, i["id"], ["lane_count"], in_only=True, average=None),
        LaneVehicleGenerator(world, i["id"], ["lane_waiting_count"], in_only=True, average="all", negative=True)
    ))

# create metric
metric = TravelTimeMetric(world)

# create env
env = TSCEnv(world, agents, metric)

# train dqn_agent
def train(episodes=args.episodes):
    for e in range(episodes):
        last_obs = env.reset()
        episodes_rewards = [0 for i in agents]
        episodes_num_decisions = 0
        for i in range(args.steps):
            if i % args.delta_t == 0:
                actions = []
                for agent_id, agent in enumerate(agents):
                    actions.append(agent.get_action(last_obs[agent_id]))
                obs, rewards, dones, info = env.step(actions)
                # print(obs)
                # print(actions)
                for agent_id, agent in enumerate(agents):
                    agent.remember(last_obs[agent_id], actions[agent_id], rewards[agent_id], obs[agent_id])
                    episodes_rewards[agent_id] += rewards[agent_id]
                episodes_num_decisions += 1
                last_obs = obs

            total_time = i + e * args.steps
            for agent_id, agent in enumerate(agents):
                if total_time > agent.learning_start and total_time % agent.update_model_freq == agent.update_model_freq - 1:
                    agent.replay()
                if total_time > agent.learning_start and total_time % agent.update_target_model_freq == agent.update_target_model_freq - 1:
                    agent.update_target_network()
            if all(dones):
                break
        if e % args.save_rate == 0:
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            for agent in agents:
                agent.save_model(args.save_dir)
        print("episode:{}/{}".format(e, episodes))
        for agent_id, agent in enumerate(agents):
            print("agent:{}, mean_episode_reward:{}".format(agent_id, episodes_rewards[agent_id]/episodes_num_decisions))

def test():
    obs = env.reset()
    for agent in agents:
        agent.load_model(args.save_dir)
    for i in range(args.steps):
        if i % args.delta_t == 0:
            actions = []
            for agent_id, agent in enumerate(agents):
                actions.append(agent.get_action(obs[agent_id]))
            obs, rewards, dones, info = env.step(actions)

        if all(dones):
            break
    print("Final Travel Time is %.4f" % env.metric.update(done=True))

# simulate
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
# train()
test()
