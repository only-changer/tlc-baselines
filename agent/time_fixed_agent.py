from . import BaseAgent
import math

class Time_Fixed_Agent(BaseAgent):
	def __init__(self, action_space):  # cycle length set to 10s default
		super().__init__(action_space)
		self.cycle_length = 10.0

	def get_ob(self):
		return 0

	def get_reward(self):
		return 0

	def get_action(self, world):
		current_time = world.eng.get_current_time()
		action_num = len(self.action_space)
		cycle_num = int(current_time/cycle_length)
		action = cycle_num%action_num

		return self.action_space[action]