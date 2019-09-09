from . import BaseAgent
import math

class Fixedtime_Agent(BaseAgent):
    def __init__(self, action_space, phases_duration):
        super().__init__(action_space)
        self.phases_duration = phases_duration
        cycle_time = 0
        for phase in self.phases_duration:
            cycle_time += phase[1]
        self.cycle_time = cycle_time
        self.last_action = -1

    def get_ob(self):
        return 0

    def get_reward(self):
        return 0

    def get_action(self, world):
        current_time = world.eng.get_current_time()
        operator_time = current_time
        while operator_time >= self.cycle_time:
            operator_time -= self.cycle_time
        index = 0
        while operator_time > self.phases_duration[index][1]:
            operator_time -= self.phases_duration[index][1]
            index = (index+1)%(len(self.phases_duration))
        action = self.phases_duration[index][0]
        self.last_action = action
        print('time',current_time, 'action', action)

        return action