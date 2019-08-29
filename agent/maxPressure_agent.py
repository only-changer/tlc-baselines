from . import BaseAgent

class MaxPressureAgent(BaseAgent):
    """
       Agent using Max-Pressure method to control traffic light
       action space must be equal to phase space

       Parameters
       ----------
       delta_t : the time interval between agent's decisions of actions
       """
    def __init__(self, action_space, ob_generator, delta_t=1):
        super().__init__(action_space)
        self.ob_generator = ob_generator
        self.world = self.ob_generator.world
        self.iid = self.ob_generator.iid
        self.delta_t = delta_t

        # the incoming lanes of this intersection
        self.lanes = []
        for road_lanes in self.ob_generator.lanes:
            for lane in road_lanes:
                self.lanes.append(lane)

        self.phase_list = range(action_space.n)

        # the minimum duration of time of one phase
        self.t_min = 20

        self.current_phase_time = self.t_min
        self.last_phase = 0

    def get_ob(self):
        return self.ob_generator.generate()

    def get_action(self, ob):
        lanes_pressure = ob

        # observation must be the pressure of each starting lane
        assert len(lanes_pressure) == len(self.world.intersection_roadLinks[self.iid]["roadLinks"])

        if self.current_phase_time < self.t_min:
            self.current_phase_time += self.delta_t
            return self.last_phase

        max_pressure = self._get_phase_pressure(self.phase_list[1], lanes_pressure)
        max_pressure_id = self.phase_list[1]
        for phase_id in self.phase_list:
            # 0 means yellow light
            if phase_id == 0:
                continue
            pressure = self._get_phase_pressure(phase_id, lanes_pressure)
            if pressure > max_pressure:
                max_pressure_id = phase_id
                max_pressure = pressure

        self.last_phase = max_pressure_id
        self.current_phase_time = self.delta_t

        return max_pressure_id

    def get_reward(self):
        return

    def _get_phase_pressure(self, phase_id, lanes_pressure):
        pressure = 0
        laneLinks = self.world.intersection_roadLinks[self.iid]["phase_available_laneLinks"][phase_id]
        for available_laneLink in laneLinks:
            start_lane = available_laneLink[0]
            lane_pressure = lanes_pressure[self.lanes.index(start_lane)]
            pressure += lane_pressure
        return pressure