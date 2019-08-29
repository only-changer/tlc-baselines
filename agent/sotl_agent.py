from . import BaseAgent

class SOTLAgent(BaseAgent):
    """
           Agent using Self-organizing Traffic Light(SOTL) Control method to control traffic light
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
        self.t_min = 30

        # some threshold to deal with phase requests
        self.min_green_vehicle = 20
        self.max_red_vehicle = 30

        self.phase_available_laneLinks = self.world.intersection_roadLinks[self.iid]["phase_available_laneLinks"]
        self.phase_startLane_mapping = []
        for phase_id, links in enumerate(self.phase_available_laneLinks):
            self.phase_startLane_mapping.append([])
            for laneLink in links:
                self.phase_startLane_mapping[phase_id].append(laneLink[0])

        self.current_phase_time = self.t_min
        self.last_phase = 1

    def get_ob(self):
        return self.ob_generator.generate()

    def get_action(self, ob):
        lanes_waiting_count = ob

        # observation must be the waiting vehicles in each starting lane
        assert len(lanes_waiting_count) == len(self.world.intersection_roadLinks[self.iid]["roadLinks"])

        if self.current_phase_time >= self.t_min:
            num_waiting_green_vehicle, num_waiting_red_vehicle = 0, 0
            for startLane in self.phase_startLane_mapping[self.last_phase]:
                num_waiting_green_vehicle += lanes_waiting_count[self.lanes.index(startLane)]
            for startLane in self.world.intersection_roadLinks[self.iid]["startLanes"]:
                num_waiting_red_vehicle += lanes_waiting_count[self.lanes.index(startLane)]
            num_waiting_red_vehicle -= num_waiting_green_vehicle

            if num_waiting_green_vehicle <= self.min_green_vehicle and num_waiting_red_vehicle > self.max_red_vehicle:
                self.last_phase = self.last_phase % (len(self.phase_list) - 1) + 1
                self.current_phase_time = self.t_min

        self.current_phase_time += self.delta_t
        return self.last_phase

    def get_reward(self):
        return