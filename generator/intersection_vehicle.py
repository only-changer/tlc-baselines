import numpy as np
from . import BaseGenerator

class IntersectionVehicleGenerator(BaseGenerator):
    """
    Generate State or Reward based on statistics of lane vehicles.

    Parameters
    ----------
    world : World object
    I : Intersection object
    fns : list of statistics to get, currently support "lane_count", "lane_waiting_count" and "pressure"
    in_only : boolean, whether to compute incoming lanes only
    average : None or str
        None means no averaging
        "own" means take only the intersection of that agent
        "all" means take average of all intersections
    negative : boolean, whether return negative values (mostly for Reward)
    """
    def __init__(self, world, I, fns, average=None, negative=False):
        self.world = world
        self.I = I
        self.intersections = world.intersections
        # subscribe functions
        self.world.subscribe(fns)
        self.fns = fns

        # calculate result dimensions
        size = len(self.intersections)
        if average == "all" or average == "own":
            size = 1
        self.ob_length = len(fns) * size

        self.average = average
        self.negative = negative

    def generate(self):
        results = [self.world.get_info(fn) for fn in self.fns]

        ret = np.array([])
        for i in range(len(self.fns)):
            result = results[i]
            fn_result = np.array([])
            if self.average == "own":
                fn_result = np.append(fn_result, result[self.I.id])
            ret = np.append(ret, fn_result)
        if self.negative:
            ret = ret * (-1)
        return ret
