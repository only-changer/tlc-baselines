import json
import os.path as osp
import cityflow

import numpy as np
from math import atan2, pi
import sys

class World:
    """
    Create a CityFlow engine and maintain informations about CityFlow world
    """
    def __init__(self, cityflow_config, thread_num):
        print("building world...")
        self.eng = cityflow.Engine(cityflow_config, thread_num=thread_num)
        self.roadnet = self._get_roadnet(cityflow_config)
        self.RIGHT = True # vehicles moves on the right side, currently always set to true due to CityFlow's mechanism

        # get all non virtual intersections
        self.intersections = [i for i in self.roadnet["intersections"] if not i["virtual"]]
        self.intersection_ids = [i["id"] for i in self.intersections]

        # get incoming and outgoing roads of each intersection, clock-wise order from North
        print("parsing road networks...")
        self.intersection_roads = {
            i_id: {
                "roads": [],
                "outs": [],
                "directions": []
            } for i_id in self.intersection_ids
        }

        # get roadLinks and phase information of each intersection
        self.intersection_roadLinks = {
            i_id: {
                "roadLinks": [],
                "startLanes": [],
                "laneLinks":[],
                "phase_available_roadLinks": [],
                "phase_available_laneLinks": []
            } for i_id in self.intersection_ids
        }

        # id of all roads and lanes
        self.all_roads = []
        self.all_lanes = []

        for road in self.roadnet["roads"]:
            self.all_roads.append(road["id"])
            i = 0
            for lane in road["lanes"]:
                self.all_lanes.append(road["id"] + "_" + str(i))
                i += 1

        def insert_road(iid, road, out):

            def get_direction(road, out=True):
                if out:
                    x = road["points"][1]["x"] - road["points"][0]["x"]
                    y = road["points"][1]["y"] - road["points"][0]["y"]
                else:
                    x = road["points"][-2]["x"] - road["points"][-1]["x"]
                    y = road["points"][-2]["y"] - road["points"][-1]["y"]
                tmp = atan2(x, y)
                return tmp if tmp >= 0 else (tmp + 2*pi)

            if iid in self.intersection_roads:
                roads = self.intersection_roads[iid]
                roads["roads"].append(road)
                roads["outs"].append(out)
                roads["directions"].append(get_direction(road, out))

        for road in self.roadnet["roads"]:
            insert_road(road["startIntersection"], road, True)
            insert_road(road["endIntersection"], road, False)

        for iid in self.intersection_ids:
            roads = self.intersection_roads[iid]
            directions = roads["directions"]
            outs = roads["outs"]
            order = sorted(range(len(roads["roads"])), key=lambda i: (directions[i], outs[i] if self.RIGHT else not outs[i]))
            roads["roads"] = [roads["roads"][i] for i in order]
            roads["directions"] = [directions[i] for i in order]
            roads["outs"] = [outs[i] for i in order]
            roads["out_roads"] = [roads["roads"][i] for i, x in enumerate(roads["outs"]) if x]
            roads["in_roads"] = [roads["roads"][i] for i, x in enumerate(roads["outs"]) if not x]

        for intersection in self.roadnet["intersections"]:
            if intersection["virtual"]:
                continue
            iid = intersection["id"]
            links = self.intersection_roadLinks[iid]
            for roadLink in intersection["roadLinks"]:
                links["roadLinks"].append([roadLink["startRoad"], roadLink["endRoad"]])
                for laneLink in roadLink["laneLinks"]:
                    startLane = roadLink["startRoad"] + "_" + str(laneLink["startLaneIndex"])
                    links["startLanes"].append(startLane)
                    endLane = roadLink["endRoad"] + "_" + str(laneLink["endLaneIndex"])
                    links["laneLinks"].append([startLane, endLane])
            for phase_id, phase in enumerate(intersection["trafficLight"]["lightphases"]):
                links["phase_available_laneLinks"].append([])
                links["phase_available_roadLinks"].append(phase["availableRoadLinks"])
                for roadLink_id in phase["availableRoadLinks"]:
                    roadLink = intersection["roadLinks"][roadLink_id]
                    for laneLink in roadLink["laneLinks"]:
                        startLane = roadLink["startRoad"] + "_" + str(laneLink["startLaneIndex"])
                        endLane = roadLink["endRoad"] + "_" + str(laneLink["endLaneIndex"])
                        links["phase_available_laneLinks"][phase_id].append([startLane, endLane])
            links["startLanes"] = list(set(links["startLanes"]))

        print("road network parsed.")
        print("world built.")

        # initializing info functions
        self.info_functions = {
            "lane_count": self.eng.get_lane_vehicle_count,
            "lane_waiting_count": self.eng.get_lane_waiting_vehicle_count,
            "lane_vehicles": self.eng.get_lane_vehicles,
            "pressure": self.get_lane_pressure,
            "time": self.eng.get_current_time
        }
        self.fns = []
        self.info = {}

    def _get_roadnet(self, cityflow_config):
        with open(cityflow_config) as f:
            cityflow_config = json.load(f)
        roadnet_file= osp.join(cityflow_config["dir"], cityflow_config["roadnetFile"])
        with open(roadnet_file) as f:
            roadnet = json.load(f)
        return roadnet

    def get_lane_pressure(self):
        lane_pressures = {}
        for lane in self.all_lanes:
            lane_pressures[lane] = 0
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        for iid in self.intersection_ids:
            links = self.intersection_roadLinks[iid]
            for laneLink in links["laneLinks"]:
                startlane = laneLink[0]
                endlane = laneLink[1]
                lane_pressures[startlane] += lane_vehicle_count[startlane] - lane_vehicle_count[endlane]
        return lane_pressures

    def subscribe(self, fns):
        if isinstance(fns, str):
            fns = [fns]
        for fn in fns:
            if fn in self.info_functions:
                if not fn in self.fns:
                    self.fns.append(fn)
            else:
                raise Exception("info function %s not exists" % fn)

    def step(self):
        self.eng.next_step()
        self._update_infos()

    def reset(self):
        self.eng.reset()
        self._update_infos()

    def _update_infos(self):
        self.info = {}
        for fn in self.fns:
            self.info[fn] = self.info_functions[fn]()

    def get_info(self, info):
        return self.info[info]


if __name__ == "__main__":
    world = World("examples/config.json", thread_num=1)