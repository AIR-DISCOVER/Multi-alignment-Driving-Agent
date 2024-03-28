# Copyright (c) # Copyright (c) 2018-2020 CVC.
# memory no coach
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

from flask import Flask, request
import threading
from collections import deque  # 用于方便地存储和更新最近5轮的反馈

app = Flask(__name__)
json_script = None


@app.route('/command', methods=['POST'])
def command():
    global json_script
    json_script = request.get_json()
    return '', 200


def start_server():
    app.run(port=5000)


server_thread = threading.Thread(target=start_server)
server_thread.start()

import random
import numpy as np
import carla
import json
import time
import openai
import queue
# import threading
import re
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from agents.navigation.behavior_types import Cautious, Aggressive, Normal

from agents.tools.misc import get_speed, positive, is_within_distance, compute_distance

import os
import csv
from datetime import datetime

os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"


class GPT4AgentInterface:
    def __init__(self):
        pass

    def receive_json(self, json_script):
        # 使用json模块解析JSON脚本
        self.commands = json.loads(json_script)

    def get_commands(self):
        # 返回解析后的命令供CARLA自动驾驶代码使用
        return self.commands


class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0
        self.stop_signalled = False

        openai.api_key = ""
        openai.organization = ""
        # conservative or risky
        self.driver_prompt_path = r""

        # # coach(非coach模式请注释下面的代码)
        # self.read_guideline = True
        # self.coach_prompt_path = r"C:\Users\gongj\Desktop\prompt\Driving Style\coach_aggressive.txt"
        # self.guideline_path = 'C:/Users/gongj/Desktop/log/guideline/feedback_aggressive.txt'
        # with open(self.guideline_path, 'a', newline='') as f:
        #     writer = csv.writer(f)

        # log
        self.gpt_log_path = 'C:/Users/gongj/Desktop/log/gpt_log/' + time.strftime('%Y-%m-%d_%H-%M-%S',
                                                                                  time.localtime()) + '.csv'
        self.canbus_log_path = 'C:/Users/gongj/Desktop/log/canbus_log/' + time.strftime('%Y-%m-%d_%H-%M-%S',
                                                                                        time.localtime()) + '.csv'
        # 初始化log
        self.gpt_log_header = ('time_s', 'time_r', 'message', 'response')
        with open(self.gpt_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.gpt_log_header)
        self.canbus_log_header = ('timestamp', 'speed', 'throttle', 'brake')
        with open(self.canbus_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.canbus_log_header)


        # self.guideline_prompt_path = r"C:\Users\gongj\Desktop\Guideline_prompt.txt"

        self.waypoint_number = 0
        self.last_left_wpt = None
        self.last_right_wpt = None
        self.now_waypoint = self._local_planner.target_waypoint
        self.old_waypoint = None
        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self.gpt4_agent_interface = GPT4AgentInterface()
        self._min_speed = 1
        self._max_speed = 2
        self._behavior = None
        self._sampling_resolution = 4.5
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        # self.suggestion_queue = queue.Queue()
        self.guideline = None  # 指导方针,持续迭代
        self.gpt3_thread = threading.Thread(target=self.get_command_from_gpt3)
        self.log_thread = threading.Thread(target=self.log_canbus)
        # self.gpt3_thread.start()
        self.command_dict = None
        self.last_speed = 20
        self.last_speed_buffer = 20
        self.last_action = "normal_behavior"
        self.last_action_buffer = "normal_behavior"

        self._vehicle = vehicle
        self.Behind_Vehicle_State = None
        self.Behind_Vehicle = None
        self.Behind_Vehicle_Speed = None
        self.Car_Lane_ID = None
        # self.Behind_Vehicle_Lane_Right_ID = None
        # self.Behind_Vehicle_Lane_Left_ID = None
        self.Car_Behind_Vehicle_Threshold = None
        self.Car_Obstacle_Detected = None

        self.reflection_counter = 0  # 用于跟踪已经完成了多少轮对话
        self.past_5_feedbacks = deque(maxlen=5)  # 用于存储最近5轮对话中GPT的反馈
        self.car_initial = True  # 用于判断车辆是否为初始状态

        self.driver_prompt = None
        # self.guideline_prompt = None
        with open(self.driver_prompt_path, 'r', encoding='utf-8') as f:
            self.driver_prompt = f.read()
        self.message_history = [
            {"role": "system",
             "content": self.driver_prompt
             }
        ]

        # # coach(非coach模式请注释下面的代码)
        # self.coach_prompt = None
        # with open( self.coach_prompt_path, 'r', encoding='utf-8') as f:
        #     self.coach_prompt = f.read()
        # self.system_prompt_for_behavior = [
        #     {"role": "system",
        #      # "content": "You are an assistant that assesses driving behaviors in the CARLA simulator and provides useful suggestions.\nYou need to assess whether my Driving Behaviors are good or bad.\nIf they are bad, please assist me in making improvements with your suggestions.\n\n[PART 1] INFORMATION\n\nPREVIOUS DRIVING BEHAVIORS: I will provide you with the following information: several turns of PREVIOUS DRIVING BEHAVIORS: Situation Analysis (Situation), Reasoning (Reasoning), and Action Instructions (Action) which are formatted as: {\"Situation\": \"...\", \"Reasoning\": \"...\", \"Action\": \"...\"}. \nYou should regard the latter ones' situations as the results of the former ones' Reasoning and Action Instructions.\n\n[PART 2] CRITERIA FOR EVALUATION AND SUGGESTION\nThere are good driving examples which are expert drivers' Situation Analysis, Reasoning, and Action Instructions, and bad driving examples which are novice drivers' Situation Analysis, Reasoning, and Action Instructions below. \nYou should evaluate the Driving Behaviors based on them as 'good' or 'bad'. \nThen, you should learn from the good driving examples when the evaluation is 'bad'.\nExample 1: Distance control\nGood:(A11)\nSituation: 'Driving on a road with a speed limit of 60km/h, with a vehicle ahead.'\nReasoning: 'Considering the speed limit is 40km/h, I will maintain a distance of about 10 meters from the vehicle in front. However, if the speed limit is 60km/h, I will increase this distance to around 20 meters. This extra space ensures I have enough time and distance for a safe reaction and braking in case of an emergency. I prioritize maintaining a safe distance and avoid tailgating the vehicle in front.'\nAction: 'Maintain a following distance of approximately 20 meters from the vehicle ahead and Adjust the distance according to the speed'\nBad:(A06)\nSituation: 'Driving on a road with a speed limit of 60km/h, with a vehicle ahead.'\nReasoning: 'Regardless of the speed limit, it's essential to maintain a proper following distance.'\nAction: 'Always maintain a car's length of distance from the vehicle in front'\nExample 2: Cross a congested road \nGood:(A14)\nSituation: 'On a road crowded with pedestrians and vehicles.'\nReasoning:  'In this congested situation, if you come to a complete stop, pedestrians and vehicles might make unpredictable moves. However, if your car proceeds at an intentionally slow pace, it signals to others that you are still moving forward. They may choose to pass you from behind if there's enough space, or if you're halfway through their path, they might let you continue. '\nAction: 'Proceed at a slow pace to facilitate safe passage.'\nBad:\nSituation: 'On a road crowded with pedestrians and vehicles.'\nReasoning: 'Considering the high density of obstacles and the increased risk of accidents, it's crucial to prioritize safety.'\nAction: 'Stop'\nExample 3: Go through the intersection with traffic lights\nGood:(P16)\nSituation: 'At an intersection with traffic lights and there are many vehicles and passengers'\nReasoning: 'When I am 20 meters away from the traffic light, I will release the accelerator foot and the brake at a crowded intersection'\nAction: 'Slow Down'\nBad:(p10)\nSituation: 'At an intersection with traffic lights and there are many vehicles and passengers'\nReasoning: 'Only look at heavy traffic point and fast cars'\nAction: 'Pass through'\n\n[PART 3] EVALUATION AND SUGGESTIONS\nYou must adhere to the specified rules:\n1. Evaluation\n1) Assess whether the provided driving behaviors align with the good and bad examples given in the CRITERIA as 'Good' or 'Bad'.\n2. Reason\n1) Give the reason of the evaluation above.\n2) Please describe using concise sentences.\n3. Suggesstion\n1) If the your evaluation is 'Bad', try to learn from the good examples and seek for a suggestion based on Good examples in CRITERIA above and the reason you evaluate.\n2) No suggestion is required if both Reasoning and Action evaluations are 'good'.\n3) Please describe using concise sentences.\n\n[PART 4] FORMAT\nYou should only respond for in the format described below:\nResponse Format: {\"Evaluation\": \"......\", \"Reason\": \"......\", \"Suggestion\": \"......\"}\n\n[PART 5] EXAMPLES\nHere's an example response:\nExample 1 \nINFORMATION: (p06)\n{\"Situation\": \"At an intersection without traffic lights and a car on the far side of the road\", \"Reasoning\": \"Slowing down helps me observe the road conditions when turning\", \"Action\": Speed_Down}\n{\"Situation\": \"At an intersection without traffic lights and the car is closing\", \"Reasoning\": \"The condition is safe, so I can turn left safely\", \"Action\": Normal_Behavior}\nEVALUATION AND SUGGESTION:\n{Evaluation: \"Good\", Reason: \"It is a responsible operation to slow down and observe the road conditions before turning left\", Suggestion: \"(None)\"}\nEXAMPLE 2\nINFORMATION: (p26)\n{\"Situation\": \"At an intersection without traffic lights and a car on the far side of the road\", \"Reasoning\": \"The vehicle is far away, so it does not slow down\", \"Action\":\"Normal_Behavior\"}\n{\"Situation\": \"At an intersection without traffic lights and the car is closing\", \"Reasoning\": \"The vehicle is far away, so it does not slow down\", \"Action\":\"Normal_Behavior\"}\n{\"Situation\": \"At an intersection without traffic lights and the car is closing\", \"Reasoning\": \"The vehicle is far away, so it does not slow down, \"Action\":\"Normal_Behavior\"}\nEVALUATION AND Suggestion:\n{\"Evaluation\": \"Bad\", \"Reason\": \"Not slowing down before a turn is not a responsible operation. It is advisable to learn from the good example (p06) to slow down when approaching a turn to ensure safety.\", \"Suggestion\": \"Learn from example (p06) to slow down when approaching a turn to ensure observation of road conditions and enhance safety.\"}\n\n"
        #      "content": self.coach_prompt
        #      }
        # ]

        # with open( self.guideline_prompt_path, 'r', encoding='utf-8') as f:
        #     self.guideline_prompt = f.read()


        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        self.now_waypoint = self._local_planner.target_waypoint
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = 2

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_waypoint != self.old_waypoint:
            self.old_waypoint = self._incoming_waypoint
            self.waypoint_number = self.waypoint_number + 1
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def get_waypoint_number(self):
        return self.waypoint_number

    def clear_waypoint_number(self):
        self.waypoint_number = 0

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, traffic_light = self._affected_by_traffic_light(lights_list, 10)
        # 缩短红灯时间
        if affected:
            if traffic_light.get_green_time() > 5.0:
                traffic_light.set_green_time(5.0)

        return affected

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")

        def dist(v):
            return v.get_location().distance(waypoint.transform.location)

        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max_distance=20, up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max_distance=20, up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max_distance=20, up_angle_th=30)

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")

        def dist(w):
            return w.get_location().distance(waypoint.transform.location)

        walker_list = [w for w in walker_list if dist(w) < 10]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(vehicle_list=walker_list, max_distance=20,
                                                                             up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(vehicle_list=walker_list, max_distance=20,
                                                                             up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(vehicle_list=walker_list, max_distance=20,
                                                                             up_angle_th=80)

        return walker_state, walker, distance

    # 判断能不能向右变道
    def is_right_lane_change_allowed(self):
        ego_vehicle_loc = self._vehicle.get_location()
        waypoint = self._map.get_waypoint(ego_vehicle_loc)
        right_turn = waypoint.right_lane_marking.lane_change
        right_wpt = waypoint.get_right_lane()
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
            vehicle_list, max_distance=10, up_angle_th=180, lane_offset=1)
        if vehicle_state:
            return False
        if right_wpt is None:
            return False

        return (
                (right_turn in [carla.LaneChange.Right, carla.LaneChange.Both])
                and (waypoint.lane_id * right_wpt.lane_id > 0)
                and (right_wpt.lane_type == carla.LaneType.Driving)
        )

    # 判断能不能向左变道
    def is_left_lane_change_allowed(self):

        ego_vehicle_loc = self._vehicle.get_location()
        waypoint = self._map.get_waypoint(ego_vehicle_loc)
        left_turn = waypoint.left_lane_marking.lane_change
        left_wpt = waypoint.get_left_lane()
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
            vehicle_list, max_distance=10, up_angle_th=180, lane_offset=-1)
        if vehicle_state:
            return False
        if left_wpt is None:
            return False

        return (
                (left_turn in [carla.LaneChange.Left, carla.LaneChange.Both])
                and (waypoint.lane_id * left_wpt.lane_id > 0)
                and (left_wpt.lane_type == carla.LaneType.Driving)
        )

    ######################################################
    ########################GPT##########################
    ######################################################
    def get_command_from_gpt3(self):
        """
        Generate a command from OpenAI GPT-4 model in a specific format.
        """

        # # coach 读guidelines
        # if self.read_guideline:
        #     self.read_guideline = False
        #     with open(self.guideline_path, 'r') as f:
        #         self.guideline = f.read()

        while True:
            lock = threading.Lock()

            with lock:
                if self._vehicle is None:
                    print("NONE")
                    break
                ego_vehicle_loc = self._vehicle.get_location()
                ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
                vehicle_list = self._world.get_actors().filter("*vehicle*")
                walker_list = self._world.get_actors().filter("*walker.pedestrian*")

                def dist(v):
                    return v.get_location().distance(ego_vehicle_wp.transform.location)

                vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]

                # walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)
                walker_ahead, walkers, walker_distances, walker_angles, walker_orientations = self._pedestrian_obstacle_detected(
                    walker_list)
                vehicle_ahead, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
                # if vehicle and vehicle in vehicle_list:
                #     vehicle_list = vehicle_list.remove(vehicle)
                is_surrounding_vehicles, vehicles, distances, lanes, angles = self._vehicle_obstacle_detected_v2(
                    vehicle_list, 15)

                # behind_vehicle_state, behind_vehicle, b_distance = self._vehicle_obstacle_detected(vehicle_list, max(
                #     self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=170)
                behind_vehicle_state, behind_vehicle, b_distance = False, None, None
                '''
                behind_vehicle_state = self.Behind_Vehicle_State
                behind_vehicle = self.Behind_Vehicle
                '''
                behind_vehicle_speed = self.Behind_Vehicle_Speed
                car_lane_id = self.Car_Lane_ID
                car_behind_vehicle_threshold = self.Car_Behind_Vehicle_Threshold
                car_obstacle_detected = self.Car_Obstacle_Detected
                # car_status = "Car_Initial: " + str(self.car_initial) + "\n"
                # lane_change = self.now_waypoint.lane_change
                # car_lane = ''
                # if lane_change == carla.LaneChange.NONE:
                #     car_lane = 'Single'
                # elif lane_change == carla.LaneChange.Left:
                #     car_lane = 'Right'
                # elif lane_change == carla.LaneChange.Right:
                #     car_lane = 'Left'
                # car_status = "Car_Lane: " + str(car_lane) + "\n"
                car_status = "Car_Speed: " + "{:3f}".format(
                    self._vehicle.get_velocity().length() * 3.6) + " km/h" + "\n"
                car_status = car_status + "Car_Speed_limit:" + str(self._speed_limit) + " km/h" + "\n"
                car_status = car_status + "Car_Direction:" + str(self._local_planner.target_road_option) + "\n"
                car_status = car_status + "Car_Incoming_direction:" + str(self._incoming_direction) + "\n"
                car_status = car_status + "Car_at_Traffic_light:" + str(self.traffic_light_manager()) + "\n"
                car_status = car_status + "is_right_lane_change_allowed:" + str(
                    self.is_right_lane_change_allowed()) + "\n"
                car_status = car_status + "is_left_lane_change_allowed:" + str(
                    self.is_left_lane_change_allowed()) + "\n"
                # car_status = car_status + "Walker_State:" + str(walker_state) + "\n"
                # car_status = car_status + "Walker:" + str(walker) + "\n"
                if walker_ahead:
                    walkers_ahead = []
                    for i in range(0, len(walkers)):
                        walker = {}
                        walker['No.'] = i + 1
                        walker['Distance'] = "{:3f}".format(walker_distances[i]) + " m"
                        walker['Relative Position'] = walker_angles[i]
                        walker['Orientation'] = walker_orientations[i]
                        walkers_ahead.append(walker)
                    car_status = car_status + "Walkers_ahead:" + str(walkers_ahead) + "\n"
                else:
                    car_status = car_status + "Walkers_ahead:" + str(walker_ahead) + "\n"

                if vehicle_ahead:
                    distance = distance - max(
                        vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                        self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
                    vehicle_json = {}
                    vehicle_json['No.'] = 1
                    vehicle_json['Distance'] = "{:3f}".format(distance) + " m"
                    vehicle_json['Speed'] = "{:3f}".format(get_speed(vehicle)) + ' km/h'
                    car_status = car_status + "Vehicle_ahead:" + str(vehicle_json) + "\n"
                    # car_status = car_status + "Vehicle:" + str(vehicle) + "\n"
                    # car_status = car_status + "Distance:" + str(vehicle) + "\n"
                    # car_status = car_status + "Vehicle_speed:" + str(get_speed(vehicle)) + "\n"
                else:
                    car_status = car_status + "Vehicle_ahead:" + str(vehicle_ahead) + "\n"

                # if is_surrounding_vehicles:
                #     vehicles_nearby = []
                #     for i in range(0, len(vehicles)):
                #         vehicle_nearby = {}
                #         vehicle_nearby['No.'] = i + 1
                #         if angles[i] == 'right' or angles[i] == 'left':
                #             dis = distances[i] - min(
                #                 vehicles[i].bounding_box.extent.y, vehicles[i].bounding_box.extent.x) - min(
                #                 self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
                #         else:
                #             dis = distances[i] - max(
                #                 vehicles[i].bounding_box.extent.y, vehicles[i].bounding_box.extent.x) - max(
                #                 self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
                #         vehicle_nearby['Distance'] = "{:3f}".format(dis) + " m"
                #         vehicle_nearby['Relative Position'] = angles[i]
                #         vehicle_nearby['Speed'] = "{:3f}".format(get_speed(vehicles[i])) + ' km/h'
                #         vehicles_nearby.append(vehicle_nearby)
                #     car_status = car_status + "Vehicles_Nearby:" + str(vehicles_nearby) + "\n"
                # else:
                #     car_status = car_status + "Vehicles_Nearby:" + str(is_surrounding_vehicles) + "\n"

                # self.car_initial = False

                # coach
                # guidelines = self.guideline

                print("\nCar Status:\n")
                print(car_status)
                command_dict = None
                message_sum = self.message_history.copy()

                if self.command_queue.empty():
                    command = None
                else:
                    command = list(self.command_queue.queue)

                message_sum.append(
                    {"role": "user",
                     "content": f"PREVIOUS DRIVING BEHAVIORS: {json.dumps(command)}, CURRENT STATE & ENVIRONMENT：{car_status}"}
                )

                # message_sum.append(
                #     {"role": "user",
                #      "content": f"PREVIOUS DRIVING BEHAVIORS: {json.dumps(command)}, CURRENT STATE & ENVIRONMENT：{car_status}, EXISTING GUIDELINES: {guidelines}"}
                # )

                print("---------------------------------------------\n")
                print(f"PREVIOUS DRIVING BEHAVIORS: {json.dumps(command)}")

                # print("---------------------------------------------\n")
                # print("GUIDELINES:" + guidelines)

                send_time = datetime.now()
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=message_sum  # Use message_history instead of the hardcoded messages
                )
                respond_time = datetime.now()
                with open(self.gpt_log_path, 'a', newline='', encoding='UTF-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([send_time, respond_time, message_sum[1]["content"],
                                     response["choices"][0]["message"]["content"]])

                if "choices" in response and len(response["choices"]) > 0:
                    choice = response["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        tem = choice["message"]["content"].strip()

                        if tem:
                            cleaned = '\n'.join([line for line in tem.splitlines() if line.strip()])
                            try:
                                self.command_dict = json.loads(cleaned)
                                # print("Cleaned JSON data:", cleaned)
                                # 记录上一条指令，用于maintain speed保持速度
                                if self.command_dict is not None:
                                    self.last_action_buffer = self.command_dict["Action"]
                                if self.command_dict[
                                    "Action"] == "maintain_speed" and self.last_action != "maintain_speed":
                                    print("we need to reset last_speed:" + str(self._speed))
                                    self.last_speed_buffer = self._speed
                                if self.command_dict[
                                    "Action"] != "lane_changing_left" and self.last_action == "lane_changing_left":
                                    self.last_left_wpt = None
                                if self.command_dict[
                                    "Action"] != "lane_changing_right" and self.last_action == "lane_changing_right":
                                    self.last_right_wpt = None
                                if self.command_queue.qsize() > 3:
                                    self.command_queue.get()

                                self.command_queue.put(self.command_dict)
                                print("\n------------------------------\n")
                                print("\nDriverAgent Output:\n")
                                print(cleaned)
                                print("\nThis is waypoint number:")
                                print(self.waypoint_number)

                                # # coach(非coach模式请注释以下行)
                                # reflection_prompt = {
                                #     "role": "user",
                                #     "content": f"PREVIOUS DRIVING BEHAVIORS: {list(self.command_queue.queue)}, EXISTING GUIDELINES: {self.guideline}."
                                # }
                                # # 使用新的反思prompt生成一个新的GPT输出
                                # send_time = datetime.now()
                                # reflection_response = openai.ChatCompletion.create(
                                #     model="gpt-4",
                                #     messages=self.system_prompt_for_behavior + [reflection_prompt]
                                # )
                                # respond_time = datetime.now()
                                # with open(self.gpt_log_path, 'a', newline='', encoding='UTF-8') as f:
                                #     writer = csv.writer(f)
                                #     writer.writerow([send_time, respond_time, reflection_prompt["content"],
                                #                      reflection_response["choices"][0]["message"]["content"]])
                                # if "choices" in reflection_response and len(reflection_response["choices"]) > 0:
                                #     choice = reflection_response["choices"][0]
                                #     if "message" in choice and "content" in choice["message"]:
                                #         tem = choice["message"]["content"].strip()
                                #         if tem:
                                #             cleaned = '\n'.join([line for line in tem.splitlines() if line.strip()])
                                #             try:
                                #                 print("\n------------------------------\n")
                                #                 print("\nCoach Agent Output:\n")
                                #                 print(cleaned)
                                #                 response_dict = json.loads(cleaned)
                                #                 self.guideline = response_dict["Integrated Guideline"]
                                #             except json.JSONDecodeError as e:
                                #                 print("Error decoding JSON Inside:", e)
                                #                 # Handle the JSON decode error appropriately

                            except json.JSONDecodeError as e:
                                print("Error decoding JSON Outside:", e)
                        else:
                            print("Received empty response content from GPT-3 API.")
                    else:
                        print("No valid 'message' or 'content' in API response.")
                else:
                    print("No valid 'choices' in API response.")

    def start_gpt3_thread(self):
        self.gpt3_thread = threading.Thread(target=self.get_command_from_gpt3)
        self.gpt3_thread.start()

    change_lane = True

    def log_canbus(self):
        with open(self.canbus_log_path, 'a', newline='', encoding='UTF-8') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), get_speed(self._vehicle), self._vehicle.get_control().throttle,
                             self._vehicle.get_control().brake])
        time.sleep(1)

    def start_log_thread(self):
        self.log_thread = threading.Thread(target=self.log_canbus)
        self.log_thread.start()

    def translate_gpt_command(self, response):

        if isinstance(response, dict) and 'Action' in response:
            function_name = response["Action"]
            if function_name == "stop":
                return self.stop()
            elif function_name == "speed_up":
                return self.speed_up()
            elif function_name == "speed_down":
                return self.speed_down()
            elif function_name == "maintain_speed":
                return self.maintain_speed()
            elif function_name == "lane_changing_left":
                return self.lane_changing_left()
            elif function_name == "lane_changing_right":
                return self.lane_changing_right()
            elif function_name == "normal_behavior":
                return self.normal_behavior()
        else:
            print("Response is not a valid dictionary or doesn't contain 'Action' key:", response)
            self.normal_behavior()

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        """
        self._update_information()
        control = None
        command_dict = None

        # Get command from GPT-4
        if not self.gpt3_thread.is_alive():
            # print("开启线程")
            self.start_gpt3_thread()

        # log
        if not self.log_thread.is_alive():
            self.start_log_thread()

        while self.command_dict is None:
            time.sleep(0.1)

        return self.translate_gpt_command(self.command_dict)

    def set_last_speed(self):
        self.last_speed = self.last_speed_buffer

    def get_last_action(self):
        return self.last_action_buffer

    def set_last_action(self):
        self.last_action = self.last_action_buffer

    ######################################################
    ####################ACTION LIBRARY####################
    ######################################################

    # 刹车
    def stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    # 正常跟着导航行驶
    def normal_behavior(self, debug=False):
        target_speed = min([
            self._behavior.max_speed,
            self._speed_limit - self._behavior.speed_lim_dist])
        self._local_planner.set_speed(target_speed)
        control = self._local_planner.run_step(debug=debug)
        return control

    # 加速
    def speed_up(self, increment=0.5, debug=False):
        current_speed = self._speed
        # target_speed = self._speed_limit * 1.1
        target_speed = 45
        if current_speed + increment >= target_speed:
            next_speed = target_speed
        else:
            next_speed = current_speed + increment

        self._local_planner.set_speed(next_speed)
        control = self._local_planner.run_step(debug=debug)
        return control

    # 减速
    def speed_down(self, decrement=1.5, debug=False):
        current_speed = self._speed

        next_speed = max(0, current_speed - decrement)
        self._local_planner.set_speed(next_speed)
        control = self._local_planner.run_step(debug=debug)
        return control

    # 保持现在的速度
    def maintain_speed(self, debug=False):
        # self._local_planner.set_speed(current_speed)
        current_speed = self.last_speed
        self._local_planner.set_speed(current_speed)
        control = self._local_planner.run_step(debug=debug)
        return control

    # 换到左道
    def lane_changing_left(self, debug=False):
        # ego_vehicle_loc = self._vehicle.get_location()
        waypoint = self._incoming_waypoint
        left_wpt = waypoint.get_left_lane()
        end_waypoint = self._local_planner.target_waypoint
        if self.last_left_wpt is None:
            print("This is left change")
            if left_wpt is None:
                left_wpt = self._incoming_waypoint
            print(left_wpt)
            print(waypoint)
            self.last_left_wpt = left_wpt
            road_end_wpt = left_wpt.next_until_lane_end(0.25)[-1]
            self.set_destination(road_end_wpt.transform.location, left_wpt.transform.location, True)
            self.set_destination(end_waypoint.transform.location, road_end_wpt.transform.location, False)
        control = self._local_planner.run_step(debug=debug)
        return control

    # 换到右道
    def lane_changing_right(self, debug=False):
        # ego_vehicle_loc = self._vehicle.get_location()
        waypoint = self._incoming_waypoint
        right_wpt = waypoint.get_right_lane()
        end_waypoint = self._local_planner.target_waypoint
        if self.last_right_wpt is None:
            print("This is right change")
            if right_wpt is None:
                right_wpt = self._incoming_waypoint
            print(right_wpt)
            print(waypoint)
            self.last_right_wpt = right_wpt
            road_end_wpt = right_wpt.next_until_lane_end(0.25)[-1]
            self.set_destination(road_end_wpt.transform.location, right_wpt.transform.location, True)
            self.set_destination(end_waypoint.transform.location, road_end_wpt.transform.location, False)
        control = self._local_planner.run_step(debug=debug)
        return control
