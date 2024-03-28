# Copyright (c) # Copyright (c) 2018-2020 CVC.
#memory no coach
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
        # 对齐过的prompt以aligned结尾，没对齐过的以without_alignment结尾
        self.driver_prompt_path = r""
        self.coach_prompt_path = r""
        # log
        self.gpt_log_path = 'C:/Users/gongj/Desktop/log/' + time.strftime('%Y-%m-%d_%H-%M-%S',
                                                                          time.localtime()) + '.csv'
        # 初始化log
        self.gpt_log_header = ('time_s', 'time_r', 'message', 'response')
        with open(self.gpt_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.gpt_log_header)

        self.guideline_path = r""
        # self.guideline_prompt_path = r"C:\Users\gongj\Desktop\Guideline_prompt.txt"

        self.waypoint_number = 0
        self.last_left_wpt = None
        self.last_right_wpt = None
        self.now_waypoint = None
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
        self.guideline = None #指导方针,持续迭代
        self.gpt3_thread = threading.Thread(target=self.get_command_from_gpt3)
        self.gpt3_thread.start()
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
        self.car_initial = True #用于判断车辆是否为初始状态

        self.driver_prompt = None
        # self.guideline_prompt = None
        self.coach_prompt = None
        with open( self.driver_prompt_path, 'r', encoding='utf-8') as f:
            self.driver_prompt = f.read()
        with open( self.coach_prompt_path, 'r', encoding='utf-8') as f:
            self.coach_prompt = f.read()
        # with open( self.guideline_prompt_path, 'r', encoding='utf-8') as f:
        #     self.guideline_prompt = f.read()
        self.message_history = [
            {"role": "system",
             # "content": "You are a driver in the CARLA simulator. Drive the agent car safely and quickly to its destination on a city road. Base your actions only on the information I provide.\n\n[PART 1] INFORMATION:\nI'll give you CARLA parameters for the agent car's current state and environment (e.g. traffic lights, vehicles, walkers). Use the provided navigation system to drive from start to destination, following lanes and waypoints.\nCURRENT STATE & ENVIRONMENT\nCar_Initial:Whether the agent car is newly generated or not\nCar_Position: Current location coordinates (x, y, z) of the agent car.\nCar_Speed: Current speed of the agent car in km/h.\nCar_Speed_limit: Maximum speed allowed for the agent car in the current area.\nCar_Lane_ID: Unique identifier for the lane the car is currently on.\nCar_Waypoint: The current waypoint of the agent car. A waypoint in CARLA is a designated position point with location data (3D coordinates: x, y, z) and rotation data (commonly represented as a quaternion or Euler angles: roll, pitch, yaw) used for navigation and simulation.\nCar_Incoming_waypoint: The upcoming waypoint a distance ahead (normally 3 waypoints ahead) the agent car is approaching. A waypoint in CARLA is a designated position point with location data (3D coordinates: x, y, z) and rotation data (commonly represented as a quaternion or Euler angles: roll, pitch, yaw) used for navigation and simulation.\nCar_Direction: The current road option that the local planner is guiding the agent car towards. You can think of it as the current action. \nCar_Incoming_direction: The road option at a distance ahead (normally 3 waypoints ahead) that the local planner is guiding the agent car towards.\nThe value of Car_Direction and Car_Incoming_direction is an integer from -1 to 6, representing:\n-1 (VOID): Represents that there is no valid movement option currently being considered for the current lane segment by the agent car.\n1 (LEFT): Represents that the agent car is currently turning left.\n2 (RIGHT): Represents that the agent car is currently turning right\n3 (STRAIGHT): Represents that the vehicle is currently moving straight along the current lane segment without making an explicit turn or road change.\n4 (LANEFOLLOW): Represents that the vehicle is currently following the current lane, emphasizing the ongoing process of continuous driving on the current lane, which may include some slight road curve following, without requiring explicit road changes or turns.\n5 (CHANGELANELEFT): Represents that the agent car is changing to the left lane.\n6 (CHANGELANERIGHT): Represents that the agent car is changing to the right lane.\nCar_Incoming_direction: The road option at a distance ahead (normally 3 waypoints ahead) that the local planner is guiding the agent car towards.\nCar_at_Traffic_light: This indicates the current status of the traffic lights in front. If this is True, it signifies that you are approaching a red light. Conversely, if this is False, it implies that you are approaching a green light.\nWalker: the walker in front of the agent car.\nWalker_State: True if there is a walker in front, False if not.\nWalker_Distance: Distance to the walker in front in meters. If the distance is equal to or smaller than 0, it indicates a collision has occurred.\nWalker_Speed: Current speed of the walker in front of the agent car in km/h.\nVehicle: the vehicle in front of the agent car.\nVehicle_Speed: Current speed of the vehicle in front in km/h.\nVehicle_State: True if there is a vehicle in front of the agent car, False if not.\nVehicle_Distance: Distance to the vehicle in front of the car in meters. If the distance is equal to or smaller than 0, it indicates a collision has occurred.\nBehind_Vehicle_State: Indicates if there's a vehicle behind the agent car. True means there's a vehicle, False means there isn't.\nBehind_Vehicle: Describes the details of the vehicle behind, like type, color, identification.\nBehind_Vehicle_Speed: Speed of the vehicle behind in km/h.\nBehind_Vehicle_Distance: Distance to the vehicle behind in meters.\nis_right_lane_change_allowed: This function is used to determine whether a right lane change is permissible in the current situation. When the function returns True, it indicates that a right lane change is allowed, and when it returns False, it signifies that a right lane change is not allowed.\nis_left_lane_change_allowed:This function is used to determine whether a left lane change is permissible in the current situation. When the function returns True, it indicates that a left lane change is allowed, and when it returns False, it signifies that a left lane change is not allowed.\nPREVIOUS DRIVING BEHAVIORS: \nYou'll get up to four past DRIVING BEHAVIORS of the agent car which are formatted as:  {\"Situation\": \"...\", \"Reasoning\": \"...\", \"Action\": \"...\"}\nGUIDELINES: \nConsider guidelines from an evaluator for professional driving insights. Integrate these when deciding driving behavior. The format is: {\"Guideline\":\"...\"}\n\n[PART 2] ACTION LIBRARY: These are pre-coded actions that the agent car can directly implement.\n1. ACTION_NAME: stop\nDescription:This function performs an emergency stop. It sets the throttle to zero, applies maximum brake, and releases the handbrake. Steering remains unchanged to keep the car in its lane.\n2. ACTION_NAME: normal_behavior\nDescription: This function makes the agent car follow a route set by global navigation, keeping its speed close to the road's limit or the car's maximum. NEVER use this function for stopping or when stopped.\n3. ACTION_NAME: maintain_speed\nDescription: This function maintains the agent car's current speed on its navigation route. NEVER use it when the car needs to stay stopped.\n4. ACTION_NAME: lane_changing_left\nDescription: This function makes the agent car change to the left lane.\n5. ACTION_NAME: lane_changing_right\nDescription: This function makes the agent car change to the right lane.\n6. ACTION_NAME: speed_up\nDescription: This function accelerates the agent car by 5 km/h, up to its maximum allowed speed, and continues on its navigation route.\n7. ACTION_NAME: speed_down\nDescription: This function decelerates the agent car by 5 km/h, down to a minimum of zero, and then proceeds on its navigation route. Use it for gradual stops.\n\n[PART 3] DRIVING BEHAVIOR\nPerform three tasks: Situation Analysis(Situation), Reasoning（Reason）, and Action Instructions(Action). Use the provided INFORMATION to guide the agent car's next action.\n\nExecution priority：\nPriority Level 1 (Mandatory):\n1.Stop no matter how much the Car_speed is if the Vehicle_Distance or the Walker_Distance is less than 1.5m.\n2.Stop no matter how much the Car_speed is if the Car_at_Traffic_light is less than true.\n3.Do not stop if the road ahead has no vehicle, pedestrian, or intersection.\nPriority Level 2:\n1.Decelerate if agent car is within 5m of a vehicle/pedestrian ahead.\n2.Slow down when a red road is detected ahead.\n3.Slow down when an intersection is detected ahead.\n4.Ensure a minimum distance of 1m from moving cars ahead.\n\n1. Situation Analysis:\n1)If Car_Initial is False, that means the agent car is not newly generated and has been generated for a period of time. Only if Car_Initial is True, that the agent car is stationary in the middle of the road because it has just been generated.\n2) Analyze and describe the agent car's situation using the given INFORMATION.\n3) Base your decision on all provided INFORMATION.\n4) For the next DRIVING BEHAVIOR, consider up to 4 previous instances and SUGGESTIONS for both \"situation\" and \"reasoning\".\n\n2. Reasoning:\n1) Decide the agent car's next action based on Situation Analysis.\n2) Prioritize safety for both driver and pedestrians.\n3) Reach the destination quickly, ensuring safety and comfort.\n4) Give clear, detailed instructions for the action.\n5) Think step by step.\n\n3. Action Instructions:\n1) Specify the action using the ACTION LIBRARY format.\n2) Follow this structure: \"Action\":\"ACTION_NAME\".\n3) Use a single phrase for each action. Avoid multiple suggestions or extra content.\n\n[PART 4] EXAMPLE\n\n[ {'Situation': 'The agent car is currently driving on the right lane and approaching a vehicle that is 1.50 meters ahead in the same lane.', 'Reasoning': 'Since Car_Initial is False, the agent car is not newly generated and has been generated for a period of time.Based on the previous driving behaviors and the current state, the agent car is too close to approaching a vehicle ahead. Given the relatively short distance and the difference in speed between the two vehicles, the agent car should stop right now.', 'Action': 'stop'},\n{\"Situation\": \"The last action of the agent car is \"speed down\". Now the agent car is currently driving along a lane and approaching a front vehicle at a distance of 10 meters.\",\"Reasoning\": \"Since Car_Initial is False, the agent car is not newly generated and has been generated for a period of time.The agent car previously slowed down and is now approaching a nearby vehicle about 10 meters ahead in the same lane. Both the agent car and the preceding vehicle are moving at slower speeds. Since the distance is not critically close and no sudden stop is required, the agent car should continue to decelerate, ensuring a safe following distance.\", \"Action\": \"speed_down\"}\n{'Situation': 'The agent car is currently driving on the right lane and approaching a vehicle that is 2.50 meters ahead in the same lane.', 'Reasoning': 'Since Car_Initial is False, the agent car is not newly generated and has been generated for a period of time.Based on the previous driving behaviors and the current state, the agent car has been driving smoothly and is now approaching a vehicle ahead. Given therelatively short distance and the difference in speed between the two vehicles, the agent car should continue to decelerate cautiously to maintain a safe following distance.', 'Action': 'speed_down'},\n{'Situation': 'The agent car is currently driving on the right lane and approaching a walker that is 1.50 meters ahead in the same lane.', 'Reasoning': 'Since Car_Initial is False, the agent car is not newly generated and has been generated for a period of time.Based on the previous driving behaviors and the current state, the agent car is too close to approaching a walke ahead. Given the relatively short distance and the difference in speed between the agent car and the walker, the agent car should stop right now.', 'Action': 'stop'}\n{'Situation': 'the agent car is driving on the road and at a red light ahead', 'Reasoning': 'Since Car_Initial is False, the agent car is not newly generated and has been generated for a period of time. And since Car_at_Traffic_light is true, the agent car is approaching a red light , 'Action': 'stop'}\n\n\n[PART 5]OUT FORMAT\nYou should only respond in the format as described below (Absolutely do not, never, ever change double quotes (\") to single quotes (').):\nRESPONSE FORMAT: \n{\"Situation\": \"......\",\"Reasoning\": \"......\",\"Action\": \"......\"}"
             "content": self.driver_prompt
             }
        ]

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
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = 2

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_waypoint != self.now_waypoint:
            self.now_waypoint = self._incoming_waypoint
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
        affected, _ = self._affected_by_traffic_light(lights_list, 10)

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
            walker_state, walker, distance = self._vehicle_obstacle_detected(vehicle_list=walker_list, max_distance=20, up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(vehicle_list=walker_list, max_distance=20, up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(vehicle_list=walker_list, max_distance=20, up_angle_th=80)

        return walker_state, walker, distance

    # 判断能不能向右变道
    def is_right_lane_change_allowed(self):
        ego_vehicle_loc = self._vehicle.get_location()
        waypoint = self._map.get_waypoint(ego_vehicle_loc)
        right_turn = waypoint.right_lane_marking.lane_change
        right_wpt = waypoint.get_right_lane()
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
            vehicle_list, max_distance=8, up_angle_th=180, lane_offset=1)
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
            vehicle_list, max_distance=8, up_angle_th=180, lane_offset=-1)
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


        while True:
            lock = threading.Lock()

            with lock:
                if self._vehicle is None:
                    print("NONE")
                    break
                ego_vehicle_loc = self._vehicle.get_location()
                ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)
                vehicle_list = self._world.get_actors().filter("*vehicle*")

                def dist(v):
                    return v.get_location().distance(ego_vehicle_wp.transform.location)

                vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]

                walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)
                vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

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
                car_status = "Car_Initial: " + str(self.car_initial) + "\n"
                car_status = car_status + "Car_Speed: " + str(self._vehicle.get_velocity().length() * 3.6) + "\n"
                car_status = car_status + "Car_Speed_limit:" + str(self._speed_limit) + "\n"
                car_status = car_status + "Car_Direction:" + str(self._local_planner.target_road_option) + "\n"
                car_status = car_status + "Car_Incoming_direction:" + str(self._incoming_direction) + "\n"
                car_status = car_status + "Car_at_Traffic_light:" + str(self.traffic_light_manager()) + "\n"
                car_status = car_status + "Walker_State:" + str(walker_state) + "\n"
                car_status = car_status + "Walker:" + str(walker) + "\n"
                if (walker is not None):
                    t_distance = w_distance - max(
                        walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                        walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                        self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
                    car_status = car_status + "Walker_speed:" + str(get_speed(walker)) + "\n"
                    car_status = car_status + "Walker_Distance:" + str(t_distance) + "\n"

                car_status = car_status + "Vehicle_State:" + str(vehicle_state) + "\n"
                car_status = car_status + "Vehicle:" + str(vehicle) + "\n"
                if vehicle is not None:
                    distance = distance - max(
                        vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                        self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
                    car_status = car_status + "Vehicle_speed:" + str(get_speed(vehicle)) + "\n"
                    car_status = car_status + "Vehicle_Distance:" + str(distance) + "\n"
                car_status = car_status + "is_right_lane_change_allowed:" + str(
                    self.is_right_lane_change_allowed()) + "\n"
                car_status = car_status + "is_left_lane_change_allowed:" + str(self.is_left_lane_change_allowed()) + "\n"


                self.car_initial = False

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

                print("---------------------------------------------\n")
                print(f"PREVIOUS DRIVING BEHAVIORS: {json.dumps(command)}")

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
                                if self.command_dict["Action"] == "maintain_speed" and self.last_action != "maintain_speed":
                                    print("we need to reset last_speed:" + str(self._speed))
                                    self.last_speed_buffer = self._speed
                                if self.command_dict["Action"] != "lane_changing_left" and self.last_action == "lane_changing_left":
                                    self.last_left_wpt = None
                                if self.command_dict["Action"] != "lane_changing_right" and self.last_action == "lane_changing_right":
                                    self.last_right_wpt = None
                                if self.command_queue.qsize() > 3:
                                    self.command_queue.get()

                                self.command_queue.put(self.command_dict)
                                print("\n------------------------------\n")
                                print("\nDriverAgent Output:\n")
                                print(cleaned)
                                print("\nThis is waypoint number:")
                                print(self.waypoint_number)
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
        target_speed = self._speed_limit * 1.1
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
        road_end_wpt = left_wpt.next_until_lane_end(1)[-1]
        if self.last_left_wpt is None:
            print("This is left change")
            if left_wpt is None:
                left_wpt = self._incoming_waypoint
            print(left_wpt)
            print(waypoint)
            self.last_left_wpt = left_wpt
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
        road_end_wpt = right_wpt.next_until_lane_end(1)[-1]
        if self.last_right_wpt is None:
            print("This is right change")
            if right_wpt is None:
                right_wpt = self._incoming_waypoint
            print(right_wpt)
            print(waypoint)
            self.last_right_wpt = right_wpt
            self.set_destination(road_end_wpt.transform.location, right_wpt.transform.location, True)
            self.set_destination(end_waypoint.transform.location, road_end_wpt.transform.location, False)
        control = self._local_planner.run_step(debug=debug)
        return control
