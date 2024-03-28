#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control with steering wheel Logitech G29.

To drive start by preshing the brake pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.

"""

from __future__ import print_function

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS

    from pygame.locals import K_COMMA
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

#import cv2

# =============================================================================
# --Recording Setting----------------------------------------------------------
# =============================================================================
is_reverse = True
is_vehicle = True
frame_rate = 30
record_image = []
image_set = []
collision_happen = False
# bd_id = 0  # 33+ for no collision  # 99 for no building
is_special = True
no_vehicle = True


# 生成行人或车辆（包括自行车）
# ；人10；自行车/摩托车20；车/卡车20；
# 自行车/摩托车: bh.crossbike
# vehicle.harley-davidson.low_rider
# kawasaki.ninja
# yamaha.yzf
# micro.microlino 老头乐
# vespa.zx125 电动车


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, actor_filter):
        self.world = carla_world
        self.hud = hud
        self.player = None

        self.walker = None  # 生成的被碰撞walker

        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint_library = self.world.get_blueprint_library()
        # blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        if self.player is not None:
            spawn_points = self.world.get_map().get_spawn_points()
            # spawn_point = spawn_points[17] if spawn_points else carla.Transform() #x=19378.0,y=23743.0,z=50.0
            # spawn_point = spawn_points[19] if spawn_points else carla.Transform() # 19 for reverse #x=19378.0,y=19177.003906,z=50.0

            if is_reverse:
                spawn_point = spawn_points[22] if spawn_points else carla.Transform()  # 22 为单线反向
                spawn_point.rotation.yaw = 90
            else:
                spawn_point = spawn_points[14] if spawn_points else carla.Transform()  # 14 为单线位置
                spawn_point.rotation.yaw = -90

            # random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

            spawn_point.location.z += 0.0  # 起始高度防止碰撞
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            # spawn_point = spawn_points[17] if spawn_points else carla.Transform() # 3 way
            # spawn_point = spawn_points[19] if spawn_points else carla.Transform() # 19 FOR REVERSE

            if is_reverse:
                spawn_point = spawn_points[22] if spawn_points else carla.Transform()  # 22 为单线反向
                spawn_point.rotation.yaw = 90
            else:
                spawn_point = spawn_points[14] if spawn_points else carla.Transform()  # 14 为单线位置
                spawn_point.rotation.yaw = -90

            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)  # 控制hud显示/是否渲染hud

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._ackermann_enabled = False
        self._ackermann_reverse = 1
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._ackermann_control = carla.VehicleAckermannControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock, sync_mode):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == K_o:
                    try:
                        if world.doors_are_open:
                            world.hud.notification("Closing Doors")
                            world.doors_are_open = False
                            world.player.close_door(carla.VehicleDoor.All)
                        else:
                            world.hud.notification("Opening doors")
                            world.doors_are_open = True
                            world.player.open_door(carla.VehicleDoor.All)
                    except Exception:
                        pass
                elif event.key == K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception:
                            pass
                elif event.key > K_0 and event.key <= K_9:
                    index_ctrl = 0
                    if pygame.key.get_mods() & KMOD_CTRL:
                        index_ctrl = 9
                    world.camera_manager.set_sensor(event.key - 1 - K_0 + index_ctrl)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))

                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_f:
                        # Toggle ackermann controller
                        self._ackermann_enabled = not self._ackermann_enabled
                        world.hud.show_ackermann_info(self._ackermann_enabled)
                        world.hud.notification("Ackermann Controller %s" %
                                               ("Enabled" if self._ackermann_enabled else "Disabled"))
                    if event.key == K_q:
                        if not self._ackermann_enabled:
                            self._control.gear = 1 if self._control.reverse else -1
                        else:
                            self._ackermann_reverse *= -1
                            # Reset ackermann control
                            self._ackermann_control = carla.VehicleAckermannControl()
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        if not self._autopilot_enabled and not sync_mode:
                            print("WARNING: You are currently in asynchronous mode and could "
                                  "experience some issues with the traffic simulation")
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else:  # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else:  # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights:  # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
                # Apply control
                if not self._ackermann_enabled:
                    world.player.apply_control(self._control)
                else:
                    world.player.apply_ackermann_control(self._ackermann_control)
                    # Update control to the last one applied by the ackermann controller.
                    self._control = world.player.get_control()
                    # Update hud with the newest ackermann control
                    world.hud.update_ackermann_control(self._ackermann_control)

            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
                world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        # if keys[K_UP] or keys[K_w]:
        #     if not self._ackermann_enabled:
        #         self._control.throttle = min(self._control.throttle + 0.01, 1.00)
        #     else:
        #         self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        # else:
        #     if not self._ackermann_enabled:
        #         self._control.throttle = 0.0
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.02, 0.70)
        else:
            self._control.throttle = max(self._control.throttle - 0.03, 0.0)
        if keys[K_DOWN] or keys[K_s]:
            self._control.throttle = self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = max(self._control.brake - 0.3, 0.0)

        # if keys[K_DOWN] or keys[K_s]:
        #     if not self._ackermann_enabled:
        #         self._control.brake = min(self._control.brake + 0.2, 1)
        #     else:
        #         self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
        #         self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        # else:
        #     if not self._ackermann_enabled:
        #         self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================


class DualControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        self._steer_idx = int(
            self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('G29 Racing Wheel', 'handbrake'))

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._parse_vehicle_wheel()
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        # toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.world.get_map().name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt(
                (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        global collision_happen
        collision_happen = True

        print("Collision happening!")

        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraRecorder -------------------------------------------------------------
# ==============================================================================
import pickle
import threading

buf = []
fid = 0


def fast_save_to_disk(path, image):
    global buf
    global fid
    print("saving recording image with size")
    data = (path, image)
    buf.append(data)
    if len(buf) >= 180:
        print("image info: ", image)
        print("buf info: ", buf)
        threading.Thread(target=save_data(), args=(buf.copy(), str(fid))).start()
        buf = []
        fid += 1


def save_data(raw_data, path):
    # raw_data is a list of tuples where each tuple is:
    # 0: path
    # 1: frame data
    #  removed     2: size of the frame (L x W)
    #  removed 3: speed of the car at the frame

    data = []
    for datum in raw_data:
        data.append((datum[0], datum[1].raw_data.tobytes()))
    with open(path, "wb") as out:
        pickle.dump(data, out, -1)


def process_and_save(rawData, path=None):
    # path is passed in so that this function can be simply switched with save_data
    # also, this does not perform well, I would recommend not using this.
    for data in rawData:
        path, raw_bytes, speed = data
        raw_bytes.save_to_disk(path + '-' + str(speed) + '.png')


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        #  carla.Rotation(pitch=-15)),
        #             carla.Transform(carla.Location(x=1.6, z=1.7))

        self._camera_transforms = [
            # carla.Transform(carla.Location(x=0.25, y=-0.2, z=1.2))]  # first person angle for tesla model3
            # carla.Transform(carla.Location(x=0, y=-0.3, z=1.2), carla.Rotation(pitch=5))]  # FOV90下的第一人称视角
            # carla.Transform(carla.Location(x=0, y=-0.3, z=1.2))]
            carla.Transform(carla.Location(x=0, y=-0.3, z=1.2))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB']]  # ,
        # ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
        # ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
        # ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
        # ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
        # ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
        #     'Camera Semantic Segmentation (CityScapes Palette)'],
        # ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                bp.set_attribute('fov', str(90))  # 设置FOV为90_
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        # global video_Writer

        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            # print("raw_sensers data: ",self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            # print(array.shape)
            # record_image.append(array)

            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            if self.recording:
                # image.save_to_disk('record_out/'+folder_name+'/%08d' % image.frame)
                # fast_save_to_disk('record_out/'+folder_name+'/%08d' % image.frame, array)
                global record_image
                global image_set
                image_set.append(image)
                record_image.append(array)
                # video_Writer.write(array)

            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


# ==============================================================================
# -- spawn_walker() ---------------------------------------------------------------
# ==============================================================================
def spawn_walker(client, loop):
    special_list = ["omafiets", "crossbike", "low_rider", "ninja", "yzf", "microlino", "zx125", "century"]

    sim_world = client.get_world()
    bp = sim_world.get_blueprint_library()

    map = sim_world.get_map()
    playerStart = map.get_spawn_points()
    # 地图中显示生成点
    # for index, spawn_point in enumerate(playerStart):
    #     location = spawn_point.location
    #     map.debug.draw_string(location, f"Spawn Point {index}", draw_shadow=False,
    #                             color=carla.Color(255, 0, 0), life_time=10.0,
    #                             persistent_lines=True)

    # Change weather for playback
    weather = sim_world.get_weather()
    weather.sun_altitude_angle = 90
    weather.fog_density = 3
    weather.fog_distance = 0
    sim_world.set_weather(weather)
    # Get the light manager and lights
    lmanager = sim_world.get_lightmanager()
    mylights = lmanager.get_all_lights()

    # Custom a specific light
    light01 = mylights[0]
    light01.turn_on()
    light01.set_intensity(50.0)

    vehicle_list = bp.filter('vehicle.*.*')

    global veh_list
    global is_reverse
    global used_len
    global bd_id
    global is_vehicle
    global lopp

    is_special = False

    if bd_id == 'FixedNull' or bd_id == 'PracNull':
        print("loop is ", loop)
        print("list is ", veh_list)
        print(" tag is ", veh_list[loop])
        if is_vehicle:
            vehicle_bp = random.choice(bp.filter('vehicle.*.' + veh_list[loop]))  # [14 - loop]
            if (vehicle_bp.tags[2] in special_list) or (
                    vehicle_bp.tags[1] in special_list) or (
                    vehicle_bp.tags[0] in special_list):
                is_special = True
                print("Special type car")
            else:
                is_special = False
            if is_reverse:
                veh_spawn_point = playerStart[52]  # 52 for reverse vehicle spawn point
                print("reverse vehicle spawn point", veh_spawn_point)
            else:
                veh_spawn_point = playerStart[61]  # 61 for vehicle spawn_point
                print("vehicle spawn point", veh_spawn_point)
            vehicle = sim_world.spawn_actor(vehicle_bp, veh_spawn_point)
        else:
            vehicle_bp = random.choice(bp.filter('walker.pedestrian.' + veh_list[loop]))
            if not is_reverse:
                spawn_point = playerStart[18]  # 18号为一号行人起始点
                print("walker spawn point", spawn_point)
            else:
                spawn_point = playerStart[20]  # 20 for reverse
                print("reverse walker spawn point", spawn_point)

            if vehicle_bp.has_attribute('is_invincible'):
                vehicle_bp.set_attribute('is_invincible', 'false')
                print("Setting not invincible Done")
            vehicle = sim_world.spawn_actor(vehicle_bp, spawn_point)
        return vehicle, is_special

    if is_vehicle:
        print("loop is ", loop)
        print("list is ", veh_list)
        print("veh trials")
        if not is_reverse:
            vehicle_bp = random.choice(vehicle_list)

            while (vehicle_bp.tags[2] in special_list) or (
                    vehicle_bp.tags[1] in special_list) or (
                    vehicle_bp.tags[0] in special_list):
                vehicle_bp = random.choice(vehicle_list)

            if (vehicle_bp.tags[2] in special_list) or (
                    vehicle_bp.tags[1] in special_list) or (
                    vehicle_bp.tags[0] in special_list):
                is_special = True
                print("Special type car")
            else:
                is_special = False

            veh_list.append(vehicle_bp)

            veh_spawn_point = playerStart[61]  # 61 for vehicle spawn_point

        else:

            print("used len ", used_len, " Loop ", loop)
            print("Lopp ", lopp)
            print(" rank ", 0 - loop + used_len)
            vehicle_bp = veh_list[lopp - loop + used_len]  # 1 # 16]  #    # 2
            print("the list ", veh_list)
            print("VVVEHICLE BP ", vehicle_bp)
            if (vehicle_bp.tags[2] in special_list) or (
                    vehicle_bp.tags[1] in special_list) or (
                    vehicle_bp.tags[0] in special_list):
                is_special = True
                print("Special type car")
            else:
                is_special = False
            veh_spawn_point = playerStart[51]  # 51 for reverse vehicle spawn point
            vehicle_bp.yaw = 90  # reverse
            # veh_list.append(vehicle_bp)
        print("Select vehicle bp:", vehicle_bp)
        vehicle = sim_world.spawn_actor(vehicle_bp, veh_spawn_point)

        return vehicle, is_special
    else:
        print("loop is ", loop)
        # print("bp is ", veh_list[loop-1])
        print("walker trials")
        if not is_reverse:
            vehicle_bp = random.choice(bp.filter('walker.pedestrian.*'))
            while (vehicle_bp in veh_list) or (vehicle_bp.tags[2] in special_list) or (
                    vehicle_bp.tags[1] in special_list) or (
                    vehicle_bp.tags[0] in special_list):
                vehicle_bp = random.choice(bp.filter('walker.pedestrian.*'))
            veh_list.append(vehicle_bp)
            print("BP is ", vehicle_bp)
            spawn_point = playerStart[18]  # 18号为一号行人起始点
            print("walker spawn point", spawn_point)
        else:
            vehicle_bp = veh_list[lopp - loop + used_len]  # 1 #] 16  #   # 2
            spawn_point = playerStart[20]  # 20 for reverse
            print("reverse walker spawn point", spawn_point)

        if vehicle_bp.has_attribute('is_invincible'):
            vehicle_bp.set_attribute('is_invincible', 'false')
            print("Setting not invincible Done")

        vehicle = sim_world.spawn_actor(vehicle_bp, spawn_point)
        return vehicle, is_special


# ==============================================================================
# -- AOI_Cal() ---------------------------------------------------------------
# ==============================================================================
def AOI_Cal(x1, x2, y1, y2, z1, z2):
    # focal_len = 960  # 1920 / 2* tan(pi/2/2)  # 90 degree = pi/2
    # tan45 = 1  # tan(45degree) = 1

    if is_reverse:
        # xx = 960 * (x2 - x1) / (y2 - y1)  # + 960
        # yy = 960 * (z1 - z2) / (y1 - y2)  # + 540
        xx = 960 + 960 * (x1 - x2) / (y2 - y1)
        yy = 540 - 960 * (z2 - z1) / (y2 - y1)

    else:
        xx = 960 - 960 * (x1 - x2) / (y1 - y2)  # + 960
        yy = 540 - 960 * (z2 - z1) / (y1 - y2)  # + 540

    return xx, yy


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate
    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)
    # New we must change from UE4's coordinate system to an "standard"
    # (x,y,z) -> (y,-z,x)
    # and we remove the fourth component also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================
def game_loop(args, loop, ran_speed):
    pygame.init()
    pygame.font.init()
    world = None
    # 更改building,中间可替换：building07
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        sim_world = client.get_world()
        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager = client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        controller = KeyboardControl(world, args.autopilot)

        tick = 0
        start = 999

        if args.sync:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()

        # Global Statement
        global is_vehicle
        global is_special
        global no_vehicle
        global bd_id
        global reaction_time
        global lopp

        no_collision = False

        if ran_speed <= 3.6:
            is_vehicle = False
        else:
            is_vehicle = True

        if not no_vehicle:
            vehicle, is_special = spawn_walker(client, loop)

        global start_tick

        if frame_rate == 60:
            wait_tick = 50  # 60帧下为50tick
        else:
            wait_tick = 25  # 30帧下为25tick

        if no_vehicle:
            if bd_id == 'FixedNull' or bd_id == 'PracNull':
                special_list = ["omafiets", "crossbike", "low_rider", "ninja", "yzf", "microlino", "zx125", "century"]
                sim_world = client.get_world()
                print("VEHICLE List ", veh_list)
                if not is_reverse:
                    print("used len ", used_len, " Loop ", loop)
                    print("Lopp ", lopp)
                    vehicle_bp = veh_list[lopp - loop]
                else:
                    vehicle_bp = veh_list[lopp - loop]  # 1 # 16]  #    # 2

                if (vehicle_bp in special_list):
                    is_special = True
                    print("Special type car")
                else:
                    is_special = False
            else:
                special_list = ["omafiets", "crossbike", "low_rider", "ninja", "yzf", "microlino", "zx125", "century"]
                sim_world = client.get_world()
                print("VEHICLE List ", veh_list)
                if not is_reverse:
                    print("used len ", used_len, " Loop ", loop)
                    print("Lopp ", lopp)
                    vehicle_bp = veh_list[lopp - loop + used_len]

                else:
                    vehicle_bp = veh_list[lopp - loop + used_len]  # 1 # 16]  #    # 2

                print("Selected vehicle ", vehicle_bp.tags)
                if (vehicle_bp.tags[2] in special_list) or (
                        vehicle_bp.tags[1] in special_list) or (
                        vehicle_bp.tags[0] in special_list):
                    is_special = True
                    print("Special type car")
                else:
                    is_special = False

        if is_vehicle:
            if not is_special:
                print("BRANCH 1")
                ratio = 6.7 / ran_speed
                if ran_speed <= 4.1:
                    ratio = ratio * 2.7
                elif ran_speed <= 4.5:
                    ratio = ratio * 2.4
                elif ran_speed <= 5:
                    ratio = ratio * 2.3
                elif ran_speed <= 6:
                    ratio = ratio * 2
                elif ran_speed <= 6.4:
                    ratio = ratio * 1.8
            else:
                print("BRANCH 2")

                ratio = 6.7 / ran_speed
                if ran_speed <= 4.5:
                    ratio = ratio * 3.3
                elif ran_speed <= 5:
                    ratio = ratio * 3
                elif ran_speed <= 6:
                    ratio = ratio * 2.7
                elif ran_speed <= 6.4:
                    ratio = ratio * 2.2
                else:
                    ratio = ratio * 1.7

        else:
            # ran_speed = random.uniform(2, 3.6)
            print("BRANCH 3")
            ratio = 3.6 / ran_speed
            if ran_speed <= 2:
                ratio = ratio * 5.5
            elif ran_speed <= 2.2:
                ratio = ratio * 5
            elif ran_speed <= 2.5:
                ratio = ratio * 4.5
            elif ran_speed <= 2.8:
                ratio = ratio * 4.2
            elif ran_speed <= 3.3:
                ratio = ratio * 4
            elif ran_speed <= 3.6:
                ratio = ratio * 3.5

        print("ran ratio ", ran_speed, " / ", ratio)
        if frame_rate == 60:
            weighted_wait_tick = int(30 * ratio)  # 60帧下为 int(30 * ratio)
        else:
            weighted_wait_tick = int(15 * ratio)  # 30 帧下为15  # - loop

        if no_collision:
            weighted_wait_tick -= 20
            start = start_tick[loop] - 20
            print("no collision")

        print("Vehicle wait_tick is:", wait_tick, " with speed ", ran_speed)
        print("Player wait tick is: ", weighted_wait_tick, " with ratio ", ratio)

        player = world.player

        bd_dist_x = 183.63  # 183.334 # 18363.0
        bd_dist_z = 0.325
        player_dist_z = 1.421

        turn_left_x = 186.78
        turn_left_z = 0.3
        turn_right_x = 191.80
        turn_right_z = 0.3
        turn_height_x = 189.80
        turn_height_z = 0.3
        if is_reverse:
            bd_dist_y = 230.87  # 230.570 # 23087.0
            player_dist_x = 190.1  # 190.75 #  189.8 + 0.3  # => camera view shift

            turn_left_y = 245.90
            turn_right_y = 243.00
            turn_height_y = 320.20
        else:
            bd_dist_y = 197.76  # 197.777 # 19776.0
            player_dist_x = 189.5  # 190.75 # 189.8 - 0.3  # => camera view shift

            # DD AOI
            turn_left_y = 182.50
            turn_right_y = 185.55
            turn_height_y = 109.00

        if not no_vehicle:
            collision_tick = -1  # 初始化；collision为录制结束帧

        else:
            # rander = int(random.uniform(-5, 5))
            print("before error: ", len(reaction_time), " full list: ", reaction_time)
            if not is_reverse:
                # start, collision_tick = reaction_time[1 - loop]  # 2
                start, collision_tick = reaction_time[lopp - loop]  # 7 # 2
            else:
                start, collision_tick = reaction_time[lopp * 2 - loop + 1]  # 14 #5

        record_toggled = False
        first_collide = True
        AOI = []
        AOI.append(("ticks",
                    "Vehicle_Position_far_x", "Vehicle_Position_far_y",
                    "Vehicle_Position_close_x", "Vehicle_Position_close_y",
                    "BD_position_x", "bd_position_y",
                    "turn_left_x", "turn_left_y",
                    "turn_right_x", "turn_right_y",
                    "turn_height_x", "turn_height_y"))

        record_tick = -1  # initialise as -1, 从下一帧开始record


        # alpha = 45 * math.pi / 180
        # print("alpha", alpha)

        player_speed = 13.8889  # 50km/h = 13.8889m/s

        while True:
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(30)  # 设置为30帧
            if controller.parse_events(client, world, clock, args.sync):
                return
            world.tick(clock)
            world.render(display)

            temp = player.get_transform()
            player_dist_y = temp.location.y

            bd_position_x, bd_position_y = AOI_Cal(player_dist_x, bd_dist_x,
                                                   player_dist_y, bd_dist_y,
                                                   player_dist_z, bd_dist_z)

            turn_left_position_x, turn_left_position_y = AOI_Cal(player_dist_x, turn_left_x,
                                                                  player_dist_y, turn_left_y,
                                                                  player_dist_z, turn_left_z)
            turn_right_position_x, turn_right_position_y = AOI_Cal(player_dist_x, turn_right_x,
                                                                    player_dist_y, turn_right_y,
                                                                    player_dist_z, turn_right_z)
            turn_height_position_x, turn_height_position_y = AOI_Cal(player_dist_x, turn_height_x,
                                                                      player_dist_y, turn_height_y,
                                                                      player_dist_z, turn_height_z)
            # print("BD AOI x ", bd_position_x, " y ",bd_position_y)

            if not no_vehicle:
                temp = vehicle.bounding_box.get_world_vertices(vehicle.get_transform())

                if is_reverse:
                    vehicle_dist_x1 = temp[1].x
                    vehicle_dist_y = temp[1].y
                    vehicle_dist_z1 = temp[1].z
                    vehicle_dist_x2 = temp[4].x
                    vehicle_dist_z2 = temp[4].z
                else:
                    # not reverse
                    vehicle_dist_x1 = temp[3].x
                    vehicle_dist_y = temp[3].y
                    vehicle_dist_z1 = temp[3].z
                    vehicle_dist_x2 = temp[6].x
                    vehicle_dist_z2 = temp[6].z

                on_screen_position1_x, on_screen_position1_y = AOI_Cal(player_dist_x, vehicle_dist_x1,
                                                                       player_dist_y, vehicle_dist_y,
                                                                       player_dist_z, vehicle_dist_z1)

                on_screen_position2_x, on_screen_position2_y = AOI_Cal(player_dist_x, vehicle_dist_x2,
                                                                       player_dist_y, vehicle_dist_y,
                                                                       player_dist_z, vehicle_dist_z2)

                # print("AOI 1 xx ", on_screen_position1_x, " yy ", on_screen_position1_y)
                # print("AOI 2 xx ", on_screen_position2_x, " yy ", on_screen_position2_y)
                if no_collision:
                    if tick == start:
                        record_toggled = True
                        world.camera_manager.toggle_recording()
                        # collision_tick = start + reaction + 40
                        print("No collision recroding start at %d" % (tick))
                elif is_reverse:
                    if (on_screen_position2_x < bd_position_x) and (not record_toggled):
                        record_toggled = True
                        start = tick
                        world.camera_manager.toggle_recording()
                        print("vehicle in ext view at tick", tick)
                        # if (tick <= weighted_wait_tick):
                        #     print("主车出发时间晚于行人")
                else:
                    if (on_screen_position2_x > bd_position_x) and (not record_toggled):
                        record_toggled = True
                        start = tick
                        world.camera_manager.toggle_recording()
                        print("vehicle in ext view at tick", tick)

            else:
                if tick == start:
                    record_toggled = True
                    world.camera_manager.toggle_recording()
                    # collision_tick = start + reaction + 30
                    print("No vehicle recroding start at %d" % (tick))  # , rander))
                on_screen_position1_x = "No vehicle"
                on_screen_position2_x = "No vehicle"
                on_screen_position1_y = "No vehicle"
                on_screen_position2_y = "No vehicle"

            if record_toggled:
                # print("y dist ", y_dist, " / vehicle dist ",v_dist)
                record_tick += 1
                AOI.append((tick, on_screen_position1_x, on_screen_position1_y,
                            on_screen_position2_x, on_screen_position2_y,
                            bd_position_x, bd_position_y,
                            turn_left_position_x, turn_left_position_y,
                            turn_right_position_x, turn_right_position_y,
                            turn_height_position_x, turn_height_position_y))

            if is_vehicle:
                if tick == weighted_wait_tick:
                    player.enable_constant_velocity(carla.Vector3D(x=player_speed, y=0, z=0))

                if (tick == wait_tick) and (not no_vehicle):
                    vehicle.enable_constant_velocity(carla.Vector3D(x=ran_speed, y=0, z=0))
                # player.apply_physics_control(physics_control)
            else:
                if tick == weighted_wait_tick:
                    player.enable_constant_velocity(carla.Vector3D(x=player_speed, y=0, z=0))

                if (tick == wait_tick) and (not no_vehicle):
                    # world.constant_velocity_enabled = True
                    walker_control = carla.WalkerControl()
                    walker_control.speed = ran_speed
                    vehicle.apply_control(walker_control)

            global collision_happen
            if collision_happen and first_collide:
                first_collide = False
                collision_tick = tick + 10  # 碰撞后1s停止
                reaction_time.append((start, collision_tick))
                print("Collision First Tick")
                temp = vehicle.get_transform()
                print(" vehicle position = ", temp.location, " ", is_reverse)
                temp = player.get_transform()
                print(" player position = ", temp.location, " ", is_reverse)

                AOI.append((tick, "COLLISION", 'stops at ' + str(collision_tick),
                            "", "", "", "", "", "", "", "", "", ""))

            # if collision_happen:
            #     # player_speed -= 0.1
            #     player.set_target_velocity(carla.Vector3D(x=player_speed, y=0, z=0))  # speed in other direction
            if tick == collision_tick:
                world.camera_manager.toggle_recording()
                # global video_Writer
                # video_Writer.release()
                print("Stop recording at tick ", tick)
                if not no_vehicle:
                    vehicle.destroy()
                break

            if tick == 240:  # 强制录制结束
                world.camera_manager.toggle_recording()

                if not no_vehicle:
                    print("NO COLLISION ! ! ! ")
                    vehicle.destroy()
                else:
                    print("Ending with no vehicle")
                break

            # UE4 中 t.MaxFPS 30 限制帧数
            pygame.display.flip()

            tick += 1

        folder_name = 'bd' + str(bd_id)
        if is_vehicle:
            folder_name += '_veh'
        else:
            folder_name += '_wal'
        if is_reverse:
            folder_name += '_rev'
        # if is_special:
        #     folder_name += '_spe'
        if no_vehicle:
            folder_name += '_noveh'
        folder_add = r"M:\Carla\carla\PythonAPI\examples\record_out\bd%s\%s" % (bd_id, folder_name)
        print("folder_add ", folder_add)
        # folder_add = '/record_out/bd' + str(bd_id) + '/' + folder_name + '/'
        if not os.path.exists(folder_add):
            os.makedirs(folder_add)
        if bd_id == 'FixedNull':
            bd_num = 5 - loop // 5
            folder_add = r"%s\%d_%s_%.1f_%d" % (
                folder_add, bd_num, folder_name, ran_speed, record_tick)  # loop for testing, ran_speed for actual
        else:
            folder_add = r"%s\%s_%.1f_%d" % (
                folder_add, folder_name, ran_speed, record_tick)  # loop for testing, ran_speed for actual

        if record_image != []:
            if (len(record_image) != record_tick):
                print("recorded tick ", len(record_image), " is not equal to record tick ", record_tick)

                AOI.append((tick, 'Missing ' + str(record_tick - len(record_image)) + 'tick(s)',
                            '', '', '', '', '', '', '', '', '', '', ''))
                if (record_tick - len(record_image)) > 2:
                    print("MIssing too many ticks ", record_tick - len(record_image))
            frameSize = (1920, 1080)

            # video_Writer = cv2.VideoWriter(folder_add + '.mp4',
            #                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30,
            #                                frameSize)
            # for im in record_image:
            #     # im.save_to_disk('record_out/'+folder_name+'/%08d' % im.frame)
            #     im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            #     video_Writer.write(im)
            # for im in image_set:
            # im.save_to_disk('record_out/' + folder_name + '1/%08d' % im.frame)
            # print(len(image_set))
            video_Writer.release()
            # print(AOI)
            # np.savetxt('record_out/bd' + str(bd_id) + '/' + folder_name + '/' + folder_name + '_AOI_' + str(loop) +
            #            '.csv', AOI, delimiter=',  ', fmt='% s')
            print(AOI)
            np.savetxt(folder_add + '_AOI.csv', AOI, delimiter=',  ', fmt='% s')

    finally:

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        # default='1280x720',
        # help='window resolution (default: 1280x720)')
        # default='3840x800',
        # help='window resolution (default: 3840x800)')
        # default='5760x1080',
        # help='window resolution (default: 5760x1080)')
        default='1920x1080',
        help='window resolution (default: 1920x1080)')

    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        # default='vehicle.*',
        # help='actor filter (default: "vehicle.*")')
        default='vehicle.tesla.model3',  # select tesla model3 for consistency
        help='actor filter (default: "vehicle.tesla.model3")')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        global is_reverse
        global record_image
        global image_set
        global collision_happen
        global is_vehicle
        global is_special
        global no_vehicle

        global reaction_time
        reaction_time = []

        is_reverse = False
        is_vehicle = True

        # is_special = True
        no_vehicle = False

        global bd_id

        # if is_vehicle:
        #     speed_list = [6.7, 6.1, 5.6, 5.1, 4.9, 4.5, 4.1]
        #
        # else:
        #     speed_list = [2.8, 2.9, 3.1, 3.3, 3.4, 3.5, 3.6]# 2.8最低, 2.7概率车辆出发时间晚于行人出现

        # start =         [83, 75, 69, 90, 92, 78,  68,  90, 92, 80, 91, 56, 72,  86, 84]
        # start_rev =     [81, 73, 68, 88, 90, 77,  66,  89, 90, 79, 90, 54, 71,  84, 83]
        # reaction =      [95, 86, 82, 97, 84, 120, 108, 98, 91, 86, 91, 83, 117, 78, 94]
        # reaction_rev =  [97, 88, 82, 98, 86, 121, 111, 98, 92, 87, 91, 85, 117, 79, 94]
        # speed_list = [4.9, 5.6, 6.1, 4.1, 5.1, 2.8, 3.6, 4.1, 4.5, 5.1, 4.5, 6.7, 3.2, 5.6, 4.9]
        start = [72, 63, 73, 79, 81]
        start_rev = [68, 63, 71, 77, 81]
        reaction = [79, 76, 115, 84, 103]
        reaction_rev = [82, 77, 118, 86, 103]
        speed_list = [6.1, 6.7, 3.2, 5.1, 4.1]
        bd_id = 'FixedNull'
        # bd_id = 'PracNull'
        # bd_id = '0'
        speed_list = [4.9, 5.6, 6.1, 5.1, 4.5]  # Fixed 0
        # speed_list = [4.1, 5.1, 5.6, 4.9, 2.8]  # Fixed 1
        # speed_list = [3.6, 4.1, 4.5, 6.7, 3.2]  # Fixed 2
        # speed_list = [5.1, 4.5, 4.1, 5.1, 2.8]  # Fixed 3
        # speed_list = [6.7, 3.2, 4.9, 5.6, 6.1]  # Fixed 4
        # speed_list = [5.6, 4.9, 3.6, 4.1, 4.5]  # Fixed 5
        # speed_list = [4.1, 4.5]
        # speed_list = [6.1]  # Practice 1
        # speed_list = [6.7]  # Practice 2
        # speed_list = [3.2]  # practice 3
        # speed_list = [5.1]  # practice 4
        # speed_list = [4.1]  # practice 5
        speed_list = [5.6, 4.9, 3.6, 4.1, 4.5,
                      6.7, 3.2, 4.9, 5.6, 6.1,
                      5.1, 4.5, 4.1, 5.1, 2.8,
                      3.6, 4.1, 4.5, 6.7, 3.2,
                      4.1, 5.1, 5.6, 4.9, 2.8,
                      4.9, 5.6, 6.1, 5.1, 4.5
                      ]
        speed_list = [4.5, 4.1, 3.6, 4.9, 5.6,
                      6.1, 5.6, 4.9, 3.2, 6.7,
                      2.8, 5.1, 4.1, 4.5, 5.1,
                      3.2, 6.7, 4.5, 4.1, 3.6,
                      2.8, 4.9, 5.6, 5.1, 4.1,
                      4.5, 5.1, 6.1, 5.6, 4.9
                      ]
        global veh_list
        global used_len
        veh_list = ['carlacola', 'impala', 'patrol_2021',
                    'sprinter', 'ninja', '0019',
                    '0022', 'cybertruck', 'a2',
                    'coupe_2020', 'cooper_s',
                    'fusorosa', '0005',
                    'zx125', 'mustang']  # Fixed
        veh_list = ['tt', 'charger_police', '0029', 'mkz_2017', 'firetruck',  # prac
                    'carlacola', 'impala', 'patrol_2021',
                    'sprinter', 'ninja', '0019',
                    '0022', 'cybertruck', 'a2',
                    'coupe_2020', 'cooper_s',
                    'fusorosa', '0005',
                    'zx125', 'mustang',
                    ]
        veh_list = ['prius', 'tt', '0041', 'patrol_2021', 'grandtourer',  # 5
                    'leon', 'a2', 'ambulance', '0016', 'mustang',  # 4
                    '0005', 'sprinter', 'firetruck', 't2', 'coupe',  # 3
                    '0039', 'cooper_s', 'charger_police', 'mkz_2017', '0029',  # 2
                    '0031', 'crossbike', 'firetruck', 'model3', 'etron',  # 1
                    'charger_2020', 'c3', 'carlacola', 'crown', 'coupe_2020'  # 0
                     ]
        used_len = len(veh_list)
        veh_list = ['crossbike']
        speed_list = [4.9]

        global lopp
        lopp = len(speed_list) - 1

        # for m in range(0, 2):
        for i in range(0, 2):
            for j in range(0, 2):
                # loop = 6  # 6
                # loop = len(speed_list) - 1  # 2
                loop = lopp
                print('Reverse Map %s' % ('On' if is_reverse else 'Off'))
                while loop >= 0:  # 0
                    # initialize
                    ran_speed = speed_list[loop]
                    print("current speed is ", ran_speed)
                    record_image = []
                    image_set = []
                    collision_happen = False

                    game_loop(args, loop, ran_speed)
                    loop -= 1

                is_reverse = not is_reverse

            no_vehicle = not no_vehicle
            print('Reverse Map %s' % ('On' if is_reverse else 'Off'))
            print("Current reaction time: ", reaction_time)
            if no_vehicle:
                print("No Vehicle Mode")

        print("Used tags: ")
        for veh in veh_list:
            if type(veh) == str:
                print(veh)
            else:
                print(veh.tags)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
