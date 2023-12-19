#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import math

import numpy as np
import pygame
import random
import time
from skimage.transform import resize

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import carla

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *
import wandb

from gym_carla.ComputerVision import ComputerVision
from gym_carla.Managers import DisplayManager, CameraManager, RadarManager
from gym_carla.controller2d import Controller2D

wandb.init(project="CARLA_RL")


class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self):
        self.waypoints = None
        print('------------------------------------INIT------------------------------------------\n\n\n')
        env_config = {
            'number_of_vehicles': 100,
            'number_of_walkers': 0,
            'display_size': 256,  # screen size of bird-eye render
            'max_past_step': 1,  # the number of past steps to draw
            'dt': 0.1,  # time interval between two frames
            'discrete': False,  # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
            'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
            'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
            'port': 2000,  # connection port
            'town': 'Town03',  # which town to simulate
            'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
            'max_time_episode': 750,  # maximum timesteps per episode
            'max_waypt': 12,  # maximum number of waypoints
            'obs_range': 32,  # observation range (meter)
            'd_behind': 12,  # distance behind the ego vehicle (meter)
            'out_lane_thres': 2.0,  # threshold for out of lane
            'desired_speed': 8,  # desired speed (m/s)
            'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
            'display_route': True,  # whether to render the desired route
            'radar_fov': 40
        }
        # parameters
        self.display_size = env_config['display_size']  # rendering screen size
        self.max_past_step = env_config['max_past_step']
        self.number_of_vehicles = env_config['number_of_vehicles']
        self.number_of_walkers = env_config['number_of_walkers']
        self.dt = env_config['dt']
        self.task_mode = env_config['task_mode']
        self.max_time_episode = env_config['max_time_episode']
        self.max_waypt = env_config['max_waypt']
        self.obs_range = env_config['obs_range']
        self.d_behind = env_config['d_behind']
        self.obs_size = int(self.obs_range / 0.125)
        self.out_lane_thres = env_config['out_lane_thres']
        self.desired_speed = env_config['desired_speed']
        self.max_ego_spawn_times = env_config['max_ego_spawn_times']
        self.display_route = env_config['display_route']

        self.sensor_tick = self.dt
        self.radar_fov = env_config['radar_fov']

        self.episode_rewards = []
        self.actor_list = []
        self.prev_acc = 0

        self.display_manager = None
        self.camera_manager = None
        self.radar_manager = None

        self.controller = None
        self.steer_method = "PurePursuit"

        # Destination
        if env_config['task_mode'] == 'roundabout':
            self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
        else:
            self.dests = None

        # action and observation spaces
        self.discrete = env_config['discrete']
        self.discrete_act = [env_config['discrete_acc']]  # acc
        self.n_acc = len(self.discrete_act[0])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc)
        else:
            self.action_space = spaces.Box(low=np.array([env_config['continuous_accel_range'][0]]),
                                           high=np.array([env_config['continuous_accel_range'][1]]),
                                           dtype=np.float32)  # acc

        observation_space_dict = {
            'speed': spaces.Box(low=-50, high=50, shape=(1,), dtype=float),
            'distance': spaces.Box(low=0, high=100, shape=(1,), dtype=float),
            'delta_V': spaces.Box(low=-100, high=100, shape=(1,), dtype=float),
            'speed_limit': spaces.Box(low=0, high=50, shape=(1,), dtype=float),
            'is_red_light': spaces.Box(low=0, high=1, shape=(1,), dtype=int),
            'prev_acc': spaces.Box(low=-3, high=3, shape=(1,), dtype=float),
        }

        self.observation_space = spaces.Dict(observation_space_dict)

        # Try to disconnect to connected world
        print('connecting to Carla server...')
        client = carla.Client('localhost', env_config['port'])

        #client = carla.client.make_carla_client('localhost', env_config['port'])
        self.client = client
        try:
            client.unload_world()
        except:
            pass

        # Connect to carla server and get world object
        client.set_timeout(10.0)
        self.world = client.load_world(env_config['town'])
        print('Carla server connected!')

        # Create a Traffic Manager
        self.traffic_manager = client.get_trafficmanager()

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # Initialize spectator
        self.spectator = self.world.get_spectator()

        # Get spawn points
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self.walker_spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_blueprint(env_config['ego_vehicle_filter'], color='49,8,8')

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Display manager
        display_width, display_height = [1280, 720]
        self.display_manager = DisplayManager(grid_size=[1, 1], window_size=[display_width, display_height])

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # Initialize the renderer
        # self._init_renderer()
        print('init end')

    def reset(self):
        print(f'-------------------------------------RESET {self.reset_step}--------------------------------------')

        # Reset previous speed and acc (output of RL agent)
        self.prev_speed = 0
        self.prev_acc = 0

        if self.reset_step != 0:
            # Log total episode reward
            wandb.log({"episode_reward": sum(self.episode_rewards)})
            self.episode_rewards = []

        # Delete sensors, vehicles and walkers
        for actor in self.actor_list:
            try:
                actor.stop()
            except:
                pass

            try:
                actor.destroy()
            except:
                pass
        self.actor_list = []

        # Clear sensor objects
        self.display_manager.destroy()
        self.collision_sensor = None
        self.camera_manager = None
        self.radar_manager = None
        self.display_manager = None

        # Display manager
        display_width, display_height = [1280, 720]
        self.display_manager = DisplayManager(grid_size=[1, 1], window_size=[display_width, display_height])

        # Disable sync mode
        self._set_synchronous_mode(True)

        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles

        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                    count -= 1
                if count <= 0:
                    break

        while count > 0:
            if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
                count -= 1

        # Spawn pedestrians
        random.shuffle(self.walker_spawn_points)
        count = self.number_of_walkers
        if count > 0:
            for spawn_point in self.walker_spawn_points:
                if self._try_spawn_random_walker_at(spawn_point):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
                count -= 1

        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)

        # Spawn the ego vehicle
        ego_spawn_times = 0
        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                print('-----------------------------RESET IN RESET-------------------------\n\n\n', flush=True)
                self.reset()

            if self.task_mode == 'random':
                transform = random.choice(self.vehicle_spawn_points)
            if self.task_mode == 'roundabout':
                self.start = [52.1 + np.random.uniform(-5, 5), -4.2, 178.66]  # random
                # self.start=[52.1,-4.2, 178.66] # static
                transform = set_carla_transform(self.start)
            if self._try_spawn_ego_vehicle_at(transform):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)

        self._spawn_sensors()

        while True:
            try:
                # Add collision sensor
                self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
                self.actor_list.append(self.collision_sensor)
                self.collision_sensor.listen(lambda event: get_collision_hist(event))
                break
            except:
                print('failed to add collision sensor')


        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_hist = []

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        # Linear interpolation to improve the results:
        # Path interpolation parameters
        INTERP_MAX_POINTS_PLOT = 10  # number of points used for displaying
        # lookahead path
        INTERP_LOOKAHEAD_DISTANCE = 20  # lookahead in meters
        INTERP_DISTANCE_RES = 0.01  # distance between interpolated points
        # Linear interpolation computations
        waypoints_np = np.array(self.waypoints)
        # Compute a list of distances between waypoints
        wp_distance = []   # distance array
        for i in range(1, waypoints_np.shape[0]):
            wp_distance.append(
                    np.sqrt((waypoints_np[i, 0] - waypoints_np[i-1, 0])**2 +
                            (waypoints_np[i, 1] - waypoints_np[i-1, 1])**2))
        wp_distance.append(0)  # last distance is 0 because it is the distance
                               # from the last waypoint to the last waypoint

        # Linearly interpolate between waypoints and store in a list
        wp_interp      = []    # interpolated values
                               # (rows = waypoints, columns = [x, y, v])
        wp_interp_hash = []    # hash table which indexes waypoints_np
                               # to the index of the waypoint in wp_interp
        interp_counter = 0     # counter for current interpolated point index
        for i in range(waypoints_np.shape[0] - 1):
            # Add original waypoint to interpolated waypoints list (and append
            # it to the hash table)
            wp_interp.append(list(waypoints_np[i]))
            wp_interp_hash.append(interp_counter)
            interp_counter+=1

            # Interpolate to the next waypoint. First compute the number of
            # points to interpolate based on the desired resolution and
            # incrementally add interpolated points until the next waypoint
            # is about to be reached.
            num_pts_to_interp = int(np.floor(wp_distance[i] /\
                                         float(INTERP_DISTANCE_RES)) - 1)
            wp_vector = waypoints_np[i+1] - waypoints_np[i]
            wp_uvector = wp_vector / np.linalg.norm(wp_vector)
            for j in range(num_pts_to_interp):
                next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                wp_interp.append(list(waypoints_np[i] + next_wp_vector))
                interp_counter+=1
        # add last waypoint at the end
        wp_interp.append(list(waypoints_np[-1]))
        wp_interp_hash.append(interp_counter)
        interp_counter+=1

        self.waypoints_interpolated = wp_interp

        # Set the path for the autopilot
        location_list = [carla.Location(x=loc[0], y=loc[1], z=loc[2]) for loc in self.waypoints]
        self.traffic_manager.set_path(self.ego, location_list)

        # As the autopilot does not work correctly, we use the Controller2D to control the steering of the ego vehicle
        self.controller = Controller2D(self.waypoints_interpolated, self.steer_method)

        return self._get_obs()

    def step(self, action):
        #print(f'------------------------------------STEP {self.time_step} --------------------------------------')
        # return (self._get_obs(), 0, False, {'waypoints': 0, 'vehicle_front': 0})

        # Calculate acceleration and steering
        if self.discrete:
            acc = self.discrete_act[0][action // self.n_steer]
        else:
            acc = action[0]

        # Convert acc to value between -3 and 3 and to throttle and brake values
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            brake = np.clip(-acc / 3, 0, 1)
            throttle = 0

        _, steer, _ = self.controller.get_commands()

        # Apply control
        #act = carla.VehicleControl(throttle=throttle, steer=self.ego.get_control().steer, brake=brake)
        sensitivity = 3
        act = carla.VehicleControl(throttle=float(throttle), steer=float(np.clip(sensitivity*steer, -1, 1)), brake=float(brake))
        self.ego.apply_control(act)

        self.world.tick()

        # Update the spectator's position to follow the ego vehicle
        transform = carla.Transform(self.ego.get_transform().transform(carla.Location(x=-4, z=2.5)),
                                    self.ego.get_transform().rotation)
        self.spectator.set_transform(transform)
        self.world.tick()

        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        # route planner
        self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

        #print("Waypoints:", self.waypoints)

        # Linear interpolation to improve the results:
        # Path interpolation parameters
        INTERP_MAX_POINTS_PLOT = 10  # number of points used for displaying
        # lookahead path
        INTERP_LOOKAHEAD_DISTANCE = 20  # lookahead in meters
        INTERP_DISTANCE_RES = 0.01  # distance between interpolated points
        # Linear interpolation computations
        waypoints_np = np.array(self.waypoints)
        # Compute a list of distances between waypoints
        wp_distance = []  # distance array
        for i in range(1, waypoints_np.shape[0]):
            wp_distance.append(
                np.sqrt((waypoints_np[i, 0] - waypoints_np[i - 1, 0]) ** 2 +
                        (waypoints_np[i, 1] - waypoints_np[i - 1, 1]) ** 2))
        wp_distance.append(0)  # last distance is 0 because it is the distance
        # from the last waypoint to the last waypoint

        # Linearly interpolate between waypoints and store in a list
        wp_interp = []  # interpolated values
        # (rows = waypoints, columns = [x, y, v])
        wp_interp_hash = []  # hash table which indexes waypoints_np
        # to the index of the waypoint in wp_interp
        interp_counter = 0  # counter for current interpolated point index
        for i in range(waypoints_np.shape[0] - 1):
            # Add original waypoint to interpolated waypoints list (and append
            # it to the hash table)
            wp_interp.append(list(waypoints_np[i]))
            wp_interp_hash.append(interp_counter)
            interp_counter += 1

            # Interpolate to the next waypoint. First compute the number of
            # points to interpolate based on the desired resolution and
            # incrementally add interpolated points until the next waypoint
            # is about to be reached.
            num_pts_to_interp = int(np.floor(wp_distance[i] / \
                                             float(INTERP_DISTANCE_RES)) - 1)
            wp_vector = waypoints_np[i + 1] - waypoints_np[i]
            wp_uvector = wp_vector / np.linalg.norm(wp_vector)
            for j in range(num_pts_to_interp):
                next_wp_vector = INTERP_DISTANCE_RES * float(j + 1) * wp_uvector
                wp_interp.append(list(waypoints_np[i] + next_wp_vector))
                interp_counter += 1
        # add last waypoint at the end
        wp_interp.append(list(waypoints_np[-1]))
        wp_interp_hash.append(interp_counter)
        interp_counter += 1

        # Update the waypoints of the controller
        self.controller.update_waypoints(wp_interp)

        # Update pose, timestamp
        # measurement_data, _ = self.client.read_data()
        # current_x, current_y, current_yaw = \
        # get_current_pose(measurement_data)
        # current_speed = measurement_data.player_measurements.forward_speed
        # current_timestamp = float(measurement_data.game_timestamp) / 1000.0


        vehicle_transform = self.ego.get_transform()
        current_x = vehicle_transform.location.x
        current_y = vehicle_transform.location.y
        current_yaw = np.radians(vehicle_transform.rotation.yaw)
        current_speed = self.ego.get_velocity()

        world_snapshot = self.world.get_snapshot()
        # Get the current simulation time
        current_timestamp = world_snapshot.timestamp.elapsed_seconds

        # Shift coordinates
        if self.steer_method == 'PurePursuit':
            length = -1.5
        elif self.steer_method == 'Stanley' or self.steer_method == 'MPC':
            length = 1.5
        else:
            length = 0.0
        #print("Current yaw:", current_yaw)
        current_x, current_y = self.controller.get_shifted_coordinate(current_x, current_y, current_yaw, length)

        #print("Current location:", current_x, current_y)
        #print("Next waypoints:", self.waypoints[0])
        closest_distance = np.linalg.norm(np.array([
            self.waypoints[0][0] - current_x,
            self.waypoints[0][1] - current_y]))

        # Update the other controller values and controls
        self.controller.update_values(current_x, current_y, current_yaw,
                                      current_speed.length(),
                                      current_timestamp, True, closest_distance)
        self.controller.update_controls()

        # state information
        info = {
            'waypoints': self.waypoints,
            'vehicle_front': self.vehicle_front
        }

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        # Log single step reward
        reward = self._get_reward()
        self.episode_rewards.append(reward)
        wandb.log({"step_reward": reward})

        # print('step end')
        return (self._get_obs(), reward, self._terminal(), copy.deepcopy(info))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode):
        pass

    def _create_vehicle_blueprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

    Returns:
      bp: the blueprint object of carla.
    """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if
                                                     int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
    """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
        blueprint = self._create_vehicle_blueprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        self.actor_list.append(vehicle)
        if vehicle is not None:
            vehicle.set_autopilot()
            return True
        return False

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)
        self.actor_list.append(walker_actor)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(1 + random.random())  # max speed between 1 and 2 (default is 1.4 m/s)
            return True
        return False

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)
            self.actor_list.append(vehicle)

        if vehicle is not None:
            #vehicle.set_autopilot(False)
            self.ego = vehicle
            self.actor_list.append(vehicle)
            self.ego.show_debug_telemetry()
            return True

        return False

    def _spawn_sensors(self):
        # Create the ComputerVision object
        self.computer_vision = ComputerVision(self.ego, 1 / self.sensor_tick)

        # Then, SensorManager is used to spawn RGBCamera and Radar and assign each of them to a grid position.
        self.camera_manager = CameraManager(self.world, self.display_manager,
                                            carla.Transform(carla.Location(x=1, z=1.5), carla.Rotation(yaw=+00)),
                                            self.ego, {'sensor_tick': f'{self.sensor_tick}'},
                                            display_pos=[0, 0],
                                            computer_vision=self.computer_vision)
        self.radar_manager = RadarManager(self.world, self.display_manager,
                                          carla.Transform(carla.Location(x=1, z=1.5)),
                                          self.ego,
                                          {'horizontal_fov': f'{self.radar_fov}', 'points_per_second': '5000',
                                           'range': '100',
                                           'sensor_tick': '0.1', 'vertical_fov': f'{self.radar_fov}'},
                                          display_pos=[0, 0],
                                          computer_vision=self.computer_vision)

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

    Args:
      filt: the filter indicating what type of actors we'll look at.

    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_obs(self):
        # return np.zeros(shape=(10,), dtype=np.float32)

        """Get the observations."""
        self.computer_vision.process_data()

        # First draw camera on the screen, then the radar
        self.camera_manager.draw_camera()
        self.radar_manager.draw_radar()

        # Render received data
        self.display_manager.render()
        """
        obs = {
            'speed': np.array(self.ego.get_velocity().length(), dtype=np.float32),
            'distance': np.array(self.computer_vision.get_distance(), dtype=np.float32),
            'delta_V': np.array(self.computer_vision.get_delta_v(), dtype=np.float32),
            'speed_limit': np.array(50.0, dtype=np.float32),
            'is_red_light': np.array(1 if self.computer_vision.get_red_light() else 0, dtype=int)
        }"""
        obs = {
            'speed': np.reshape(np.array(self.ego.get_velocity().length(), dtype=float), (1,)),
            'distance': np.reshape(np.array(self.computer_vision.get_distance(), dtype=float), (1,)),
            'delta_V': np.reshape(np.array(self.computer_vision.get_delta_v(), dtype=float), (1,)),
            'speed_limit': np.reshape(np.array(self.desired_speed, dtype=float), (1,)),
            'is_red_light': np.reshape(np.array(1 if self.computer_vision.get_red_light() else 0, dtype=int), (1,)),
            'prev_acc': np.reshape(np.array(self.prev_acc, dtype=float), (1,))
        }

        return obs

    def _get_reward(self):
        """Calculate the step reward."""
        # return 1
        # reward for speed tracking
        speed = self.ego.get_velocity().length()

        # Calculate the acceleration and the change in acceleration to make the ride smooth and energy efficient
        acceleration = abs(speed - self.prev_speed)
        change_in_acc = abs(acceleration - self.prev_acc)
        self.prev_speed = speed
        self.prev_acc = acceleration

        # reward for collision
        collision = 1 if len(self.collision_hist) > 0 else 0

        # cost for too fast
        to_fast = 1 if speed > self.desired_speed else 0

        # Ideal following distance
        following_vehicle_speed = self.computer_vision.get_delta_v()
        following_distance = self.computer_vision.get_distance()
        if following_vehicle_speed is None or following_distance is None:
            following_distance_error = 0
        else:
            ideal_following_distance = 5 + 2 * following_vehicle_speed
            if following_distance == 100 or following_distance > ideal_following_distance:  # No vehicle in front
                following_distance_error = 0
            else:
                # How much to close is the ego vehicle to the vehicle in front (max 30m to close)
                following_distance_error = min(abs(following_distance - ideal_following_distance), 30)

        # If the vehicle doesn't move, the negative reward for acceleration (braking) is not necessary
        if speed < 0.01:
            acceleration = 0
            change_in_acc = 0

        if collision:
            reward = -10000
        else:
            print('v', speed, ', a', acceleration, ', da', change_in_acc, ', follow_e', following_distance_error)
            reward = (1.5 * speed) - (10 * to_fast * (speed - self.desired_speed) + 3 * acceleration + 1.5 * change_in_acc + following_distance_error)

        return reward

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # print('--------------------------------TERMINAL-----------------------------------------\n\n\n')
        # Get ego state
        ego_x, ego_y = get_pos(self.ego)

        # If collides
        if len(self.collision_hist) > 0:
            print('TERMINATION - collision')
            return True

        # If reach maximum timestep
        if self.time_step > self.max_time_episode:
            print('TERMINATION - max timestep reached')
            return True

        # If at destination
        if self.dests is not None:  # If at destination
            for dest in self.dests:
                if np.sqrt((ego_x - dest[0]) ** 2 + (ego_y - dest[1]) ** 2) < 4:
                    print('TERMINATION - at destination')
                    return True

        # If out of lane
        '''dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.out_lane_thres:
            print('TERMINATION - out of lane')
            return True'''
        # print('No termination')
        return False

def get_current_pose(measurement):
    """Obtains current x,y,yaw pose from the client measurements

    Obtains the current x,y, and yaw pose from the client measurements.

    Args:
        measurement: The CARLA client measurements (from read_data())

    Returns: (x, y, yaw)
        x: X position in meters
        y: Y position in meters
        yaw: Yaw position in radians
    """
    x   = measurement.player_measurements.transform.location.x
    y   = measurement.player_measurements.transform.location.y
    yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)

    return (x, y, yaw)