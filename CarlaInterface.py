"""
Carla interface testing file
**
..
**
"""

import glob
import os
import re
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import time
import math
import random
import numpy as np
import csv

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')

    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))

    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()


# The DisplayManager is in charge of displaying the data collected from the sensors on the screen.
class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0] / self.grid_size[1]), int(self.window_size[1] / self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display is not None


def save_rbg_image_to_disk(image):
    image.save_to_disk(f"RGBCameraData/{image.frame}.png")


def save_radar_data_to_disk(data_points):
    if not os.path.isfile('./RadarData.csv'):
        print(f" RadarData.csv created. ")
        # We create a csv file to save our radar-data.
        # Let's define the headings of our csv file and save them.
        header = ['Velocity', 'Altitude', 'Azimuth', 'Depth']
        with open('RadarData.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    # sava radar data to disk
    data_points[:, 1] = np.degrees(data_points[:, 1])  # change Altitude from rad to degrees.
    data_points[:, 2] = np.degrees(data_points[:, 2])  # change Azimuth from rad to degrees.
    data_points[:, 0] = np.round(data_points[:, 0], 0)  # round Velocity.
    data_points[:, 3] = np.round(data_points[:, 3], 0)  # round Depth.
    try:
        with open('RadarData.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(data_points)
    except Exception as e:
        print(f"Error writing row to CSV: {e}")


class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))
            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.display_rgb_image)
            return camera

        elif sensor_type == "Radar":
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
            radar.listen(self.display_radar_image)
            return radar

        else:
            return None

    def get_sensor(self):
        return self.sensor

    def display_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        # Display camera data on screen.
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        # save camera data to disk.
        # save_rbg_image_to_disk(image)

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def display_radar_image(self, radar_data):
        t_start = self.timer.time()
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))
        data_points = points.copy()

        # save radar data to disk.
        # save_radar_data_to_disk(data_points)

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            # give color to radar data: white = neutral; red= move closer; blue=moving away.
            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            velocity_range = 7.5  # m/s
            norm_velocity = detect.velocity / velocity_range  # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            # display radar data on screen.
            if self.display_man.render_enabled():
                self.world.debug.draw_point(
                    radar_data.transform.location + fw_vec,
                    size=0.075,
                    life_time=0.06,
                    persistent_lines=False,
                    color=carla.Color(r, g, b)
                )

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()


def set_town(client):
    # set a random map to spawn our actors in.
    # client.load_world(None)
    maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10', 'Town11', 'Town12']
    new_map = maps[random.randint(0, 10)]
    print('Map: %s' % new_map)
    client.load_world(new_map)


def set_weather(world):
    # set a random weather condition.
    weather = random.choice(find_weather_presets())
    print('Weather: %s' % weather[1])
    world.set_weather(weather[0])


def manual_control(ego_vehicle, throttle, brake):
    control = carla.VehicleControl()

    # Manual throttle and brake
    control.throttle = throttle
    control.brake = brake

    return control


def game_loop(args):
    world = None
    original_settings = None
    display_manager = None
    timer = CustomTimer()
    actor_list = []
    vehicle_list = []

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        # set a random town from our list.
        set_town(client)

        # Getting the world
        world = client.get_world()
        blueprint = world.get_blueprint_library()
        spawnPoints = world.get_map().get_spawn_points()

        # get original_settings to reset the world after use.
        original_settings = world.get_settings()

        if args.sync:
            traffic_manager = client.get_trafficmanager(args.tm_port)
            if args.respawn:
                traffic_manager.set_respawn_dormant_vehicles(True)
            if args.hybrid:
                traffic_manager.set_hybrid_physics_mode(True)
                traffic_manager.set_hybrid_physics_radius(70.0)
            if args.seed is not None:
                traffic_manager.set_random_device_seed(args.seed)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

        # set a random weather preset.
        set_weather(world)

        # Instantiating the vehicle to which we attached the sensors
        transform = random.choice(spawnPoints)
        ego_bp = random.choice(blueprint.filter('vehicle'))
        ego_bp.set_attribute('role_name', 'ego')
        ego_vehicle = world.spawn_actor(ego_bp, transform)
        vehicle_list.append(ego_vehicle)
        print('created ego: %s' % ego_vehicle.type_id)
        ego_vehicle.set_autopilot(True)

        # set the spectator to follow our ego_vehicle.
        spectator = world.get_spectator()
        transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)),
                                    ego_vehicle.get_transform().rotation)
        spectator.set_transform(transform)

        # Display Manager organize all the sensors and its display in a window
        # It can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[1, 1], window_size=[args.width, args.height])

        # Then, SensorManager is used to spawn RGBCamera and Radar and assign each of them to a grid position.
        SensorManager(world, display_manager, 'RGBCamera',
                      carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)),
                      ego_vehicle, {'sensor_tick': '0.0'}, display_pos=[0, 0])

        SensorManager(world, display_manager, 'Radar',
                      carla.Transform(carla.Location(x=0, z=2.4)),
                      ego_vehicle,
                      {'horizontal_fov': '90', 'points_per_second': '1500', 'range': '75',
                       'sensor_tick': '0.0', 'vertical_fov': '60'}, display_pos=[0, 0])

        # But the city now is probably quite empty, let's add a few more vehicles.
        transform.location += carla.Location(x=40, y=-3.2)
        transform.rotation.yaw = -180.0
        number_of_vehicles = 0
        while number_of_vehicles < args.number_of_vehicles:
            transform.location.x += 8.0

            bp = random.choice(blueprint.filter('vehicle'))

            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = world.try_spawn_actor(bp, transform)
            if npc is not None:
                vehicle_list.append(npc)
                npc.set_autopilot(True)
                number_of_vehicles += 1
                print('created %s' % npc.type_id)

        # Set automatic vehicle lights update if specified
        if args.car_lights_on:
            all_vehicle_actors = world.get_actors(vehicle_list)
            for actor in all_vehicle_actors:
                traffic_manager.update_vehicle_lights(actor, True)

        # Simulation loop
        call_exit = False
        while True:
            # controlling throttle and brake
            """ RL output: 
            throttle_input = 0
            brake_input = 0
            control = manual_control(ego_vehicle, throttle=throttle_input, brake=brake_input)
            ego_vehicle.apply_control(control)
            """

            # Update the spectator's position to follow the ego vehicle
            transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)),
                                        ego_vehicle.get_transform().rotation)
            spectator.set_transform(transform)

            # Carla Tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # Render received data
            display_manager.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break

            if call_exit:
                break

    finally:
        if display_manager:
            display_manager.destroy()

        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

        if traffic_manager:
            traffic_manager.set_synchronous_mode(False)

        world.apply_settings(original_settings)


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA World generation')
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
        '-n', '--number-of-vehicles',
        metavar='N',
        default=random.randint(20, 150),
        type=int,
        help='Number of vehicles (between 6 and 150)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=random.randint(2, 50),
        type=int,
        help='Number of walkers (between 2 and 50)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=0,
        type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable automatic car light management')
    argparser.add_argument(
        '--respawn',
        action='store_true',
        default=False,
        help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--async',
        dest='sync',
        action='store_false',
        help='Asynchronous mode execution')
    argparser.set_defaults(sync=True)
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
