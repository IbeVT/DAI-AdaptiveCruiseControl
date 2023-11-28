import datetime
import glob
import os
import sys

from ComputerVision import ComputerVision

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
import datetime

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


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
        return self.display != None


class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos,
                 computer_vision):
        self.computer_vision = computer_vision
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
            camera_bp.set_attribute('sensor_tick', '1.0')

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            # camera.listen(self.save_rgb_image)
            camera.listen(self.draw_bounding_box)
            return camera

        elif sensor_type == "Radar":
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
            # radar.listen(self.save_radar_image)
            radar.listen(self.draw_distance_speed)
            return radar
        else:
            return None

    def get_sensor(self):
        return self.sensor

    def draw_bounding_box(self, image):
        t_start = self.timer.time()
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        # Display camera data on screen.
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        cords = self.computer_vision.get_bounding_box(array)
        # Draw bounding box on screen.
        if cords is not None:
            [x_lower, y_lower, x_upper, y_upper] = cords
            pygame.draw.rect(self.surface, (255, 0, 0), (x_lower, y_lower, x_upper - x_lower, y_upper - y_lower), 2)

        t_end = self.timer.time()
        self.time_processing += (t_end - t_start)
        self.tics_processing += 1

    def draw_distance_speed(self, radar_points):
        distance, speed, object_points, alt_lower, alt_upper, azi_left, azi_right = self.computer_vision.predict_distance(radar_points)

        current_rot = radar_points.transform.rotation
        # Colors for radar points.
        color_object = carla.Color(255, 0, 0)
        color_no_object = carla.Color(255, 255, 255)
        # Draw the points on the screen.
        for point in radar_points:
            azi = point.azimuth
            alt = point.altitude
            depth = point.depth
            delta_v = point.velocity

            azi_deg = math.degrees(azi)
            alt_deg = math.degrees(alt)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt_deg,
                    yaw=current_rot.yaw + azi_deg,
                    roll=current_rot.roll)).transform(fw_vec)

            # give color to radar data: white = neutral; red= move closer; blue=moving away.
            # def clamp(min_v, max_v, value):
            #     return max(min_v, min(value, max_v))
            #
            # velocity_range = 7.5  # m/s
            # norm_velocity = delta_v / velocity_range  # range [-1, 1]
            # r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            # g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            # b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)

            # As it does not work to check whether point is in object_points, we manually check the coordinates.
            if alt_lower < alt < alt_upper and azi_left < azi < azi_right:
                color = color_object
            else:
                color = color_no_object
            # display radar data on screen.
            if self.display_man.render_enabled():
                self.world.debug.draw_point(
                    radar_points.transform.location + fw_vec,
                    size=0.075,
                    life_time=0.1,
                    persistent_lines=False,
                    color=color
                )

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()


def run_simulation(args, client):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """

    display_manager = None
    vehicle = None
    vehicle_list = []
    timer = CustomTimer()

    try:
        # Getting the world
        world = client.get_world()
        # get original_settings to reset the world after use.
        original_settings = world.get_settings()

        if args.sync:
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

        # Instantiating the vehicle to which we attached the sensors
        transform = random.choice(world.get_map().get_spawn_points())
        ego_bp = random.choice(world.get_blueprint_library().filter('vehicle'))
        ego_bp.set_attribute('role_name', 'ego')
        vehicle = world.spawn_actor(ego_bp, transform)
        vehicle_list.append(vehicle)
        print('created %s' % vehicle.type_id)
        vehicle.set_autopilot(True)

        # Display Manager organize all the sensors and its display in a window
        # It can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[1, 1], window_size=[args.width, args.height])

        # Create the ComputerVision object
        computer_vision = ComputerVision()

        # Then, SensorManager is used to spawn RGBCamera and Radar and assign each of them to a grid position.
        SensorManager(world, display_manager, 'RGBCamera',
                      carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)),
                      vehicle, {'sensor_tick': '0.5'}, display_pos=[0, 0], computer_vision=computer_vision)

        SensorManager(world, display_manager, 'Radar',
                      carla.Transform(carla.Location(x=0, z=2.4)),
                      vehicle,
                      {'horizontal_fov': '30', 'points_per_second': '5000', 'range': '100',
                       'sensor_tick': '0', 'vertical_fov': '30'}, display_pos=[0, 0], computer_vision=computer_vision)

        # But the city now is probably quite empty, let's add a few more vehicles.
        transform.location += carla.Location(x=40, y=-3.2)
        transform.rotation.yaw = -180.0
        number_of_vehicles = 0
        while number_of_vehicles < 100:
            transform.location.x += 2.0

            bp = random.choice(world.get_blueprint_library().filter('vehicle'))

            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = world.try_spawn_actor(bp, transform)
            if npc is not None:
                vehicle_list.append(npc)
                npc.set_autopilot(True)
                number_of_vehicles += 1
                print('created %s' % npc.type_id)

        # We create a csv file to save our radar-data.
        # Let's define the headings of our csv file and save them.
        header = ['Velocity', 'Altitude', 'Azimuth', 'Depth']
        with open('RadarData.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(header)

        # Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        while True:
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

        world.apply_settings(original_settings)


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor tutorial')
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
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        run_simulation(args, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
