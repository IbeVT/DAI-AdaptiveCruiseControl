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
    import pygame.freetype
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
        pygame.display.set_caption("Sensor Data")
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
    def __init__(self, world, display_man, transform, attached, sensor_options, display_pos,
                 computer_vision):
        self.computer_vision = computer_vision
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.tics_processing = 0
        self.display_man.add_sensor(self)
        self.surface = None

    def init_sensor(self, transform, attached, sensor_options):
        pass

    def get_sensor(self):
        return self.sensor

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()


class CameraManager(SensorManager):
    def __init__(self, world, display_man, transform, attached, sensor_options, display_pos,
                 computer_vision):
        SensorManager.__init__(self, world, display_man, transform, attached, sensor_options, display_pos,
                               computer_vision)
        self.camera_array = None
        self.bb_cords = None

    def init_sensor(self, transform, attached, sensor_options):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        disp_size = self.display_man.get_display_size()
        camera_bp.set_attribute('image_size_x', str(disp_size[0]))
        camera_bp.set_attribute('image_size_y', str(disp_size[1]))
        camera_bp.set_attribute('sensor_tick', '1.0')

        for key in sensor_options:
            camera_bp.set_attribute(key, sensor_options[key])

        camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)

        # Build camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        self.computer_vision.set_inverse_camera_matrix(world_2_camera)
        # Get the attributes from the camera
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        self.computer_vision.build_projection_matrix(image_w, image_h, fov)
        camera.listen(self.process_camera)
        return camera

    def process_camera(self, image):
        print("Processing camera data")
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_array = array
        # Get bounding box from image array.
        self.bb_cords = self.computer_vision.get_bounding_box(array)
        self.tics_processing += 1
        print("Finished processing camera data")

    def draw_camera(self):
        # Create surface from image array.
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(self.camera_array.swapaxes(0, 1))
        # Draw bounding box on screen.
        if self.bb_cords is not None:
            [x_lower, y_lower, x_upper, y_upper] = self.bb_cords
            if self.display_man.render_enabled():
                pygame.draw.rect(self.surface, (255, 0, 0), (x_lower, y_lower, x_upper - x_lower, y_upper - y_lower), 2)

            # Display the last distance and speed on screen.
            font = pygame.freetype.SysFont('Arial', 30)
            if self.display_man.render_enabled() and self.surface is not None:
                # Display distance and speed on screen.
                text, rect = font.render(f'Distance: {self.computer_vision.get_last_distance()}', (255, 0, 0))
                x = x_lower
                y = y_lower - rect.height - 5
                self.surface.blit(text, (x, y))
                text, rect = font.render(f'Speed: {self.computer_vision.get_last_speed()}', (255, 0, 0))
                y -= rect.height - 10


class RadarManager(SensorManager):
    def __init__(self, world, display_man, transform, attached, sensor_options, display_pos,
                 computer_vision):
        SensorManager.__init__(self, world, display_man, transform, attached, sensor_options, display_pos,
                               computer_vision)
        self.object_points = None
        self.speed = None
        self.distance = None
        self.radar_points = None

    def init_sensor(self, transform, attached, sensor_options):
        radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
        for key in sensor_options:
            radar_bp.set_attribute(key, sensor_options[key])
        radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
        radar.listen(self.process_radar)
        return radar

    def process_radar(self, radar_points):
        print("Processing radar data")
        self.radar_points = radar_points
        self.distance, self.speed, self.object_points = self.computer_vision.predict_distance(radar_points)
        print("Finished processing radar data")

    # Draws the radar points on the screen. It should be executed AFTER the camera has been drawn.
    def draw_radar(self):
        current_rot = self.radar_points.transform.rotation
        # Colors for radar points.
        color_object = carla.Color(255, 0, 0)
        color_no_object = carla.Color(255, 255, 255)
        # Draw the points on the screen.
        for point in self.radar_points:
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

            # As it does not work to check whether point is in object_points, we manually check the coordinates.
            color = color_no_object
            for object_point in self.object_points:
                if object_point.altitude == point.altitude and object_point.azimuth == point.azimuth:
                    color = color_object
                    break

            # display radar data on screen.
            if self.display_man.render_enabled():
                self.world.debug.draw_point(
                    self.radar_points.transform.location + fw_vec,
                    size=0.075,
                    life_time=0.1,
                    persistent_lines=False,
                    color=color
                )


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
        camera_manager = CameraManager(world, display_manager,
                                       carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)),
                                       vehicle, {'sensor_tick': '0.1'}, display_pos=[0, 0],
                                       computer_vision=computer_vision)
        radar_manager = RadarManager(world, display_manager,
                                     carla.Transform(carla.Location(x=0, z=2.4)),
                                     vehicle, {'horizontal_fov': '40', 'points_per_second': '5000', 'range': '100',
                                               'sensor_tick': '0.1', 'vertical_fov': '40'}, display_pos=[0, 0],
                                     computer_vision=computer_vision)

        # But the city now is probably quite empty, let's add a few more vehicles.
        transform.location += carla.Location(x=4, y=-3.2)
        transform.rotation.yaw = -180.0
        number_of_vehicles = 0
        while number_of_vehicles < 200:
            transform.location.x += 4.0

            bp = random.choice(world.get_blueprint_library().filter('vehicle'))

            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = world.try_spawn_actor(bp, transform)
            if npc is not None:
                vehicle_list.append(npc)
                npc.set_autopilot(True)
                number_of_vehicles += 1
                print('created %s' % npc.type_id)

        # Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        while True:
            # Carla Tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # First draw camera on the screen, then the radar
            if camera_manager.camera_array is not None:
                camera_manager.draw_camera()
            if radar_manager.radar_points is not None:
                radar_manager.draw_radar()

            # Render received data
            display_manager.render()

            print("Rendered")

            camera_manager.finished = False
            radar_manager.finished = False

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
