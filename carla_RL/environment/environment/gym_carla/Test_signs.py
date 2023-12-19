import datetime
import glob
import os
import sys

from ComputerVision_signs import ComputerVision

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
        pygame.display.set_caption("Traffic_signs")
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

    def init_sensor(self, transform, attached, sensor_options):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        disp_size = self.display_man.get_display_size()
        x_res = disp_size[0]
        y_res = disp_size[1]
        camera_bp.set_attribute('image_size_x', str(x_res))
        camera_bp.set_attribute('image_size_y', str(y_res))
        camera_bp.set_attribute('sensor_tick', '1.0')

        self.computer_vision.set_resolution(x_res, y_res)

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
        camera.listen(self.update_camera)
        return camera

    def update_camera(self, image):
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_array = array
        self.computer_vision.image = array

    def draw_camera(self):
        following_bb = self.computer_vision.get_current_bounding_box()
        # Create surface from image array.
        if self.display_man.render_enabled() and self.camera_array is not None:
            self.surface = pygame.surfarray.make_surface(self.camera_array.swapaxes(0, 1))
            # Draw bounding boxes on screen.
            # Draw the bounding box of the followed vehicle.
            if (following_bb is not None) and (self.surface is not None):
                [x_lower, y_lower, x_upper, y_upper] = following_bb["cords"]
                pygame.draw.rect(self.surface, (0, 255, 0), (x_lower, y_lower, x_upper - x_lower, y_upper - y_lower), 2)

                # Display the last distance and speed on screen.
                font = pygame.freetype.SysFont('Arial', 30)
                # Display distance and speed on screen.
                distance = self.computer_vision.get_distance()
                x = x_lower
                y = y_lower - 5
                # Catch errors, mostly when a NaN occurs.
                try:
                    distance = round(distance)
                    text, rect = font.render(f'Distance: {distance}', (0, 255, 0))
                    y -= rect.height
                    self.surface.blit(text, (x, y))
                except:
                    pass
                speed = self.computer_vision.get_delta_v()
                try:
                    speed = round(speed)
                    text, rect = font.render(f'Speed: {speed}', (0, 255, 0))
                    y -= rect.height - 10
                    self.surface.blit(text, (x, y))
                except:
                    pass

                # Display the vehicle category on screen.
                text, rect = font.render(following_bb["class_id"], (0, 255, 0))
                y -= rect.height
                self.surface.blit(text, (x, y))

            # Draw all other bounding boxes on screen.
            boxes = self.computer_vision.get_boxes()
            if boxes is not None:
                for box in boxes:
                    # print('box1',box)

                    if "cords" in box:
                        cords = box["cords"]
                    if "cords2" in box:
                        cords = box["cords2"]

                    if following_bb is not None and cords != following_bb["cords"]:
                        [x_lower, y_lower, x_upper, y_upper] = cords
                        pygame.draw.rect(self.surface, (0, 0, 255),
                                         (x_lower, y_lower, x_upper - x_lower, y_upper - y_lower), 2)

            # Draw low confidence bounding boxes on screen.
            low_confidence_boxes = self.computer_vision.get_low_conf_boxes()
            if low_confidence_boxes is not None:
                for box in low_confidence_boxes:
                    # print('box2',box)

                    if "cords" in box:
                        cords = box["cords"]
                    if "cords2" in box:
                        cords = box["cords2"]

                    if cords != following_bb:
                        [x_lower, y_lower, x_upper, y_upper] = cords
                        pygame.draw.rect(self.surface, (255, 0, 0),
                                         (x_lower, y_lower, x_upper - x_lower, y_upper - y_lower), 2)

            # For debug purposes: draw the steer vector endpoint
            if self.computer_vision.steer_vector_endpoint is not None:
                pygame.draw.circle(self.surface, (0, 0, 255), self.computer_vision.steer_vector_endpoint, 15)


class RadarManager(SensorManager):
    def __init__(self, world, display_man, transform, attached, sensor_options, display_pos,
                 computer_vision):
        SensorManager.__init__(self, world, display_man, transform, attached, sensor_options, display_pos,
                               computer_vision)

    def init_sensor(self, transform, attached, sensor_options):
        radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
        for key in sensor_options:
            radar_bp.set_attribute(key, sensor_options[key])
        radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
        radar.listen(self.update_radar)
        return radar

    def update_radar(self, radar_points):
        self.computer_vision.radar_points = radar_points

    # Draws the radar points on the screen. It should be executed AFTER the camera has been drawn.
    def draw_radar(self):
        radar_points = self.computer_vision.radar_points
        if radar_points is not None:
            object_points = self.computer_vision.get_object_points()
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

                # As it does not work to check whether point is in object_points, we manually check the coordinates.
                color = color_no_object
                if object_points is not None:
                    for object_point in object_points:
                        if object_point.altitude == point.altitude and object_point.azimuth == point.azimuth:
                            color = color_object
                            break

                # display radar data on screen.
                if self.display_man.render_enabled():
                    self.world.debug.draw_point(
                        radar_points.transform.location + fw_vec,
                        size=0.075,
                        life_time=0.1,
                        persistent_lines=False,
                        color=color
                    )


def run_simulation(args, client):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """
    testCase = args.testcase
    # set town and aerial spectator based on the test case
    if testCase == 1 or testCase == 2 or testCase == 3:
        town = client.load_world('Town02')
        # specLocation = carla.Location(x=90, y=180, z=200)
    elif testCase == 4:
        client.load_world('Town03')
        # specLocation = carla.Location(x=0, y=0, z=500)
    elif testCase == 5:
        client.load_world('Town04')
        # specLocation = carla.Location(x=0, y=0, z=500)
    elif testCase == 6:
        client.load_world('Town7')
        # specLocation = carla.Location(x=150, y=150, z=300)

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

        spawn_points = world.get_map().get_spawn_points()
        blueprint = world.get_blueprint_library()
        ego_bp = blueprint.find('vehicle.mercedes.coupe_2020')
        vehicle_bp = random.choice(blueprint.filter('vehicle.*.*'))

        if testCase == 1:
            spawn_point_1 = spawn_points[19]
            # Create route:
            route = []
            for i in range(100):
                route.append('Straight')  # 'Left', 'Right', 'Straight'

            # spawn first car
            vehicle1 = world.spawn_actor(vehicle_bp, spawn_point_1)
            print('created vehicle: %s' % vehicle1.type_id)
            vehicle1.set_autopilot(True)
            update_traffic_manager(traffic_manager, actor=vehicle1, speed_diff=20, route=route)

            ego_vehicle = None
            while ego_vehicle is None:
                # spawn ego vehicle
                ego_vehicle = world.try_spawn_actor(ego_bp, spawn_point_1)
                world.tick()
            print('created ego: %s' % ego_vehicle.type_id)
            ego_vehicle.set_autopilot(True)
            update_traffic_manager(traffic_manager, actor=ego_vehicle, speed_diff=0, route=route)

        if testCase == 2:
            spawn_point_1 = spawn_points[19]
            spawn_point_2 = spawn_points[25]
            # Create route:
            route = []
            for i in range(100):
                route.append('Straight')  # 'Left', 'Right', 'Straight'

            max_vehicles = 20
            spawn_delay = 20
            counter = spawn_delay
            alt = False
            vehicle_list = []
            n_vehicles = len(vehicle_list)

            while n_vehicles < max_vehicles:
                world.tick()

                n_vehicles = len(vehicle_list)

                # spawn vehicles only after a delay
                if counter == spawn_delay:
                    vehicle_bp = random.choice(blueprint.filter('vehicle.*.*'))
                    # Alternate spawn points
                    if alt:
                        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_1)
                    else:
                        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_2)

                    if vehicle:  # IF vehicle is succesfully spawned
                        print('created vehicle: %s' % vehicle.type_id)
                        vehicle_list.append(vehicle)
                        print(len(vehicle_list))
                        vehicle.set_autopilot(True)  # Give TM control over vehicle
                        update_traffic_manager(traffic_manager, actor=vehicle, speed_diff=0, route=route)

                        alt = not alt
                        vehicle = None

                    counter -= 1
                elif counter > 0:
                    counter -= 1
                elif counter == 0:
                    counter = spawn_delay

            ego_vehicle = None
            while ego_vehicle is None:
                # spawn ego vehicle
                ego_vehicle = world.try_spawn_actor(ego_bp, spawn_point_1)
                world.tick()
            print('created ego: %s' % ego_vehicle.type_id)
            ego_vehicle.set_autopilot(True)
            update_traffic_manager(traffic_manager, actor=ego_vehicle, speed_diff=0, route=route)

        if testCase == 3:
            spawn_point_1 = spawn_points[19]
            spawn_point_2 = spawn_points[25]
            spawn_point_ego = spawn_points[60]
            # Create route1:
            route1 = []
            for i in range(100):
                route1.append('Straight')  # 'Left', 'Right', 'Straight'
            # Create route2: for ego:
            route2 = []
            for i in range(50):
                route2.append('Right')

            max_vehicles = 50
            spawn_delay = 20
            counter = spawn_delay
            alt = False
            vehicle_list = []
            n_vehicles = len(vehicle_list)

            while n_vehicles < max_vehicles:
                world.tick()

                n_vehicles = len(vehicle_list)

                # spawn vehicles only after a delay
                if counter == spawn_delay:
                    vehicle_bp = random.choice(blueprint.filter('vehicle.*.*'))
                    # Alternate spawn points
                    if alt:
                        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_1)
                    else:
                        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_2)

                    if vehicle:  # IF vehicle is succesfully spawned
                        print('created vehicle: %s' % vehicle.type_id)
                        vehicle_list.append(vehicle)
                        print(len(vehicle_list))
                        vehicle.set_autopilot(True)  # Give TM control over vehicle
                        update_traffic_manager(traffic_manager, actor=vehicle, speed_diff=0, route=route1)

                        alt = not alt
                        vehicle = None

                    counter -= 1
                elif counter > 0:
                    counter -= 1
                elif counter == 0:
                    counter = spawn_delay

            ego_vehicle = None
            while ego_vehicle is None:
                # spawn ego vehicle
                ego_vehicle = world.try_spawn_actor(ego_bp, spawn_point_ego)
                world.tick()
            print('created ego: %s' % ego_vehicle.type_id)
            ego_vehicle.set_autopilot(True)
            update_traffic_manager(traffic_manager, actor=ego_vehicle, speed_diff=0, route=route2)

        # Display Manager organize all the sensors and its display in a window
        # It can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[1, 1], window_size=[args.width, args.height])

        camera_h_fov = 40
        camera_v_fov = 40
        radar_sample_rate = 10

        # Create the ComputerVision object
        computer_vision = ComputerVision(ego_vehicle, radar_sample_rate)

        # Then, SensorManager is used to spawn RGBCamera and Radar and assign each of them to a grid position.
        camera_manager = CameraManager(world, display_manager,
                                       carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)),
                                       ego_vehicle, {'sensor_tick': f'{1 / radar_sample_rate}'}, display_pos=[0, 0],
                                       computer_vision=computer_vision)
        radar_manager = RadarManager(world, display_manager,
                                     carla.Transform(carla.Location(x=0, z=2.4)),
                                     ego_vehicle,
                                     {'horizontal_fov': f'{camera_h_fov}', 'points_per_second': '5000', 'range': '100',
                                      'sensor_tick': '0.1', 'vertical_fov': f'{camera_v_fov}'}, display_pos=[0, 0],
                                     computer_vision=computer_vision)

        # def speed():

        # Simulation loop
        call_exit = False
        while True:
            # Carla Tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            computer_vision.process_data()

            # First draw camera on the screen, then the radar
            camera_manager.draw_camera()
            radar_manager.draw_radar()

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


def update_traffic_manager(traffic_manager, actor, speed_diff, route):
    # Set parameters of TM vehicle control, we don't want lane changes
    traffic_manager.update_vehicle_lights(actor, True)
    # traffic_manager.random_left_lanechange_percentage(actor, 0)
    # traffic_manager.random_right_lanechange_percentage(actor, 0)
    traffic_manager.auto_lane_change(actor, False)
    # traffic_manager.ignore_lights_percentage(vehicle1, 50)
    traffic_manager.vehicle_percentage_speed_difference(actor, speed_diff)

    traffic_manager.set_route(actor, route)


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
    argparser.add_argument(
        '-t', '--testcase',
        required=True,
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help='Argument decides test case to run (value from 1 to 5)'
    )
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