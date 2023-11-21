"""
This script will spawn some npc's (vehicles and pedestrians) in a random town and random weather conditions.
view controlled by spectator.
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


def game_loop(args, client):
    world = None
    original_settings = None

    actor_list = []
    vehicle_list = []

    # set a random map to spawn our actors in.
    # client.load_world(None)
    maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10', 'Town11', 'Town12']
    new_map = maps[random.randint(0, 10)]
    print('Map: %s' % new_map)
    client.load_world(new_map)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        # Getting the world
        world = client.get_world()
        # get original_settings to reset the world after use.
        original_settings = world.get_settings()

        if args.sync:
            traffic_manager = client.get_trafficmanager(args.tm_port)
            traffic_manager.set_global_distance_to_leading_vehicle(2.5)
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

        # set a random weather condition.
        weather = random.choice(find_weather_presets())
        print('Weather: %s' % weather[1])
        world.set_weather(weather[0])

        # Instantiating the vehicle to which we attached the sensors
        transform = random.choice(world.get_map().get_spawn_points())
        ego_bp = random.choice(world.get_blueprint_library().filter('vehicle'))
        ego_bp.set_attribute('role_name', 'ego')
        ego_vehicle = world.spawn_actor(ego_bp, transform)
        vehicle_list.append(ego_vehicle)
        print('created %s' % ego_vehicle.type_id)
        ego_vehicle.set_autopilot(True)

        # set the spectator to follow our ego_vehicle.
        spectator = world.get_spectator()
        transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)),
                                    ego_vehicle.get_transform().rotation)
        spectator.set_transform(transform)


        # But the city now is probably quite empty, let's add a few more vehicles.
        transform.location += carla.Location(x=40, y=-3.2)
        transform.rotation.yaw = -180.0
        number_of_vehicles = 0
        while number_of_vehicles < args.number_of_vehicles:
            transform.location.x += 8.0

            bp = random.choice(world.get_blueprint_library().filter('vehicle'))

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
            # Carla Tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            if call_exit:
                break

    finally:
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

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
