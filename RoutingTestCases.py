"""
script to set a route to follow for trying the different test cases.
"""

import carla
import random
import argparse

# add command-line arguments
argparser = argparse.ArgumentParser(
    description='CARLA World generation')
argparser.add_argument(
    '-t', '--testcase',
    required=True,
    type=int,
    default=1,
    choices=[1, 2, 3, 4, 5, 6],
    help='Argument decides test case to run (value from 1 to 5)'
)
args = argparser.parse_args()
testCase = args.testcase

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)

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

world = client.get_world()

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True  # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Set up the TM in synchronous mode
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# Set a seed so behaviour can be repeated if necessary
# traffic_manager.set_random_device_seed(1)
# random.seed(1)

spawn_points = world.get_map().get_spawn_points()
blueprint = world.get_blueprint_library()

ego_bp = blueprint.find('vehicle.mercedes.coupe_2020')
vehicle_bp = random.choice(blueprint.filter('vehicle.*.*'))

# We will also set up the spectator, so we can see what we do
spectator = world.get_spectator()
# transform = carla.Transform(specLocation, carla.Rotation(-90))
# spectator.set_transform(transform)


def update_traffic_manager(traffic_manager, actor, speed_diff, route):
    # Set parameters of TM vehicle control, we don't want lane changes
    traffic_manager.update_vehicle_lights(actor, True)
    traffic_manager.random_left_lanechange_percentage(actor, 0)
    traffic_manager.random_right_lanechange_percentage(actor, 0)
    traffic_manager.auto_lane_change(actor, False)
    # traffic_manager.ignore_lights_percentage(vehicle1, 50)
    traffic_manager.vehicle_percentage_speed_difference(actor, speed_diff)

    traffic_manager.set_route(actor, route)


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


while True:

    # Update the spectator's position to follow the ego vehicle
    transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)),
                                ego_vehicle.get_transform().rotation)
    spectator.set_transform(transform)

    # Carla Tick
    world.tick()
