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
elif testCase == 5:
    client.load_world('Town04')
elif testCase == 6:
    client.load_world('Town07')

world = client.get_world()
# get original_settings to reset the world after use.
original_settings = world.get_settings()

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True  # Enables synchronous mode
settings.fixed_delta_seconds = 0.01
world.apply_settings(settings)

# Set up the TM in synchronous mode
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# Set a seed so behaviour can be repeated if necessary
traffic_manager.set_random_device_seed(1)
random.seed(1)

spawn_points = world.get_map().get_spawn_points()
blueprint = world.get_blueprint_library()

ego_bp = blueprint.find('vehicle.mercedes.coupe_2020')
vehicle_bp = random.choice(blueprint.filter('vehicle.*.*'))

# We will also set up the spectator, so we can see what we do
spectator = world.get_spectator()

directions = ['Left', 'Right', 'Straight']
vehicle_list = []


def update_traffic_manager(traffic_manager, actor, speed_diff, route, lane_change=False, change_rate=0):
    # Set parameters of TM vehicle control, we don't want lane changes
    traffic_manager.random_left_lanechange_percentage(actor, change_rate)
    traffic_manager.random_right_lanechange_percentage(actor, change_rate)
    traffic_manager.update_vehicle_lights(actor, True)
    traffic_manager.auto_lane_change(actor, lane_change)
    # traffic_manager.ignore_lights_percentage(vehicle1, 50)
    traffic_manager.vehicle_percentage_speed_difference(actor, speed_diff)

    traffic_manager.set_route(actor, route)


if testCase == 1:
    print('Initiating test case 1.')
    spawn_point_1 = spawn_points[19]
    # Create route:
    route = []
    for i in range(100):
        route.append('Straight')  # 'Left', 'Right', 'Straight'

    # spawn first car
    vehicle = world.spawn_actor(vehicle_bp, spawn_point_1)
    print('created vehicle: %s' % vehicle.type_id)
    vehicle.set_autopilot(True)
    vehicle_list.append(vehicle)
    update_traffic_manager(traffic_manager, actor=vehicle, speed_diff=20, route=route)

    ego_vehicle = None
    while ego_vehicle is None:
        # spawn ego vehicle
        ego_vehicle = world.try_spawn_actor(ego_bp, spawn_point_1)
        world.tick()
    print('created ego: %s' % ego_vehicle.type_id)
    vehicle_list.append(ego_vehicle)
    ego_vehicle.set_autopilot(True)
    update_traffic_manager(traffic_manager, actor=ego_vehicle, speed_diff=0, route=route)

if testCase == 2:
    print('Initiating test case 2.')
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

    while len(vehicle_list) < max_vehicles:
        world.tick()

        # spawn vehicles only after a delay
        if counter == spawn_delay:
            vehicle_bp = random.choice(blueprint.filter('vehicle.*.*'))
            # Alternate spawn points
            if alt:
                vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_1)
            else:
                vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_2)

            if vehicle:  # IF vehicle is successfully spawned
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
    vehicle_list.append(ego_vehicle)
    ego_vehicle.set_autopilot(True)
    update_traffic_manager(traffic_manager, actor=ego_vehicle, speed_diff=0, route=route)

if testCase == 3:
    print('Initiating test case 3.')
    spawn_point_1 = spawn_points[19]
    spawn_point_2 = spawn_points[25]
    spawn_point_ego = spawn_points[60]
    # Create route:
    route = ['Straight']

    max_vehicles = 100
    spawn_delay = 20
    counter = spawn_delay
    alt = False

    while len(vehicle_list) < max_vehicles:
        world.tick()

        # spawn vehicles only after a delay
        if counter == spawn_delay:
            vehicle_bp = random.choice(blueprint.filter('vehicle.*.*'))
            # Alternate spawn points
            if alt:
                vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_1)
            else:
                vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_2)

            if vehicle:  # IF vehicle is successfully spawned
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
        ego_vehicle = world.try_spawn_actor(ego_bp, spawn_point_ego)
        world.tick()
    print('created ego: %s' % ego_vehicle.type_id)
    vehicle_list.append(ego_vehicle)
    ego_vehicle.set_autopilot(True)
    update_traffic_manager(traffic_manager, actor=ego_vehicle, speed_diff=0, route=route)

if testCase == 4:
    print('Initiating test case 4.')
    spawn_point = [spawn_points[240], spawn_points[146], spawn_points[163], spawn_points[8]]
    spawn_point_ego = spawn_points[8]

    # Create route:
    route = []

    max_vehicles = 40
    spawn_delay = 20
    counter = spawn_delay

    while len(vehicle_list) < max_vehicles / 2:
        world.tick()

        # spawn vehicles only after a delay
        if counter == spawn_delay:
            vehicle_bp = random.choice(blueprint.filter('vehicle.*.*'))
            # Alternate spawn points
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point[len(vehicle_list) % 4])

            if vehicle:  # IF vehicle is successfully spawned
                print('created vehicle: %s' % vehicle.type_id)
                vehicle_list.append(vehicle)
                print(len(vehicle_list))
                vehicle.set_autopilot(True)  # Give TM control over vehicle
                update_traffic_manager(traffic_manager, actor=vehicle, speed_diff=0, route=route)

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
    vehicle_list.append(ego_vehicle)
    ego_vehicle.set_autopilot(True)
    update_traffic_manager(traffic_manager, actor=ego_vehicle, speed_diff=0, route=route)

    while len(vehicle_list) < max_vehicles:
        world.tick()

        # spawn vehicles only after a delay
        if counter == spawn_delay:
            vehicle_bp = random.choice(blueprint.filter('vehicle.*.*'))
            # Alternate spawn points
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point[len(vehicle_list) % 4])

            if vehicle:  # IF vehicle is successfully spawned
                print('created vehicle: %s' % vehicle.type_id)
                vehicle_list.append(vehicle)
                print(len(vehicle_list))
                vehicle.set_autopilot(True)  # Give TM control over vehicle
                update_traffic_manager(traffic_manager, actor=vehicle, speed_diff=0, route=route)

                vehicle = None

            counter -= 1
        elif counter > 0:
            counter -= 1
        elif counter == 0:
            counter = spawn_delay

if testCase == 5:
    print('Initiating test case 5.')
    spawn_point = [spawn_points[333], spawn_points[120], spawn_points[55], spawn_points[94]]
    spawn_point_ego = spawn_points[98]

    # Create route:
    route = []
    for i in range(100):
        route.append('Straight')

    max_vehicles = 100
    spawn_delay = 20
    counter = spawn_delay

    while len(vehicle_list) < max_vehicles:
        world.tick()

        # spawn vehicles only after a delay
        if counter == spawn_delay:
            vehicle_bp = random.choice(blueprint.filter('vehicle.*.*'))
            # Alternate spawn points
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point[len(vehicle_list) % 4])

            if vehicle:  # IF vehicle is successfully spawned
                print('created vehicle: %s' % vehicle.type_id)
                vehicle_list.append(vehicle)
                print(len(vehicle_list))
                vehicle.set_autopilot(True)  # Give TM control over vehicle
                update_traffic_manager(traffic_manager, actor=vehicle, speed_diff=0, route=route, lane_change=True,
                                       change_rate=10)

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
    vehicle_list.append(ego_vehicle)
    ego_vehicle.set_autopilot(True)
    update_traffic_manager(traffic_manager, actor=ego_vehicle, speed_diff=0, route=route, lane_change=True,
                           change_rate=2)

if testCase == 6:
    print('Initiating test case 6.')
    spawn_point_1 = spawn_points[36]
    spawn_point_2 = spawn_points[53]
    spawn_point_ego = spawn_points[36]
    # Create route1:
    route = []
    for i in range(100):
        route.append('Straight')

    max_vehicles = 50
    spawn_delay = 20
    counter = spawn_delay
    alt = False

    while len(vehicle_list) < max_vehicles:
        world.tick()

        # spawn vehicles only after a delay
        if counter == spawn_delay:
            vehicle_bp = random.choice(blueprint.filter('vehicle.*.*'))
            # Alternate spawn points
            if alt:
                vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_1)
            else:
                vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_2)

            if vehicle:  # IF vehicle is successfully spawned
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
        ego_vehicle = world.try_spawn_actor(ego_bp, spawn_point_ego)
        world.tick()
    print('created ego: %s' % ego_vehicle.type_id)
    vehicle_list.append(ego_vehicle)
    ego_vehicle.set_autopilot(True)
    update_traffic_manager(traffic_manager, actor=ego_vehicle, speed_diff=0, route=route)

try:
    while True:
        # Update the spectator's position to follow the ego vehicle
        transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)),
                                    ego_vehicle.get_transform().rotation)
        spectator.set_transform(transform)

        # Carla Tick
        world.tick()

finally:
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

    if traffic_manager:
        traffic_manager.set_synchronous_mode(False)

    world.apply_settings(original_settings)
