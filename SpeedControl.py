"""
This script allows the user to control the speed of the ego vehicle,
while the rest of the driving is still handled by the carla-autopilot
"""

import carla
import random
import time


def manual_control(vehicle, throttle, brake):
    control = carla.VehicleControl()

    # Manual throttle and brake
    control.throttle = throttle
    control.brake = brake

    return control


# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

# Load the world and get the ego vehicle
world = client.get_world()
blueprint_library = world.get_blueprint_library()
ego_vehicle_bp = blueprint_library.filter('vehicle.*')[0]
ego_transform = random.choice(world.get_map().get_spawn_points())
# ego_transform = carla.Transform(carla.Location(x=100, y=100, z=2), carla.Rotation())
ego_vehicle = world.spawn_actor(ego_vehicle_bp, ego_transform)

# Attach a spectator to the ego vehicle
spectator = world.get_spectator()
transform = ego_vehicle.get_transform()
spectator.set_transform(carla.Transform(transform.location + carla.Location(z=3)))

# Create a Traffic Manager
traffic_manager = client.get_trafficmanager()

try:
    # Set up the Traffic Manager for the ego vehicle
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.auto_lane_change(ego_vehicle, False)
    traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, 0.0)
    traffic_manager.distance_to_leading_vehicle(ego_vehicle, 1.0)
    ego_vehicle.set_autopilot(True)

    # Main control loop
    while True:
        # Example: Manual control with constant throttle and no brake
        throttle_input = 0.00001  # todo: update these values based on the RL-agent.
        brake_input = 0.0  # todo: update these values based on the RL-agent.

        # Get the ego vehicle control
        control = manual_control(ego_vehicle, throttle_input, brake_input)

        # Apply the control to the ego vehicle
        ego_vehicle.apply_control(control)

        # Update the spectator's position to follow the ego vehicle
        transform = ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                                carla.Rotation(pitch=-90)))
        #spectator.set_transform(ego_vehicle.get_transform())

        # Wait for the next frame
        world.tick()

finally:
    # Clean up
    traffic_manager.set_synchronous_mode(False)
    ego_vehicle.destroy()
