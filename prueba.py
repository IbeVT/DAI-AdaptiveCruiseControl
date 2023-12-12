import datetime
import glob
import os
import sys
import random
import time
import numpy as np
import cv2

from ComputerVision_signs import ComputerVision

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def process_img(image):
    i = np.array(image.raw_data)
    i2.reshape((640,480,4))
    i3 = i2[:,:,:3]
    cv2.imshow("",i3)
    cv2,waitkey(1)
    return(i3/255.0)


actor_list = []

try:
    client = carla.Client('localhost',2000)
    client.set_timeout(2.0)
    
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.filter('model3')[0]
    
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn_point)
    
    #vehicle.set_autopilot(True)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)
    
    camera_bp = blueprint_library().find('sensor.camera.rgb')
    
    camera_bp.set_attribute('image_size_x', "640")
    camera_bp.set_attribute('image_size_y', "480")
    camera_bp.set_attribute('fov', "110")
    camera_bp.set_attribute('sensor_tick', '1.0')
    
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    
    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)
    
    sensor.listen(Lambda data: process_img(data))
    
    
    #blueprint = 
    
    
    
finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")


import argparse
import time
import math

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


world = client.load_world('Town06')
weather = carla.WeatherParameters(
    cloudiness=0.0,
    precipitation=0.0,
    sun_altitude_angle=10.0,
    sun_azimuth_angle = 70.0,
    precipitation_deposits = 0.0,
    wind_intensity = 0.0,
    fog_density = 0.0,
    wetness = 0.0, 
)
world.set_weather(weather)

bp_lib = world.get_blueprint_library() 
spawn_points = world.get_map().get_spawn_points()

vehicle_bp = bp_lib.find('vehicle.audi.etron')
ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[79])

spectator = world.get_spectator()
transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),ego_vehicle.get_transform().rotation)
spectator.set_transform(transform)

for i in range(200):  
    vehicle_bp = random.choice(bp_lib.filter('vehicle')) 
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))


for v in world.get_actors().filter('*vehicle*'): 
    v.set_autopilot(True) 
ego_vehicle.set_autopilot(True) 


pygame.init()


clock = pygame.time.Clock()
done = False

while not done:
    world.tick()

    # Update the display and check for the quit event
    pygame.display.flip()
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Sleep to ensure consistent loop timing
    clock.tick(60)