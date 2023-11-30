import math
from datetime import datetime

import cv2
from ultralytics import YOLO
import numpy as np
import carla


class ComputerVision:
    def __init__(self, vehicle):
        self.delta_v = None
        self.result_boxes = []
        self.image = None
        self.object_points = None
        self.vehicle = vehicle
        self.inverse_camera_matrix = None
        self.radar_points = None
        self.model = YOLO('best.pt')
        self.vehicle_classes = ['bus', 'bike', 'car', 'motorcycle', 'vehicle']
        self.camera_x_pixels = 720
        self.camera_y_pixels = 1280
        self.n_points = 50
        self.max_depth = 100
        self.max_speed = 120 / 3.6
        self.max_distance_following_vehicle = 500
        self.following_vehicle_cords = None
        self.distance = None
        self.steer_vector_endpoint = None

    def set_resolution(self, x, y):
        self.camera_x_pixels = x
        self.camera_y_pixels = y

    def process_data(self):
        # Start by detecting objects in the image
        if self.image is None or self.radar_points is None:
            return
        results = self.model.predict(source=self.image, save=False)
        result = results[0]

        # Check which car is in front, if any
        # To find the vehicle in front, we will use the steer angle and the azimuth angle We do it as follows:
        # 1. Get all boxes that are vehicles
        # 2. Group all points that belong to the same box
        # 3. For each box, calculate the median distance and speed
        # 4. Calculate where the vector with length=median_distance, azimuth=steer_angle and
        # altitude=mean_altitude_points would end on the camera
        # 5. Calculate the distance from the vector to the center of the bounding box of the vehicle
        # Reset the values
        self.following_vehicle_cords = None
        self.distance = self.max_depth
        self.delta_v = self.max_speed

        smallest_distance_to_box = self.max_distance_following_vehicle      # The distance from the vector to the
        # center of the bounding box of the vehicle
        wheel_locations = carla.VehicleWheelLocation()
        steer_angle = self.vehicle.get_wheel_steer_angle(wheel_locations.FL_Wheel)
        vehicle_boxes = []
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            print("Object type:", class_id)
            print("Coordinates:", cords)
            print("Probability:", conf)
            print("---")
            if str(class_id) in self.vehicle_classes:
                vehicle_boxes.append(cords)

        # 2. Group all points that belong to the same box
        distances = []  # The distances of all points that belong to the i-th box
        velocities = []  # The speeds of all points that belong to the i-th box
        altitudes = []  # The altitudes of all points that belong to the i-th box
        for point in self.radar_points:
            [x, y] = self.get_image_coordinates_from_radar_point(point.azimuth, point.altitude, point.depth)
            for i, box in enumerate(vehicle_boxes):
                # If there is not yet a list for the i-th box, create it
                if len(distances) <= i:
                    distances.append([])
                    velocities.append([])
                    altitudes.append([])
                [x_lower, y_lower, x_upper, y_upper] = box
                if x_lower < x < x_upper and y_lower < y < y_upper:
                    distances[i].append(point.depth)
                    velocities[i].append(point.velocity)
                    altitudes[i].append(point.altitude)
                    break
        # 3. For each box, calculate the median distance and speed
        distances = [np.median(i) for i in distances]
        velocities = [np.median(i) for i in velocities]
        altitudes = [np.mean(i) for i in altitudes]
        # Now we have the median distance and speed for each box

        # 4. Calculate where the vector with length=median_distance and angle=steer_angle would end on the camera
        # 5. Calculate the distance from the vector to the center of the bounding box of the vehicle
        for i, box in enumerate(vehicle_boxes):
            cords = self.get_image_coordinates_from_radar_point(steer_angle, altitudes[i], distances[i])
            distance_to_box = math.sqrt((cords[0] - np.mean([box[0], box[2]])) ** 2 +
                                        (cords[1] - np.mean([box[1], box[3]])) ** 2)
            print("Cords:", cords)
            print("Box:", box)
            print("Distance to box:", distance_to_box)
            if distance_to_box < smallest_distance_to_box:
                self.following_vehicle_cords = box
                self.distance = distances[i]
                self.delta_v = velocities[i]
                self.steer_vector_endpoint = cords
        self.result_boxes = result.boxes

    def get_boxes(self):
        return self.result_boxes

    def get_current_bounding_box(self):
        return self.following_vehicle_cords

    def get_distance(self):
        return self.distance

    def get_delta_v(self):
        return self.delta_v

    def get_object_points(self):
        return self.object_points

    def build_projection_matrix(self, w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        self.projection_matrix = K
        return K

    def get_image_coordinates_from_radar_point(self, azimuth, altitude, depth):  # in radians
        # Based on https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
        # Convert RadarDetection to carla 3D location
        azi_deg = math.degrees(azimuth)
        alt_deg = math.degrees(altitude)
        loc = carla.Vector3D(x=depth)
        carla.Transform(
            carla.Location(x=0, y=0, z=0),
            carla.Rotation(
                pitch=alt_deg,
                yaw=azi_deg,
                roll=0)).transform(loc)

        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(self.inverse_camera_matrix, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth component also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(self.projection_matrix, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

    def set_inverse_camera_matrix(self, inverse_matrix):
        self.inverse_camera_matrix = inverse_matrix
