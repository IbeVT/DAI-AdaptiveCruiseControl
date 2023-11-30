import math
from datetime import datetime

import cv2
from ultralytics import YOLO
import numpy as np
import carla


class ComputerVision:
    def __init__(self):
        self.inverse_camera_matrix = None
        self.radar_points = None
        self.model = YOLO('best.pt')
        self.vehicle_classes = ['bus', 'bike', 'car', 'motorcycle', 'vehicle']
        self.camera_h_fov = math.radians(90)
        self.camera_v_fov = math.radians(60)
        self.camera_x_pixels = 720
        self.camera_y_pixels = 1280
        self.n_points = 50
        self.radar_h_fov = 30 * (2 * 3.14159 / 360)
        self.radar_v_fov = 30 * (2 * 3.14159 / 360)
        self.max_depth = 100
        self.max_speed = 120 / 3.6
        self.n_points_median = 11
        self.image = None
        self.radar = None
        self.image_updated = False
        self.radar_updated = False
        self.following_vehicle_cords = None
        self.last_distance = None

    def set_fov(self, h_fov_degrees, v_fov_degrees):
        self.camera_h_fov = math.radians(h_fov_degrees)
        self.camera_v_fov = math.radians(v_fov_degrees)

    def get_bounding_boxes(self, image):
        self.image = image
        results = self.model.predict(source=image, save=False)
        # Check which car is in front, if any
        result = results[0]
        following_vehicle_cords = None
        smallest_distance_to_center = float('inf')
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
                [x_lower, y_lower, x_upper, y_upper] = cords
                x_center = (x_lower + x_upper) / 2
                y_center = (y_lower + y_upper) / 2
                distance_to_center = np.sqrt(
                    (x_center - (self.camera_x_pixels / 2)) ** 2 + (y_center - (self.camera_y_pixels / 2)) ** 2)
                if distance_to_center < smallest_distance_to_center:
                    smallest_distance_to_center = distance_to_center
                    following_vehicle_cords = cords
        self.following_vehicle_cords = following_vehicle_cords
        return following_vehicle_cords, result.boxes

    def get_current_bounding_box(self):
        return self.following_vehicle_cords

    def predict_distance(self, radar_points):
        self.radar_points = radar_points
        # If there is a car in front
        object_points = []
        if self.following_vehicle_cords:
            [x_lower, y_lower, x_upper, y_upper] = self.following_vehicle_cords

            object_point_depths = []
            object_point_speeds = []
            for point in radar_points:
                [x, y] = self.get_image_point(point)
                if x_lower < x < x_upper and y_lower < y < y_upper:
                    object_point_depths.append(point.depth)
                    object_point_speeds.append(point.velocity)
                    object_points.append(point)
            self.last_distance = np.median(object_point_depths)
            self.last_speed = np.median(object_point_speeds)
        else:
            self.last_distance = self.max_depth
            self.last_speed = 0
        return self.last_distance, self.last_speed, object_points

    def get_last_distance(self):
        return self.last_distance

    def get_last_speed(self):
        return self.last_speed

    def build_projection_matrix(self, w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        self.projection_matrix = K
        return K

    def get_image_point(self, point):
        # Based on https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
        # Convert RadarDetection to carla 3D location
        azi_deg = math.degrees(point.azimuth)
        alt_deg = math.degrees(point.altitude)
        loc = carla.Vector3D(x=point.depth)
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