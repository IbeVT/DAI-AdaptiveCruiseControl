import math
from datetime import datetime

import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ComputerVision:
    def __init__(self):
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

    def set_fov(self, h_fov_degrees, v_fov_degrees):
        self.camera_h_fov = math.radians(h_fov_degrees)
        self.camera_v_fov = math.radians(v_fov_degrees)

    def get_bounding_box(self, image):
        self.image = image
        results = self.model.predict(source=image, save=True)
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
        print(f"Following vehicle cords: {following_vehicle_cords}")
        return following_vehicle_cords

    def predict_distance(self, radar_points):
        self.radar_points = radar_points
        # If there is a car in front
        if self.following_vehicle_cords:
            [x_lower, y_lower, x_upper, y_upper] = self.following_vehicle_cords
            # Convert coordinates to angles
            alt_upper = -((y_lower - self.camera_y_pixels / 2) / self.camera_y_pixels) * self.camera_v_fov
            alt_lower = -((y_upper - self.camera_y_pixels / 2) / self.camera_y_pixels) * self.camera_v_fov

            azi_left = ((x_lower - self.camera_x_pixels / 2) / self.camera_x_pixels) * self.camera_h_fov
            azi_right = ((x_upper - self.camera_x_pixels / 2) / self.camera_x_pixels) * self.camera_h_fov

            print(f"alt_upper: {alt_upper}, alt_lower: {alt_lower}, azi_left: {azi_left}, azi_right: {azi_right}")

            object_point_depths = []
            object_point_speeds = []
            object_points = []
            for point in radar_points:
                # [delta_v, alt, azi, depth] = point
                azi = point.azimuth
                alt = point.altitude
                depth = point.depth
                delta_v = point.velocity

                if alt_lower < alt < alt_upper and azi_left < azi < azi_right:
                    object_point_depths.append(depth)
                    object_point_speeds.append(delta_v)
                    object_points.append(point)

            print('Number of points:', len(object_point_depths))
            print('Depths:', object_point_depths)
            print('Median depth:', round(np.median(object_point_depths), 1), '  Median speed:',
                  round(np.median(object_point_speeds), 1))
            return np.median(object_point_depths), np.median(object_point_speeds), object_points
        else:
            return self.max_depth, 0, []

