from datetime import datetime

import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image


class ComputerVision:
    def __init__(self):
        self.model = YOLO('best.pt')
        self.vehicle_classes = ['bus', 'bike', 'car', 'motorcycle', 'vehicle']
        self.camera_h_fov = 90 * (2 * 3.14159 / 360)
        self.camera_x_pixels = 810
        self.camera_y_pixels = 1080
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

    def predict(self, image, radar_points):
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

        # If there is a car in front
        if following_vehicle_cords:
            alt_upper = -((y_lower - self.camera_y_pixels / 2) / self.camera_x_pixels) * self.camera_h_fov
            alt_lower = -((y_upper - self.camera_y_pixels / 2) / self.camera_x_pixels) * self.camera_h_fov

            azi_left = ((x_lower - self.camera_x_pixels / 2) / self.camera_x_pixels) * self.camera_h_fov
            azi_right = ((x_upper - self.camera_x_pixels / 2) / self.camera_x_pixels) * self.camera_h_fov

            object_point_depths = []
            object_point_speeds = []
            for point in radar_points:
                # [alt, azi, depth, delta_v] = point
                [delta_v, alt, azi, depth] = point
                if alt_lower < alt < alt_upper and azi_left < azi < azi_right:
                    object_point_depths.append(depth)
                    object_point_speeds.append(delta_v)

            print('Median depth:', round(np.median(object_point_depths), 1), '  Median speed:',
                  round(np.median(object_point_speeds), 1))

            # Add bounding box to image
            print(image.shape)
            print(type(image))
            print("\n\n\n\n\n\n")
            cv2.rectangle(image, (x_lower, y_lower), (x_upper, y_upper), (0, 255, 0), 2)
            # Add statistics to image
            text = f'Depth: ({np.median(object_point_depths)}, speed: {np.median(object_point_speeds)})'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (255, 255, 255)
            font_thickness = 1
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = x_center - text_size[0] // 2
            text_y = y_center + text_size[1] // 2
            cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
            # Save image
            now = datetime.now()
            cv2.imwrite(f"Predictions/{now}.jpg", image)

            return np.median(object_point_depths), np.median(object_point_speeds)
        else:
            return self.max_depth, 0

    def update_image(self, image):
        self.image = image
        if self.radar_updated:
            self.predict(self.image, self.radar)
            self.image_updated = False
            self.radar_updated = False
        else:
            self.image_updated = True

    def update_radar(self, radar):
        self.radar = radar
        self.radar_updated = True
        if self.image_updated:
            self.predict(self.image, self.radar)
            self.image_updated = False
            self.radar_updated = False
        else:
            self.radar_updated = True
