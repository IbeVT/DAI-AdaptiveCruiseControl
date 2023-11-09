from ultralytics import YOLO
import numpy as np


class ComputerVision:
    def __int__(self):
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
        print(f"model: {self.model}")

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
            return np.median(object_point_depths), np.median(object_point_speeds)
        else:
            return self.max_depth, 0
