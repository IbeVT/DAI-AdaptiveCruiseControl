import math

import carla
import numpy as np
from ultralytics import YOLO

from gym_carla.LowpassFilter import LowpassFilter


import os

class ComputerVision:
    def __init__(self, vehicle, radar_sample_rate=10):
        self.image = None
        self.object_points = None
        self.vehicle = vehicle
        self.inverse_camera_matrix = None
        self.radar_points = None
        self.model = YOLO('/home/carla/PythonScripts/Stijn/DAI-AdaptiveCruiseControl/carla_RL/environment/environment/gym_carla/best.pt')
        self.vehicle_classes = ['bus', 'bike', 'car', 'motorcycle', 'vehicle']
        self.camera_x_pixels = 720
        self.camera_y_pixels = 1280
        self.n_points = 50
        self.max_depth = 100
        self.max_speed = 120 / 3.6  # 120 km/h
        self.max_distance_following_vehicle = 50  # The maximum distance between the endpoint of the steer vector and the center of the bounding box of the vehicle
        self.following_vehicle_box = None
        self.following_vehicle_type = None
        self.distance = self.max_depth
        self.delta_v = self.max_speed
        self.steer_vector_endpoint = None
        self.wheel_angles = []
        self.radar_sample_rate = radar_sample_rate
        self.low_pass_filter = LowpassFilter(2, radar_sample_rate, 5)
        self.boxes = []
        self.low_conf_boxes = []

    def set_resolution(self, x, y):
        self.camera_x_pixels = x
        self.camera_y_pixels = y

    def process_data(self):
        # Start by detecting objects in the image
        if self.image is None or self.radar_points is None:
            return
        results = self.model.predict(source=self.image, save=False, conf=0.1)
        result = results[0]

        # Check which car is in front, if any To find the vehicle in front, we will use the steer angle and the
        # azimuth angle We do it as follows: 1. Get all boxes that are vehicles 2. Group all points that belong to
        # the same box 3. For each box, calculate the median distance and speed 4. Calculate where the vector with
        # length=median_distance, azimuth=steer_angle and altitude=mean_altitude_points would end on the camera 5.
        # Calculate the distance from the vector to the center of the bounding box of the vehicle Reset the values
        # Afterward, check if the center of the beam does not contain too much close points. The computer vision
        # might have missed an object that is close.
        self.following_vehicle_box = None
        self.following_vehicle_type = None
        self.distance = self.max_depth
        self.delta_v = self.max_speed

        # center of the bounding box of the vehicle
        wheel_locations = carla.VehicleWheelLocation()
        self.wheel_angles.append(self.vehicle.get_wheel_steer_angle(wheel_locations.FL_Wheel))
        n_points = int(self.radar_sample_rate * 2)  # Use the data from the last 2 seconds for the low pass filter
        # If the list is too short for filtering, just use the last value. It only happens for the first two seconds
        # anyway.
        list_len = len(self.wheel_angles)
        if list_len > n_points:
            filtered_wheel_angles = self.low_pass_filter.butter_lowpass_filter(self.wheel_angles[-n_points:])
            # Only use the filtered value if it is smaller than the original value.
            if math.fabs(filtered_wheel_angles[-1]) < math.fabs(self.wheel_angles[-1]):
                steer_angle = filtered_wheel_angles[-1]
            else:
                steer_angle = self.wheel_angles[-1]
        else:
            steer_angle = self.wheel_angles[-1]
        # Periodically reset the list to avoid taking up too much memory
        if len(self.wheel_angles) > 10 * n_points:
            self.wheel_angles = self.wheel_angles[-n_points:]

        vehicle_boxes = []
        previous_results = self.boxes
        previous_low_conf_results = self.low_conf_boxes
        self.boxes = []
        self.low_conf_boxes = []
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            # If the confidence is high enough, immediately save the box
            if conf > 0.5:
                self.boxes.append({"class_id": class_id, "cords": cords, "conf": conf})
                if str(class_id) in self.vehicle_classes:
                    vehicle_boxes.append({"class_id": class_id, "cords": cords, "conf": conf})
                continue
            # Check if a similar box was detected in the previous frame
            found = False
            for previous_box in previous_results:
                class_id_previous = previous_box["class_id"]
                cords_previous = previous_box["cords"]
                # Check if the class is the same
                if class_id == class_id_previous:
                    # Check whether the boxes overlap
                    if do_boxes_overlap(cords, cords_previous):
                        # If approximately the same box was detected in the previous frame, we will suppose that it
                        # is indeed a true positive
                        self.boxes.append({"class_id": class_id, "cords": cords, "conf": conf})
                        if str(class_id) in self.vehicle_classes:
                            vehicle_boxes.append({"class_id": class_id, "cords": cords, "conf": conf})
                        found = True
                        break

            if not found:
                # Check if a similar box was detected with low confidence in the previous frame
                for previous_box in previous_low_conf_results:
                    class_id_previous = previous_box["class_id"]
                    cords_previous = previous_box["cords"]
                    # Check if the class is the same
                    if class_id == class_id_previous:
                        # Check whether the boxes overlap
                        if do_boxes_overlap(cords, cords_previous):
                            # If approximately the same box was detected in the previous frame, we will suppose that it
                            # is indeed a true positive
                            if previous_box["conf"] + conf > 0.6:
                                self.boxes.append({"class_id": class_id, "cords": cords, "conf": conf})
                                if str(class_id) in self.vehicle_classes:
                                    vehicle_boxes.append({"class_id": class_id, "cords": cords, "conf": conf})
                                found = True
                                break
            if not found:
                # Still save the box, for debugging purposes
                self.low_conf_boxes.append({"class_id": class_id, "cords": cords, "conf": conf})

        # 2. Group all points that belong to the same box
        distances = []  # The distances of all points that belong to the i-th box
        velocities = []  # The speeds of all points that belong to the i-th box
        azimuths = []  # The azimuths of all points that belong to the i-th box
        points_in_box = []
        for point in self.radar_points:
            [x, y] = self.get_image_coordinates_from_radar_point(point.azimuth, point.altitude, point.depth)
            for i, box in enumerate(vehicle_boxes):
                # If there is not yet a list for the i-th box, create it
                if len(distances) <= i:
                    distances.append([])
                    velocities.append([])
                    azimuths.append([])
                    points_in_box.append([])
                [x_lower, y_lower, x_upper, y_upper] = box["cords"]
                if x_lower < x < x_upper and y_lower < y < y_upper:
                    distances[i].append(point.depth)
                    velocities[i].append(point.velocity)
                    azimuths[i].append(point.azimuth)
                    points_in_box[i].append(point)
                    break
        # 3. For each box, calculate the global distance, speed and azimuth
        car_speed = self.vehicle.get_velocity().length()
        # Remove the points that have a speed value that is close to the opposite of the speed of the car. These points are probably reflections of the ground.
        for i, array in enumerate(velocities):
            ind_to_remove = []
            for j, value in enumerate(array):
                if abs(value + car_speed) < 1:
                    ind_to_remove.append(j)
            # Only remove if there are enough points left, if not, the car itself is probably not moving
            if vehicle_boxes[i]["class_id"] == "bike" or vehicle_boxes[i]["class_id"] == "motorcycle":
                # Use a different threshold for bieks and motorbikes, as they are smaller, and thus, more points in the bounding box are reflected on the ground
                if len(ind_to_remove) > 7 * len(array) / 8:
                    continue
            else:
                if len(ind_to_remove) > 2 * len(array) / 3:
                    continue

            # Remove the elements in reverse order to avoid changing the indices of the elements that still need to be removed
            for j in sorted(ind_to_remove, reverse=True):
                del velocities[i][j]
                del distances[i][j]
                del azimuths[i][j]
                del points_in_box[i][j]

        # Take the median
        distances = [np.median(i) for i in distances]
        azimuths = [np.median(i) for i in azimuths]
        velocities = [np.median(i) for i in velocities]
        # Now we have the median distance and speed for each box

        # 4. Determine which box to follow
        candidates = []
        candidate_velocities = []
        candidate_distances = []
        candidate_points_in_box = []
        following_azimuth = 10
        for i, box in enumerate(vehicle_boxes):
            # If there are no points in the box, skip it
            if math.isnan(azimuths[i]):
                continue
            angle_diff = steer_angle - azimuths[i]

            if abs(angle_diff) < np.radians(10) or abs(azimuths[i]) < np.radians(20):
                # Calculate the absolute speed of the other car. If it is rapidly approaching, it will drive in th other direction
                box_velocity = velocities[i] + car_speed
                if box_velocity > -5:
                    candidates.append(box)
                    candidate_velocities.append(velocities[i])
                    candidate_distances.append(distances[i])
                    candidate_points_in_box.append(points_in_box[i])
        if len(candidates) == 1:
            self.following_vehicle_box = candidates[0]
            self.following_vehicle_type = candidates[0]["class_id"]
            self.distance = candidate_distances[0]
            self.delta_v = candidate_velocities[0]
            self.object_points = candidate_points_in_box[0]
        elif len(candidates) > 1:
            # This code can be optimized, but the effect is negligible due to the small size of candidates
            for i, box in enumerate(candidates):
                # Check if there is a box that is closer
                for j, box2 in enumerate(candidates):
                    if i == j:
                        continue
                    box_to_follow = None
                    if is_box2_closer(box, box2):
                        if azimuths[j] < following_azimuth:
                            # Choose the car that is the closest to the center of the screen
                            box_to_follow = box2
                            index = j
                    else:
                        if azimuths[i] < following_azimuth:
                            # Choose the car that is the closest to the center of the screen
                            box_to_follow = box
                            index = i
                    if box_to_follow is not None:
                        self.following_vehicle_box = box_to_follow
                        self.following_vehicle_type = box_to_follow["class_id"]
                        self.distance = candidate_distances[index]
                        self.delta_v = candidate_velocities[index]
                        self.object_points = candidate_points_in_box[index]

        # If the car is close to a bus, the computer vision might not be able to detect it. In that case,
        # we still want to use the radar distance. Therefore, we need some sort of detection to check whether there
        # is an object close to the car. We will do this by checking whether there are points in the middle of the
        # radar point cloud that are close to the car.
        radar_angle = np.radians(5)
        radar_center_distance = []
        radar_center_velocities = []
        for point in self.radar_points:
            if abs(point.azimuth) < radar_angle and abs(point.altitude) < radar_angle:
                radar_center_distance.append(point.depth)
                radar_center_velocities.append(point.velocity)
        # Remove the points that have a speed value that is close to the opposite of the speed of the car. These
        # points are probably reflections on the ground.
        ind_to_remove = []
        for i, value in enumerate(radar_center_velocities):
            if abs(value + car_speed) < 1:
                ind_to_remove.append(i)
        # Only remove if there are enough points left, if not, the car itself is probably not moving
        if len(ind_to_remove) < 2 * len(radar_center_velocities) / 3:
            # Remove the elements in reverse order to avoid changing the indices of the elements that still need to be removed
            for i in sorted(ind_to_remove, reverse=True):
                del radar_center_velocities[i]
                del radar_center_distance[i]
        center_distance_mean = np.mean(radar_center_distance)
        if center_distance_mean < 10:
            self.distance = center_distance_mean
            self.delta_v = np.mean(radar_center_velocities)

        # Do a last check to make sure there are no NaN values
        if math.isnan(self.distance):
            self.distance = self.max_depth
        if math.isnan(self.delta_v):
            self.delta_v = self.max_speed

    def get_current_bounding_box(self):
        return self.following_vehicle_box

    def get_distance(self):
        return self.distance

    def get_delta_v(self):
        return self.delta_v

    def get_object_points(self):
        return self.object_points

    def get_boxes(self):
        return self.boxes

    def get_low_conf_boxes(self):
        return self.low_conf_boxes

    def get_red_light(self):
        return False

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

        # New we must change from UE4's coordinate system to a "standard"
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


def do_boxes_overlap(box1, box2):
    x1_lower, y1_lower, x1_upper, y1_upper = box1
    x2_lower, y2_lower, x2_upper, y2_upper = box2

    # Check if one box is to the left of the other
    if x1_upper < x2_lower or x2_upper < x1_lower:
        return False

    # Check if one box is above the other
    if y1_upper < y2_lower or y2_upper < y1_lower:
        return False

    # If the above conditions are not met, the boxes overlap
    return True

def is_box2_closer(box1, box2):
    x1_lower, y1_lower, x1_upper, y1_upper = box1["cords"]
    x2_lower, y2_lower, x2_upper, y2_upper = box2["cords"]

    # Check if one box is to the left of the other
    if x1_upper < x2_lower or x2_upper < x1_lower:
        return False

    # Check if one box is to the right of the other
    if x1_lower > x2_upper or x2_lower > x1_upper:
        return False

    # Check if y values of the second box are smaller
    if y2_upper < y1_upper:
        return False

    # If the above conditions are not met, box2 is closer
    return True

def is_point_in_box(box, point):
    x_lower, y_lower, x_upper, y_upper = box
    x, y = point
    if x_lower < x < x_upper and y_lower < y < y_upper:
        return True
    return False
