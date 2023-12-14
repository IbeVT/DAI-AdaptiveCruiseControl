import math

import carla
import numpy as np
from ultralytics import YOLO

from LowpassFilter import LowpassFilter


class ComputerVision:
    def __init__(self, vehicle, radar_sample_rate=10):
        self.delta_v = None
        self.image = None
        self.object_points = None
        self.vehicle = vehicle
        self.inverse_camera_matrix = None
        self.radar_points = None
        self.model = YOLO('best.pt')
        self.model2 = YOLO('signs_best.pt')
        self.speed_classes = ['Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100', 'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30', 'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70', 'Speed Limit 80', 'Speed Limit 90', 'Stop']
        self.vehicle_classes = ['bus', 'bike', 'car', 'motorcycle', 'vehicle']
        self.camera_x_pixels = 720
        self.camera_y_pixels = 1280
        self.n_points = 50
        self.max_depth = 100
        self.max_speed = 120 / 3.6  # 120 km/h
        self.max_distance_following_vehicle = 50  # The maximum distance between the endpoint of the steer vector and the center of the bounding box of the vehicle
        self.following_vehicle_box = None
        self.following_vehicle_type = None
        self.distance = None
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
        
        results2 = self.model2.predict(source=self.image, save=False, conf=0.1)
        result2 = results2[0]

        # Check which car is in front, if any
        # To find the vehicle in front, we will use the steer angle and the azimuth angle We do it as follows:
        # 1. Get all boxes that are vehicles
        # 2. Group all points that belong to the same box
        # 3. For each box, calculate the median distance and speed
        # 4. Calculate where the vector with length=median_distance, azimuth=steer_angle and
        # altitude=mean_altitude_points would end on the camera
        # 5. Calculate the distance from the vector to the center of the bounding box of the vehicle
        # Reset the values
        self.following_vehicle_box = None
        self.distance = self.max_depth
        self.delta_v = self.max_speed

        smallest_distance_to_box = self.max_distance_following_vehicle  # The distance from the vector to the
        # center of the bounding box of the vehicle
        wheel_locations = carla.VehicleWheelLocation()
        self.wheel_angles.append(self.vehicle.get_wheel_steer_angle(wheel_locations.FL_Wheel))
        n_points = self.radar_sample_rate * 2  # Use the data from the last 2 seconds for the low pass filter
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
        speed_boxes = []
        previous_results = self.boxes
        previous_low_conf_results = self.low_conf_boxes
        self.boxes = []
        self.low_conf_boxes = []
        for box1 in result.boxes:

            class_id = result.names[box1.cls[0].item()]
            cords = box1.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box1.conf[0].item(), 2)
            print("Object type class1:", class_id)
            print("Coordinates:", cords)
            print("Probability:", conf)
            print("---")
                
            # If the confidence is high enough, immediately save the box
            if conf > 0.5:
                print("New box1 detected")
                self.boxes.append({"class_id": class_id, "cords": cords, "conf": conf})
                if str(class_id) in self.vehicle_classes:
                    vehicle_boxes.append({"class_id": class_id, "cords": cords, "conf": conf})
                
                continue
            # Check if a similar box was detected in the previous frame
            found = False
            for previous_box in previous_results:
                
                if "class_id" in previous_box:
                    class_id_previous = previous_box["class_id"]
                    cords_previous = previous_box["cords"]
                    
                if "class_id2" in previous_box:
                    class_id_previous = previous_box["class_id2"]
                    cords_previous = previous_box["cords2"]
                    
                
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
                    print("inicio",previous_box, "fin")
                    
                    if "class_id" in previous_box:
                        class_id_previous = previous_box["class_id"]
                        cords_previous = previous_box["cords"]
                        
                    if "class_id2" in previous_box:
                        class_id_previous = previous_box["class_id2"]
                        cords_previous = previous_box["cords2"]
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
        
        for box2 in result2.boxes:
            
            class_id2 = result2.names[box2.cls[0].item()]
            cords2 = box2.xyxy[0].tolist()
            cords2 = [round(x) for x in cords2]
            conf2 = round(box2.conf[0].item(), 2)
            print("Object type class2:", class_id2)
            print("Coordinates class2:", cords2)
            print("Probability class2:", conf2)
            print("---")
            
            if class_id2 != 'Green Light' and class_id2 != 'Red Light' and class_id2 != 'Stop':
                self.delta_v = int(class_id2[-3:])
                print('speed prueba 1', self.delta_v)
                
                print('\n\n\n\n\n AQUIIIIIIIIIIIIIIIIII \n\n\n\n\n')
                
            
            if conf2 > 0.5:
                print("New box2 detected")
                self.boxes.append({"class_id2": class_id2, "cords2": cords2, "conf2": conf2})
                if str(class_id) in self.vehicle_classes:
                    speed_boxes.append({"class_id2": class_id2, "cords2": cords2, "conf2": conf2})
                
                continue
               
                
            found = False
            for previous_box in previous_results:
                
                if "class_id" in previous_box:
                    class_id_previous = previous_box["class_id"]
                    cords_previous = previous_box["cords"]
                    
                if "class_id2" in previous_box:
                    class_id_previous = previous_box["class_id2"]
                    cords_previous = previous_box["cords2"]
                
                # Check if the class is the same
                if class_id2 == class_id_previous:
                    # Check whether the boxes overlap
                    if do_boxes_overlap(cords2, cords_previous):
                        # If approximately the same box was detected in the previous frame, we will suppose that it
                        # is indeed a true positive
                        self.boxes.append({"class_id2": class_id2, "cords2": cords2, "conf2": conf2})
                        
                        if str(class_id2) in self.speed_classes:
                            speed_boxes.append({"class_id2": class_id2, "cords2": cords2, "conf2": conf2})
                        found = True
                        break

            if not found:
                # Check if a similar box was detected with low confidence in the previous frame
                for previous_box in previous_low_conf_results:
                    
                    if "class_id" in previous_box:
                        class_id_previous = previous_box["class_id"]
                        cords_previous = previous_box["cords"]
                    
                    if "class_id2" in previous_box:
                        class_id_previous = previous_box["class_id2"]
                        cords_previous = previous_box["cords2"]
                    
                    # Check if the class is the same
                    if class_id2 == class_id_previous:
                        # Check whether the boxes overlap
                        if do_boxes_overlap(cords2, cords_previous):
                            # If approximately the same box was detected in the previous frame, we will suppose that it
                            # is indeed a true positive
                            if previous_box["conf2"] + conf2 > 0.6:
                                self.boxes.append({"class_id2": class_id2, "cords2": cords2, "conf2": conf2})
                                if str(class_id2) in self.speed_classes:
                                    speed_boxes.append({"class_id2": class_id2, "cords2": cords2, "conf2": conf2})
                                found = True
                                break
            if not found:
                # Still save the box, for debugging purposes
                self.low_conf_boxes.append({"class_id2": class_id2, "cords2": cords2, "conf2": conf2})
        # 2. Group all points that belong to the same box
        distances = []  # The distances of all points that belong to the i-th box
        velocities = []  # The speeds of all points that belong to the i-th box
        altitudes = []  # The altitudes of all points that belong to the i-th box
        for point in self.radar_points:
            [x, y] = self.get_image_coordinates_from_radar_point(point.azimuth, point.altitude, point.depth)
            for i, box in enumerate(vehicle_boxes):
                print(box)
                # If there is not yet a list for the i-th box, create it
                if len(distances) <= i:
                    distances.append([])
                    velocities.append([])
                    altitudes.append([])
                [x_lower, y_lower, x_upper, y_upper] = box["cords"]
                if x_lower < x < x_upper and y_lower < y < y_upper:
                    distances[i].append(point.depth)
                    velocities[i].append(point.velocity)
                    altitudes[i].append(point.altitude)
                    break
        # 3. For each box, calculate the median distance and speed
        distances = [np.median(i) for i in distances]
        velocities = [np.median(i) for i in velocities]
        altitudes = [np.median(i) for i in altitudes]
        # Now we have the median distance and speed for each box

        # 4. Calculate where the vector with length=median_distance and angle=steer_angle would end on the camera
        # 5. Calculate the distance from the vector to the center of the bounding box of the vehicle
        # Initialize the steer vector endpoint to half of the maximum depth and elevation 0
        self.steer_vector_endpoint = self.get_image_coordinates_from_radar_point(steer_angle, 0, self.max_depth / 2)
        for i, box in enumerate(vehicle_boxes):
            altitude = altitudes[i]
            # If there are no points in the box, skip it
            if math.isnan(altitude):
                continue
            cords = self.get_image_coordinates_from_radar_point(steer_angle, altitude, distances[i])
            distance_to_box = math.sqrt((cords[0] - np.mean([box["cords"][0], box["cords"][2]])) ** 2 +
                                        ((cords[1] - np.mean([box["cords"][1], box["cords"][
                                            3]])) / 2) ** 2)  # Y is less important than X
            print("Box:", box)
            print("Distance to box:", distance_to_box)
            if distance_to_box < smallest_distance_to_box:
                self.following_vehicle_box = box
                self.distance = distances[i]
                self.delta_v = velocities[i]
                self.steer_vector_endpoint = cords

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
