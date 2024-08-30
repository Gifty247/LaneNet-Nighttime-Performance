import sys
import os
import torch
import numpy as np
import cv2
import random
import time
import queue
import glob
import math
import argparse
import csv
from datetime import datetime

# Add the CARLA egg file to the PYTHONPATH from the examples folder
sys.path.append(os.path.abspath('carla-0.9.10-py3.7-win-amd64.egg'))

# Add the LaneNet model directory to the PYTHONPATH
sys.path.append('C:/Users/giftc/lanenet-lane-detection-pytorch')  # Ensure this path is correct

try:
    import carla
except ImportError as e:
    print("CARLA module not found. Please check the .egg file path and installation.")
    raise e

try:
    from model.lanenet.LaneNet import LaneNet
except ImportError as e:
    print("LaneNet module not found. Please check the model directory path.")
    raise e

from model.lanenet.LaneNet import LaneNet
import carla

def get_speed(vehicle):
    vel = vehicle.get_velocity()
    speed = 3.6*math.sqrt(vel.x**2 +vel.y**2+vel.z**2)
    print(f"Current Speed: {speed:.2f} km/h")
    return speed    

def get_position(vehicle):
    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    print(f"Vehicle Position: (x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f})")
    print(f"Vehicle Orientation: (yaw={rotation.yaw:.2f}, pitch={rotation.pitch:.2f}, roll={rotation.roll:.2f})")
    return location, rotation

def save_lane_image(image, lanes, filename="lane_output.png"):
    lane_image = visualize_lanes_on_image(image, lanes)
    cv2.imwrite(filename, lane_image)
    print(f"Saved lane output image as {filename}")

class VehiclePIDController():
    def __init__(self, vehicle, args_lateral, args_longitudinal, max_throttle = 0.75, max_break = 0.3, max_steering= 0.8):
        self.max_break = max_break
        self.max_steering = max_steering
        self.max_throttle = max_throttle


        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.past_steering = self.vehicle.get_control().steer
        self.long_controller = PIDLongitudinalControl(self.vehicle, **args_longitudinal)
        self.lat_controller = PIDLateralControl(self.vehicle, **args_lateral)
        

    def run_step(self, target_speed, waypoint):
        # Get current position and orientation
        location, rotation = get_position(self.vehicle)

        # Control calculation
        acceleration = self.long_controller.run_step(target_speed)
        current_steering = self.lat_controller.run_step(waypoint)
        control = carla.VehicleControl()

        if acceleration>=0.0:
            control.throttle = min(abs(acceleration), self.max_break)
            control.brake = 0.0
        else :
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_break)

        if current_steering > self.past_steering+0.1:
            current_steering = self.past_steering+0.1
        
        elif current_steering<self.past_steering-0.1:
            current_steering = self.past_steering-0.1
        
        if current_steering>=0:
            steering = min(self.max_steering , current_steering)
           
        else:
            steering = max(-self.max_steering , current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering
        
        # Debug output
        print(f"Throttle: {control.throttle:.2f}, Brake: {control.brake:.2f}, Steer: {control.steer:.2f}")

        return control

class PIDLongitudinalControl():
    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt = 0.03):       
        self.vehicle = vehicle
        self.K_D = K_D
        self.K_P = K_P
        self.K_I = K_I
        self.dt = dt
        self.errorBuffer = queue.deque(maxlen = 10)
    
    def pid_controller(self,target_speed,current_speed):

        error = target_speed-current_speed

        self.errorBuffer.append(error)

        if len(self.errorBuffer)>=2:
            de = (self.errorBuffer[-1]-self.errorBuffer[-2])/self.dt

            ie = sum(self.errorBuffer)*self.dt 
        
        else:
            de = 0.0
            ie = 0.0
        
        return np.clip(self.K_P*error+self.K_D*de+self.K_I*ie , -1.0,1.0)


    def run_step(self, target_speed):
        current_speed = get_speed(self.vehicle)
        return self.pid_controller(target_speed,current_speed)
    
class PIDLateralControl():

    def __init__(self, vehicle, K_P=1.0, K_D=0.0, K_I=0.0, dt = 0.03):
        
        self.vehicle = vehicle
        self.K_D = K_D
        self.K_P = K_P
        self.K_I = K_I
        self.dt = dt
        self.errorBuffer = queue.deque(maxlen = 10)
    
    def run_step(self,waypoint):

        return self.pid_controller(waypoint, self.vehicle.get_transform())

    def pid_controller(self,waypoint, vehicle_transform):

        v_begin = vehicle_transform.location
        v_end = v_begin+carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),y=math.sin(math.radians(vehicle_transform.rotation.yaw)))
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])

        w_vec = np.array([waypoint.transform.location.x - v_begin.x, waypoint.transform.location.y - v_begin.y, 0.0])

        dot = math.acos(np.clip(np.dot(w_vec,v_vec)/np.linalg.norm(w_vec)*np.linalg.norm(v_vec),-1.0,1.0))

        cross = np.cross(v_vec,w_vec)
        if cross[2]<0:
            dot*=-1

        self.errorBuffer.append(dot)

        if len(self.errorBuffer)>=2:
                de = (self.errorBuffer[-1]-self.errorBuffer[-2])/self.dt
                ie = sum(self.errorBuffer)*self.dt
        
        else:
            de = 0.0
            ie = 0.0
        
        return np.clip((self.K_P*dot)+(self.K_I*ie)+(self.K_D*de), -1.0,1.0)

def preprocess_image(image):
    image = cv2.resize(image, (512, 256))  # Resize as required by your model
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Add batch and channel dimensions
    return image

def postprocess_lanes(output):
    binary_mask = output['binary_seg_pred'].squeeze(0).cpu().numpy()
    return binary_mask

def visualize_lanes_on_image(image, lane_mask):
    lane_image = np.zeros_like(image)
    lane_image[lane_mask == 1] = [0, 255, 0]  # Green lanes
    combined_image = cv2.addWeighted(image, 1.0, lane_image, 0.5, 0)
    return combined_image


def adjust_vehicle_control_based_on_lanes(control_vehicle, lanes_lines):
    if len(lanes_lines) == 2:
        # Example: Find the midpoint between two lane lines and adjust steering
        left_lane = lanes_lines[0]
        right_lane = lanes_lines[1]
        
        # Calculate the midpoint of the lanes
        left_midpoint = np.mean(left_lane, axis=0)[0]
        right_midpoint = np.mean(right_lane, axis=0)[0]
        lane_midpoint = (left_midpoint + right_midpoint) / 2

        # Example logic to adjust steering based on the lane midpoint
        image_center = 400  # Assuming 800px wide image
        offset = lane_midpoint[0] - image_center

        # Adjust steering: if offset is positive, steer right; if negative, steer left
        control_vehicle.max_steering = np.clip(-0.001 * offset, -1, 1)  # Adjust gain as needed


def main():
    actor_list = []
    try:

        # Open CSV file for writing
        with open('carla_simulation_output.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the headers
            writer.writerow(['Time', 'Speed (km/h)', 'Throttle', 'Brake', 'Steer'])
        
        
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        map = world.get_map

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('model3')[0]
        spawnpoint = carla.Transform(carla.Location(x=-75.4,y=-1.0,z=15),carla.Rotation(pitch = 0, yaw = 180, roll = 0))
        vehicle = world.spawn_actor(vehicle_bp, spawnpoint)
        actor_list.append(vehicle)

        # Load the trained LaneNet model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lanenet_model = LaneNet(arch='ENet')
        lanenet_model.load_state_dict(torch.load('C:/Users/giftc/lanenet-lane-detection-pytorch/log/best_model.pth', map_location=device))
        lanenet_model.to(device)
        lanenet_model.eval()

        control_vehicle = VehiclePIDController(vehicle , args_lateral={'K_P':1, 'K_D':0.0, 'K_I':0.0}, args_longitudinal={'K_P':1, 'K_D':0.0, 'K_I':0.0})

        camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5,y=2.4))
        camera = world.spawn_actor(camera_bp , camera_transform)
        actor_list.append(camera)

        # Define a callback to process images and use LaneNet model
        def process_image(image):
            # Convert CARLA image to a numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Convert BGRA to RGB

            # Preprocess the image for LaneNet
            image_tensor = preprocess_image(array)
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                output = lanenet_model(image_tensor)

            # Post-process the LaneNet output to extract lane information
            lanes = postprocess_lanes(output)

            # Modify vehicle control based on lane information
            adjust_vehicle_control_based_on_lanes(control_vehicle, lanes)
           
        # Attach the process_image callback to the camera sensor
        camera.listen(lambda image: process_image(image))

        while True:
            waypoints = world.get_map().get_waypoint(vehicle.get_location())
            waypoint = np.random.choice(waypoints.next(0.3))
            control_signal = control_vehicle.run_step(5, waypoint)
            vehicle.apply_control(control_signal)

            #speed = get_speed(vehicle)
            #vehicle_location = vehicle.get_location()
            #vehicle_rotation = vehicle.get_transform().rotation
            #distance_to_waypoint = vehicle_location.distance(waypoint.transform.location)
            
            # Log to CSV
            with open('carla_simulation_output.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now().strftime('%H:%M:%S.%f'),
                    get_speed(vehicle),  # Speed
                    control_signal.throttle,  # Throttle
                    control_signal.brake,  # Brake
                    control_signal.steer  # Steer
                ])

    finally:
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])


if __name__ == '__main__':
    main()