"""fixed_path_dynamic_multiple_RSSI controller."""

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for environments without a display
import matplotlib.pyplot as plt
import numpy as np
from controller import Robot, Motor, GPS, Accelerometer, Gyro
from collections import deque

TIME_STEP = 1000
number_of_rssi_sources = 100  # Define how many RSSI measurements you expect

def is_gps_data_valid(gps_data):
    """Utility function to check if GPS data is valid."""
    return gps_data is not None and not np.isnan(gps_data).any()

def calculate_heading(current_position, target_position):
    """Calculate the desired heading angle from current position to target position."""
    dy = target_position[1] - current_position[1]
    dx = target_position[0] - current_position[0]
    return np.arctan2(dy, dx)

def follow_path(waypoints, robot, wheels, gps):
    """Follow the predefined waypoints."""
    current_waypoint_index = 0
    while robot.step(TIME_STEP) != -1 and current_waypoint_index < len(waypoints):
        current_position = gps.getValues()[:2]  # Assuming the first two values are latitude and longitude
        if np.linalg.norm(np.array(current_position) - np.array(waypoints[current_waypoint_index])) < 0.5:
            current_waypoint_index += 1
            if current_waypoint_index == len(waypoints):
                break  # Stop moving after the last waypoint is reached
        desired_heading = calculate_heading(current_position, waypoints[current_waypoint_index])
        left_speed = right_speed = 5.0  # Adjust speed as necessary
        wheels[0].setVelocity(left_speed)
        wheels[1].setVelocity(right_speed)
        wheels[2].setVelocity(left_speed)
        wheels[3].setVelocity(right_speed)

# Global list to store GPS data history
gps_history = deque(maxlen=10)

def read_sensors(gps, acc, gyro):
    gps_data = gps.getValues()
    acc_data = acc.getValues()
    gyro_data = gyro.getValues()
    imu_data = np.hstack([acc_data, gyro_data])
    if is_gps_data_valid(gps_data):
        gps_history.append(gps_data)
    else:
        gps_data = np.array([np.nan, np.nan, np.nan])
    rssi_measurements = [np.random.normal(-70, 4) for _ in range(number_of_rssi_sources)]
    return gps_data, imu_data, rssi_measurements

def inverse_distance_weighting(current_time_index, all_positions, n_neighbors=3, power=2):
    valid_indices = [i for i in range(max(0, current_time_index - n_neighbors), current_time_index) if not np.isnan(all_positions[i]).any()]
    if not valid_indices:
        return np.array([np.nan, np.nan, np.nan])
    distances = [abs(i - current_time_index) for i in valid_indices]
    weights = [1 / (d ** power) for d in distances]
    weighted_positions = np.array([all_positions[i] * weights[idx] for idx, i in enumerate(valid_indices)])
    weighted_sum = np.sum(weighted_positions, axis=0)
    total_weight = np.sum(weights)
    return weighted_sum / total_weight if total_weight else np.array([np.nan, np.nan, np.nan])

def kalman_filter_update(x, P, gps_data, imu_data, rssi_data, H, R, F, B, Q):
    # GPS interpolation or use direct GPS data here
    z = np.hstack([gps_data, *rssi_data])  # Ensure rssi_data is a list of RSSI values
    u = imu_data  # Control input: IMU data
    x_pred = F @ x + B @ u  # Predict state
    P_pred = F @ P @ F.T + Q  # Predict uncertainty
    y = z - H @ x_pred  # Measurement residual
    S = H @ P_pred @ H.T + R  # Residual covariance
    K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
    x_updated = x_pred + K @ y  # Update estimate
    P_updated = (np.eye(len(x)) - K @ H) @ P_pred  # Update uncertainty
    return x_updated, P_updated

def plot_residuals(residuals):
    plt.figure(figsize=(10, 5))
    plt.plot(residuals[:, 0], label='Longitude Residuals')
    plt.plot(residuals[:, 1], label='Latitude Residuals')
    plt.title('Residuals of Kalman Filter')
    plt.xlabel('Time Step')
    plt.ylabel('Residual')
    plt.legend()
    plt.grid(True)
    plt.savefig('Residuals_Kalman_Filter.png')
    plt.close()

def main():
    robot = Robot()
    gps = robot.getDevice("global")
    gps.enable(TIME_STEP)
    acc = robot.getDevice("accelerometer")
    acc.enable(TIME_STEP)
    gyro = robot.getDevice("gyro")
    gyro.enable(TIME_STEP)
    wheels = []
    wheel_names = ['wheel1', 'wheel2', 'wheel3', 'wheel4']
    for name in wheel_names:
        wheel = robot.getDevice(name)
        wheel.setPosition(float('inf'))
        wheel.setVelocity(0)
        wheels.append(wheel)
    x = np.zeros(9)
    P = np.eye(9) * 100
    # Define a simple path
    waypoints = [(0, 0), (10, 0), (10, 10), (0, 10)]
    follow_path(waypoints, robot, wheels, gps)

if __name__ == "__main__":
    main()
