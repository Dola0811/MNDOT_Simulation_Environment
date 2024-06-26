"""EKF_fixed_path controller."""

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for environments without a display
import matplotlib.pyplot as plt
import numpy as np
from controller import Robot, Motor, Keyboard, GPS, Accelerometer, Gyro
from collections import deque
import logging

TIME_STEP = 1000
number_of_rssi_sources = 5  # Define how many RSSI measurements you expect

def setup_logger():
    logging.basicConfig(filename='simulation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_speed(wheels, leftSpeed, rightSpeed):
    """ Set the speed of the wheels """
    for i, wheel in enumerate(wheels):
        wheel.setVelocity(leftSpeed if i % 2 == 0 else rightSpeed)

# Define path for simulation
path = [
    (1.0, 0.0, 10),  # Move forward for 10 steps
    (0.0, 1.0, 5),   # Rotate clockwise for 5 steps
    (1.0, 0.0, 10),  # Move forward for 10 steps
    (0.0, -1.0, 5)   # Rotate counter-clockwise for 5 steps
]

def follow_path(wheels, path):
    for speed, turn, duration in path:
        set_speed(wheels, speed + turn, speed - turn)
        yield from range(duration)

# Non-linear state transition and observation functions
def f(x, u):
    dt = 1
    return np.array([
        x[0] + u[0] * np.cos(x[2]) * dt,
        x[1] + u[0] * np.sin(x[2]) * dt,
        x[2] + u[1] * dt
    ])

def h(x):
    return np.array([x[0], x[1]])

# Jacobians
def jacobian_f(x, u):
    dt = 1
    return np.array([
        [1, 0, -u[0] * np.sin(x[2]) * dt],
        [0, 1, u[0] * np.cos(x[2]) * dt],
        [0, 0, 1]
    ])

def jacobian_h(x):
    return np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])

def ekf_update(x, P, gps_data, imu_data, rssi_data, R, Q, u):
    F = jacobian_f(x, u)
    x_pred = f(x, u)
    P_pred = F @ P @ F.T + Q

    if np.isnan(gps_data).any():
        gps_data = x_pred[:2]  # Use predicted state when GPS data is missing
        R[:2, :2] *= 10

    H = jacobian_h(x_pred)
    z_pred = h(x_pred)
    z = gps_data
    y = z - z_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_updated = x_pred + K @ y
    P_updated = (np.eye(len(x)) - K @ H) @ P_pred

    return x_updated, P_updated

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mae(predictions, targets):
    return np.abs(predictions - targets).mean()

def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()

def max_error(predictions, targets):
    return np.max(np.abs(predictions - targets))

def simulate_robot_movement(x, u, dt=1):
    """ Simulate the real movement of the robot, which will be considered as ground truth """
    noise_level = 0.05  # Noise level in the control input
    noisy_u = u + np.random.randn(*u.shape) * noise_level
    return np.array([
        x[0] + noisy_u[0] * np.cos(x[2]) * dt,
        x[1] + noisy_u[0] * np.sin(x[2]) * dt,
        x[2] + noisy_u[1] * dt
    ])

def main():
    setup_logger()
    robot = Robot()
    gps = robot.getDevice("global")
    gps.enable(TIME_STEP)
    acc = robot.getDevice("accelerometer")
    acc.enable(TIME_STEP)
    gyro = robot.getDevice("gyro")
    gyro.enable(TIME_STEP)
    wheels = [robot.getDevice(name) for name in ['wheel1', 'wheel2', 'wheel3', 'wheel4']]
    for wheel in wheels:
        wheel.setPosition(float('inf'))
        wheel.setVelocity(0)

    x = np.zeros(3)  # Initial state for EKF
    true_x = np.copy(x)  # Separate state for ground truth
    P = np.eye(3) * 100  # Initial covariance
    Q = np.eye(3) * 0.1  # Process noise
    R = np.eye(2) * 9  # Measurement noise
    path_iterator = follow_path(wheels, path)

    ground_truth = []
    measurements = []
    filtered_positions = []
    step_count = 0  # Initialize step count

    while robot.step(TIME_STEP) != -1:
        try:
            next(path_iterator)  # Proceed to the next step in the fixed path
        except StopIteration:
            break  # End the simulation if the path is complete

        # Control inputs
        u = np.array([1.0, 0.1])  # Example control inputs
        true_x = simulate_robot_movement(true_x, u)  # Update ground truth

        # Generate noisy GPS data
        gps_data = true_x[:2] + np.random.randn(2) * 0.1  # Add noise to ground truth for GPS measurement

        imu_data = np.random.randn(3)
        rssi_data = np.random.randn(number_of_rssi_sources)
        
        x, P = ekf_update(x, P, gps_data, imu_data, rssi_data, R, Q, u)

        ground_truth.append(true_x)  # Append the true state
        measurements.append(gps_data)
        filtered_positions.append(x[:2])

        logging.info(f"Step {step_count}: True state: {true_x}, EKF state: {x}, GPS data: {gps_data}")
        print(f"Step {step_count}: Ground Truth: {ground_truth[-1]}, GPS Measurement: {measurements[-1]}, EKF Position: {filtered_positions[-1]}")
        step_count += 1

    # Convert lists to numpy arrays for easier manipulation
    ground_truth_np = np.array(ground_truth)
    filtered_positions_np = np.array(filtered_positions)

    # Ensure there is ground truth and filtered data to compare
    if len(ground_truth_np) > 0 and len(filtered_positions_np) > 0:
        # Calculate metrics
        ground_truth_xy = ground_truth_np[:, :2]  # Assuming ground truth contains [x, y, theta]
        filtered_xy = filtered_positions_np

        error_rmse = rmse(filtered_xy, ground_truth_xy)
        error_mae = mae(filtered_xy, ground_truth_xy)
        error_mse = mse(filtered_xy, ground_truth_xy)
        error_max = max_error(filtered_xy, ground_truth_xy)

        # Print out the metrics
        print(f"RMSE: {error_rmse}")
        print(f"MAE: {error_mae}")
        print(f"MSE: {error_mse}")
        print(f"Max Error: {error_max}")

    plot_positions(ground_truth, measurements, filtered_positions)

def plot_positions(ground_truth, measured, filtered):
    plt.figure(figsize=(10, 8))
    if ground_truth:
        plt.plot([pos[0] for pos in ground_truth], [pos[1] for pos in ground_truth], 'g-', label='Ground Truth')
    if measured:
        plt.plot([pos[0] for pos in measured], [pos[1] for pos in measured], 'r:', label='Measured GPS')
    if filtered:
        plt.plot([pos[0] for pos in filtered], [pos[1] for pos in filtered], 'b--', label='Filtered GPS')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('GPS Tracking with EKF')
    plt.legend()
    plt.grid(True)
    plt.savefig('GPS_Tracking_with_EKF.png')
    plt.close()

if __name__ == "__main__":
    main()
