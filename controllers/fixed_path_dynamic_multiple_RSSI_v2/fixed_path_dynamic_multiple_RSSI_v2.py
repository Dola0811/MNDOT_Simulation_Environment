import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for environments without a display
import matplotlib.pyplot as plt
import numpy as np
from controller import Robot, Motor, Keyboard, GPS, Accelerometer, Gyro
from collections import deque
import logging

TIME_STEP = 1000

number_of_rssi_sources = 5  # Define how many RSSI measurements you expect

# Setup logging
logging.basicConfig(filename='filter_performance.log', level=logging.INFO, format='%(asctime)s - %(message)s')


def is_gps_data_valid(gps_data):
    """Utility function to check if GPS data is valid."""
    return gps_data is not None and not np.isnan(gps_data).any()

def handle_keyboard(kb, wheels):
    """Handle keyboard inputs to control robot's wheels."""
    key = kb.getKey()
    leftSpeed = 0.0
    rightSpeed = 0.0
    if key == ord('W'):  # Forward
        leftSpeed = 1.0
        rightSpeed = 1.0
    elif key == ord('S'):  # Backward
        leftSpeed = -1.0
        rightSpeed = -1.0
    elif key == ord('D'):  # Turn right
        leftSpeed = 1.0
        rightSpeed = -1.0
    elif key == ord('A'):  # Turn left
        leftSpeed = -1.0
        rightSpeed = 1.0
    for wheel in wheels:
        wheel.setVelocity(leftSpeed if wheel.getName() in ['wheel1', 'wheel3'] else rightSpeed)

# Global list to store GPS data history
gps_history = deque(maxlen=10)  # Adjust size based on desired history length

def read_sensors(gps, acc, gyro):
    gps_data = gps.getValues()
    acc_data = acc.getValues()
    gyro_data = gyro.getValues()
    imu_data = np.hstack([acc_data, gyro_data])

    if is_gps_data_valid(gps_data):
        gps_history.append(gps_data)
    else:
        gps_data = np.array([np.nan, np.nan, np.nan])  # Mark as missing

    # Simulate multiple RSSI readings
    rssi_measurements = [np.random.normal(-70, 4) for _ in range(number_of_rssi_sources)]

    return gps_data, imu_data, rssi_measurements

def inverse_distance_weighting(current_time_index, all_positions, n_neighbors=3, power=2):
    """Apply IDW to interpolate missing GPS data based on recent measurements."""
    # Ensure all positions are NumPy arrays
    all_positions = [np.array(pos) for pos in all_positions if isinstance(pos, list) or isinstance(pos, np.ndarray)]
    
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
    if np.isnan(gps_data).any():
        # Attempt to interpolate using IDW if GPS data is missing
        interpolated_gps = inverse_distance_weighting(len(gps_history), list(gps_history))
        if np.isnan(interpolated_gps).any():
            gps_data = x[:3]  # Use last predicted state if IDW fails
            R[:3, :3] *= 10  # Increase measurement uncertainty
        else:
            gps_data = interpolated_gps  # Use interpolated data

    # Use RSSI patterns as part of the state
    rssi_measurement = np.mean(rssi_data)  # Simplified: use mean RSSI as a single measurement
    z = np.hstack([gps_data, rssi_measurement])  # Extend measurement vector to include RSSI
    u = imu_data  # Control input: IMU data
    x_pred = F @ x + B @ u  # Predict state
    P_pred = F @ P @ F.T + Q  # Predict uncertainty
    y = z - H @ x_pred  # Measurement residual
    S = H @ P_pred @ H.T + R  # Residual covariance
    K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
    x_updated = x_pred + K @ y  # Update estimate
    P_updated = (np.eye(len(x)) - K @ H) @ P_pred  # Update uncertainty
    return x_updated, P_updated

def calculate_residuals(measured, filtered):
    """Calculate residuals between measured and filtered GPS data."""
    return np.array(measured) - np.array(filtered)

def plot_residuals(residuals):
    """Plot the residuals of GPS estimates."""
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

def plot_positions(ground_truth, measured, filtered):
    plt.figure(figsize=(10, 8))

    # Convert lists to numpy arrays for easier handling
    ground_truth = np.array(ground_truth)
    measured = np.array([m if m is not None else [np.nan, np.nan, np.nan] for m in measured])
    filtered = np.array(filtered)

    # Handling NaN values explicitly
    ground_truth = np.where(np.isnan(ground_truth), np.nan, ground_truth)
    measured = np.where(np.isnan(measured), np.nan, measured)
    filtered = np.where(np.isnan(filtered), np.nan, filtered)

    # Plotting ground truth
    if not np.all(np.isnan(ground_truth)):
        plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'g-', label='Ground Truth', marker='s', markersize=12, linewidth=1)

    # Plotting measured GPS positions; uses NaN to handle missing data seamlessly
    if not np.all(np.isnan(measured)):
        plt.plot(measured[:, 0], measured[:, 1], 'r:', label='Measured GPS', linewidth=2)

    # Plotting filtered GPS positions
    if not np.all(np.isnan(filtered)):
        plt.plot(filtered[:, 0], filtered[:, 1], 'b--', label='Filtered GPS', linewidth=2, marker='^', markersize=10)

    # Setting up plot labels and layout
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Tracking with Kalman Filter')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.grid(True)
    plt.axis('equal')  # Ensures equal aspect ratio

    plt.savefig('GPS_Tracking_with_Kalman_Filter.png')
    plt.close()

def build_measurement_model(number_of_rssi_sources):
    # Create the top part for GPS data
    H_gps = np.zeros((3, 9))  # Assuming state vector size is 9
    np.fill_diagonal(H_gps, 1)
    
    # Create the bottom part for RSSI data
    H_rssi = np.zeros((1, 9))
    H_rssi[0, -1] = 1  # Assuming RSSI affects the last state
    
    # Combine both parts
    return np.vstack((H_gps, H_rssi))

# Use this function to generate H dynamically
H = build_measurement_model(number_of_rssi_sources)

def build_noise_covariance(number_of_rssi_sources):
    R_gps = np.array([[9, 0, 0], [0, 9, 0], [0, 0, 9]])  # GPS uncertainty
    R_rssi = np.array([[16]])  # RSSI uncertainty for a single measurement
    
    # Combine both matrices
    return np.block([
        [R_gps, np.zeros((3, 1))],
        [np.zeros((1, 3)), R_rssi]
    ])

# Use this function to generate R dynamically
R = build_noise_covariance(number_of_rssi_sources)

# Example path with more detailed movements: [(leftSpeed, rightSpeed), duration in steps]
path = [
    (1.0, 1.0, 5),   # Move forward for 5 steps
    (1.0, -1.0, 2),  # Turn right for 2 steps
    (1.0, 1.0, 3),   # Move forward for 3 steps
    (-1.0, -1.0, 2), # Move backward for 2 steps
    (-1.0, 1.0, 2),  # Turn left for 2 steps
    (1.0, 1.0, 4),   # Move forward for 4 steps
    (0.0, 0.0, 1),   # Stop for 1 steps
    (-1.0, 1.0, 2),  # Turn left for 2 steps
    (1.0, 1.0, 3),   # Move forward for 3 steps
    (1.0, -1.0, 2),  # Turn right for 2 steps
    (0.0, 0.0, 1)    # Stop for 1 steps
]

def calculate_rmse(ground_truth, predictions):
    """Calculate the Root Mean Square Error (RMSE) between ground truth and predictions."""
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    # Ensure we only calculate where both ground truth and predictions are available
    mask = ~np.isnan(ground_truth).any(axis=1) & ~np.isnan(predictions).any(axis=1)
    valid_ground_truth = ground_truth[mask]
    valid_predictions = predictions[mask]
    return np.sqrt(np.mean((valid_ground_truth - valid_predictions) ** 2))

def calculate_mae(ground_truth, predictions):
    """Calculate the Mean Absolute Error (MAE) between ground truth and predictions."""
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    mask = ~np.isnan(ground_truth).any(axis=1) & ~np.isnan(predictions).any(axis=1)
    valid_ground_truth = ground_truth[mask]
    valid_predictions = predictions[mask]
    return np.mean(np.abs(valid_ground_truth - valid_predictions))

def calculate_mse(ground_truth, predictions):
    """Calculate the Mean Squared Error (MSE) between ground truth and predictions."""
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    mask = ~np.isnan(ground_truth).any(axis=1) & ~np.isnan(predictions).any(axis=1)
    valid_ground_truth = ground_truth[mask]
    valid_predictions = predictions[mask]
    return np.mean((valid_ground_truth - valid_predictions) ** 2)

def calculate_max_error(ground_truth, predictions):
    """Calculate the maximum error at any step between ground truth and predictions."""
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    mask = ~np.isnan(ground_truth).any(axis=1) & ~np.isnan(predictions).any(axis=1)
    valid_ground_truth = ground_truth[mask]
    valid_predictions = predictions[mask]
    return np.max(np.abs(valid_ground_truth - valid_predictions))

def plot_metrics(steps, rmse_values, mae_values, mse_values, max_error_values):
    plt.figure(figsize=(12, 8))

    titles = ['Root Mean Square Error Over Time', 'Mean Absolute Error Over Time',
              'Mean Squared Error Over Time', 'Maximum Error Over Time']
    data_series = [rmse_values, mae_values, mse_values, max_error_values]
    labels = ['RMSE', 'MAE', 'MSE', 'Max Error']

    for i, data in enumerate(data_series):
        plt.subplot(2, 2, i + 1)
        plt.plot(steps, data, label=labels[i], marker='o', linestyle='-', color='b')
        plt.title(titles[i])
        plt.xlabel('Simulation Step')
        plt.ylabel(labels[i])
        plt.grid(True)
        plt.autoscale(enable=True, axis='y', tight=None)

    plt.tight_layout()
    plt.savefig('Performance_Metrics_Over_Time.png')
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
        wheel.setPosition(float('inf'))  # Set to infinite rotation
        wheel.setVelocity(0)
        wheels.append(wheel)

    kb = Keyboard()
    kb.enable(TIME_STEP)

    x = np.zeros(9)
    P = np.eye(9) * 100
    F = np.array([
        [1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    B = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    Q = np.eye(9) * 0.1
    step_count = 0
    ground_truth_positions = []
    measured_positions = []
    filtered_positions = []

    current_command = 0
    command_duration = 0

    consecutive_missing_steps = 3  # Number of consecutive steps with missing GPS data
    missing_gps_start_step = 5  # Step at which missing GPS data starts

    # Initialize lists to store metrics for plotting
    rmse_values = []
    mae_values = []
    mse_values = []
    max_error_values = []
    steps = []  # Initialize steps list here to track each step count

    while robot.step(TIME_STEP) != -1 and current_command < len(path):
        handle_keyboard(kb, wheels)  # Handle keyboard inputs

        if command_duration >= path[current_command][2]:
            current_command += 1
            command_duration = 0

        if current_command < len(path):
            leftSpeed, rightSpeed, _ = path[current_command]
            for i in range(4):
                wheels[i].setVelocity(leftSpeed if i % 2 == 0 else rightSpeed)
            command_duration += 1

        # Read sensors
        gps_data, imu_data, rssi_data = read_sensors(gps, acc, gyro)
        x, P = kalman_filter_update(x, P, gps_data, imu_data, rssi_data, H, R, F, B, Q)
        ground_truth_positions.append(gps_data)
        
        # Introduce consecutive missing GPS data
        if missing_gps_start_step <= step_count < missing_gps_start_step + consecutive_missing_steps:
            noisy_gps = np.array([np.nan, np.nan, np.nan])
        else:
            noisy_gps = gps_data + np.random.normal(0, 0.05, 3)  # Adding noise to GPS for realism

        measured_positions.append(noisy_gps)
        filtered_positions.append(x[:3])

        print(f"Step {step_count}:")
        print("Ground Truth:", gps_data)
        print("Measured GPS:", noisy_gps if not np.isnan(noisy_gps).all() else "Data Missing")
        print("Filtered GPS:", x[:3])

        step_count += 1

    residuals = calculate_residuals(measured_positions, filtered_positions)
    plot_residuals(residuals)
    plot_positions(ground_truth_positions, measured_positions, filtered_positions)

    # Metric calculation after simulation ends
    rmse_value = calculate_rmse(ground_truth_positions, filtered_positions)
    mae_value = calculate_mae(ground_truth_positions, filtered_positions)
    mse_value = calculate_mse(ground_truth_positions, filtered_positions)
    max_error_value = calculate_max_error(ground_truth_positions, filtered_positions)

    # Print metrics to console
    print(f"Final RMSE: {rmse_value}")
    print(f"Final MAE: {mae_value}")
    print(f"Final MSE: {mse_value}")
    print(f"Maximum Error: {max_error_value}")

    # Log the metrics for later review and analysis
    logging.info(f"RMSE: {rmse_value}, MAE: {mae_value}, MSE: {mse_value}, Max Error: {max_error_value}")

    ground_truth = gps.getValues()  # This method might not exist; adjust based on your system
    predicted = x[:3]  # Assuming 'x' holds the predicted state including position
   
    if len(ground_truth_positions) > 0 and len(filtered_positions) > 0:  # Ensure there's data to calculate metrics
        current_rmse = calculate_rmse([ground_truth], [predicted])
        current_mae = calculate_mae([ground_truth], [predicted])
        current_mse = calculate_mse([ground_truth], [predicted])
        current_max_error = calculate_max_error([ground_truth], [predicted])

        rmse_values.append(current_rmse)
        mae_values.append(current_mae)
        mse_values.append(current_mse)
        max_error_values.append(current_max_error)
        steps.append(step_count)  # Track the step count for plotting   

    plot_metrics(steps, rmse_values, mae_values, mse_values, max_error_values)

if __name__ == "__main__":
    main()

