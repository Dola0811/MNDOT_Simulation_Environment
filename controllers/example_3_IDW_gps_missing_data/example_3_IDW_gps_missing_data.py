import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for environments without a display
import matplotlib.pyplot as plt
import numpy as np
from controller import Robot, Motor, Keyboard, GPS, Accelerometer, Gyro
from collections import deque

TIME_STEP = 1000

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

def read_sensors(gps, acc, gyro):
    """Read sensor data from Webots devices, handle missing GPS data robustly."""
    gps_data = gps.getValues() 
    acc_data = acc.getValues()
    gyro_data = gyro.getValues()
    imu_data = np.hstack([acc_data, gyro_data])

    # Assume missing data when the GPS reading fails certain criteria
    if not is_gps_data_valid(gps_data):
        gps_data = np.array([np.nan, np.nan, np.nan])

    rssi_measurement = np.random.normal(-70, 4)  # Simulated RSSI value
    return gps_data, imu_data, rssi_measurement






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
    """Perform a Kalman filter update using the provided measurements and noise characteristics."""
    if np.isnan(gps_data).any():
        gps_data = x[:3]  # Use predicted position if GPS data is missing
        R[:3, :3] *= 10  # Increase measurement uncertainty
    z = np.hstack([gps_data, rssi_data])  # Measurement vector
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
    plt.show(block=True)


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

    kb = robot.getKeyboard()
    kb.enable(TIME_STEP)


    x = np.zeros(9)
    P = np.eye(9) * 100
    availability_state = {'available': True, 'next_change': 50}
    
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
    H = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    Q = np.eye(9) * 0.1
    R = np.array([
        [9, 0, 0, 0],
        [0, 9, 0, 0],
        [0, 0, 9, 0],
        [0, 0, 0, 16]
    ])

    step_count = 0  # Initialize step counter for simulated GPS outages
    ground_truth_positions = []
    measured_positions = []
    filtered_positions = []

    # Initialize counters
    missing_steps = 0  # Steps until data is missing again
    available_steps = np.random.randint(5, 11)  # Randomly c

    while robot.step(TIME_STEP) != -1:
        key = kb.getKey()
        leftSpeed = 0.0
        rightSpeed = 0.0
        if key == 315:  # Forward
            leftSpeed = 1.0
            rightSpeed = 1.0
        elif key == 317:  # Backward
            leftSpeed = -1.0
            rightSpeed = -1.0
        elif key == 316:  # Turn right
            leftSpeed = 1.0
            rightSpeed = -1.0
        elif key == 314:  # Turn left
            leftSpeed = -1.0
            rightSpeed = 1.0

        for i in range(4):
            wheels[i].setVelocity(leftSpeed if i % 2 == 0 else rightSpeed)

        gps_data, imu_data, rssi_data = read_sensors(gps, acc, gyro)
        x, P = kalman_filter_update(x, P, gps_data, imu_data, rssi_data, H, R, F, B, Q)
        ground_truth_positions.append(gps_data)
    
        if available_steps > 0:
            noisy_gps = gps_data + np.random.normal(0, 0.05, 3)
            available_steps -= 1
        else:
            if missing_steps == 0:
                # Switch to missing data for 5 to 10 steps randomly
                missing_steps = np.random.randint(5, 11)
            noisy_gps = np.array([np.nan, np.nan, np.nan])
            missing_steps -= 1
            if missing_steps == 0:
                # Switch back to available data
                available_steps = np.random.randint(5, 11)

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

if __name__ == "__main__":
    main()
