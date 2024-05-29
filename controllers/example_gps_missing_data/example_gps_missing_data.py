import matplotlib
matplotlib.use('TkAgg')  # Or another backend appropriate for your system
import matplotlib.pyplot as plt
import numpy as np

from controller import Robot, Motor, Keyboard, GPS, Accelerometer, Gyro

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
    """Read sensor data from Webots devices."""
    gps_data = gps.getValues() 
    acc_data = acc.getValues()
    gyro_data = gyro.getValues()
    imu_data = np.hstack([acc_data, gyro_data])
    rssi_measurement = np.random.normal(-70, 4)  # Simulated RSSI value
    return gps_data, imu_data, rssi_measurement

def kalman_filter_update(x, P, gps_data, imu_data, rssi_data, H, R, F, B, Q):
    """Kalman filter update step with mechanism to use estimated GPS data when actual data is missing."""
    if not is_gps_data_valid(gps_data):
        gps_data = x[:3]  # Use the predicted state as the 'measurement' for GPS data
        R[0:3, 0:3] = np.eye(3) * 100  # Increase uncertainty for GPS data

    u = imu_data
    z = np.hstack([gps_data, rssi_data])

    x_pred = F @ x + B @ u
    P_pred = F @ P @ F.T + Q

    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_updated = x_pred + K @ y
    P_updated = (np.eye(len(x)) - K @ H) @ P_pred

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

    ground_truth_positions = []
    measured_positions = []
    filtered_positions = []

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
        # Simulate that the measured GPS data may sometimes be missing
        if np.random.rand() > 0.2:  # 80% chance to have measured data
            noisy_gps = gps_data + np.random.normal(0, 0.05, 3)
            measured_positions.append(noisy_gps)
            print("Measured GPS:", noisy_gps)
        else:
            measured_positions.append([None, None, None])  # Append None to indicate missing data
            print("Measured GPS: Data Missing")
        filtered_positions.append(x[:3])
    
        print("Ground Truth:", gps_data)
        print("Filtered GPS:", x[:3])

    residuals = calculate_residuals(measured_positions, filtered_positions)
    plot_residuals(residuals)
    plot_positions(np.array(ground_truth_positions), np.array(measured_positions), np.array(filtered_positions))

if __name__ == "__main__":
    main()
