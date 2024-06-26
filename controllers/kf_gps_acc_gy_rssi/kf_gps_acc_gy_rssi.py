import numpy as np
import matplotlib.pyplot as plt
from controller import Robot, Motor, Keyboard, GPS, InertialUnit, Accelerometer, Gyro

TIME_STEP = 64

def kalman_filter_update(x, P, gps_data, imu_data, rssi_data):
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
    H_combined = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    Q = np.eye(9) * 0.1 # Increasing from 0.01 to 0.1 to account for more dynamic unpredictabilit
    R_combined = np.array([
        [9, 0, 0, 0], # Variance for GPS X (3m)^2
        [0, 9, 0, 0], # Variance for GPS Y (3m)^2
        [0, 0, 9, 0], # Variance for GPS Z (3m)^2
        [0, 0, 0, 16] # Variance for RSSI (4 units)^2
    ])

    u = np.hstack(imu_data)
    x = F @ x + B @ u
    P = F @ P @ F.T + Q

    z = np.hstack([gps_data, rssi_data])
    y = z - H_combined @ x
    S = H_combined @ P @ H_combined.T + R_combined
    K = P @ H_combined.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(len(x)) - K @ H_combined) @ P

    return x, P

def calculate_residuals(measured, filtered):
    return np.array(measured) - np.array(filtered)

def plot_residuals(residuals):
    print("Plotting residuals...")
    plt.figure(figsize=(10, 5))
    plt.plot(residuals[:, 0], label='Longitude Residuals')
    plt.plot(residuals[:, 1], label='Latitude Residuals')
    plt.title('Residuals of Kalman Filter')
    plt.xlabel('Time Step')
    plt.ylabel('Residual')
    plt.legend()
    plt.grid(True)
    plt.savefig('C:/Users/dolai/OneDrive/Documents/mndot_project_simulation_environment/controllers/kf_gps_acc_gy_rssi/Residuals_Kalman_Filter.png')
    plt.close()
    print("Residual plot saved.")

def plot_positions(ground_truth, measured, filtered):
    gt = np.array(ground_truth)
    mp = np.array(measured)
    fp = np.array(filtered)

    plt.figure(figsize=(10, 8))
    plt.plot(gt[:, 0], gt[:, 1], 'g-', label='Ground Truth', linewidth=2)  # Green line for ground truth
    plt.plot(mp[:, 0], mp[:, 1], 'ro-', label='Measured GPS', markersize=5, markerfacecolor='red', alpha=0.75, linestyle='--', linewidth=2)  # Red, more visible
    plt.plot(fp[:, 0], fp[:, 1], 'bo-', label='Filtered GPS', markersize=4, markerfacecolor='blue')  # Blue line for filtered GPS
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Tracking with Kalman Filter')
    plt.legend()
    plt.grid(True)
    plt.axis('tight')  # Adjust plot limits to the data range

    # Specify the path and file name for Windows
    save_path = r"C:\Users\dolai\OneDrive\Documents\mndot_project_simulation_environment\controllers\kf_gps_acc_gy_rssi\GPS_Tracking_with_Kalman_Filter.png"
    
    # Save the plot to a file
    plt.savefig(save_path)  # Make sure to use the raw string for the path

    plt.show(block=True)  # Display the plot with blocking enabled




def main():
    robot = Robot()
    gps = robot.getDevice("global")
    gps.enable(TIME_STEP)
    imu = robot.getDevice("imu")
    imu.enable(TIME_STEP)
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
    gps_noise_std_dev = 0.00005

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

        noisy_gps = np.array(gps.getValues()) + np.random.normal(0, gps_noise_std_dev, 3)
        imu_acc = np.array(acc.getValues())
        imu_gyro = np.array(gyro.getValues())
        imu_data = np.hstack([imu_acc, imu_gyro])
        rssi_measurement = np.random.normal(-70, 4)  # Simulated RSSI value

        x, P = kalman_filter_update(x, P, noisy_gps, imu_data, rssi_measurement)

        ground_truth_positions.append(np.array(gps.getValues()))
        measured_positions.append(noisy_gps)
        filtered_positions.append(x[:3])

        print("Ground Truth:", np.array(gps.getValues()))
        print("Measured GPS:", noisy_gps)
        print("Filtered GPS:", x[:3])

    residuals = calculate_residuals(measured_positions, filtered_positions)
    plot_residuals(residuals)
    plot_positions(ground_truth_positions, measured_positions, filtered_positions)

if __name__ == "__main__":
    main()
