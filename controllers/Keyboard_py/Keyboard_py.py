from controller import Robot, Motor, Keyboard, GPS, InertialUnit, Accelerometer, Gyro
import matplotlib.pyplot as plt
import numpy as np

TIME_STEP = 64

def kalman_filter_update(x, P, gps_data, imu_data, rssi_data):
    # Define the matrices F, B, H_combined, Q, R_combined here 
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
    Q = np.eye(9) * 0.01  # Process noise covariance
    R_combined = np.array([
        [0.0001, 0, 0, 0],
        [0, 0.0001, 0, 0],
        [0, 0, 0.0001, 0],
        [0, 0, 0, 2]
    ])  # Measurement noise covariance

    # Prediction
    u = np.hstack(imu_data)
    x = F @ x + B @ u
    P = F @ P @ F.T + Q

    # Update
    z = np.hstack([gps_data, rssi_data])
    y = z - H_combined @ x  # Measurement residual
    S = H_combined @ P @ H_combined.T + R_combined  # Residual covariance
    K = P @ H_combined.T @ np.linalg.inv(S)  # Kalman gain
    x = x + K @ y
    P = (np.eye(len(x)) - K @ H_combined) @ P

    return x, P

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
    save_path = r"C:\Users\dolai\OneDrive\Documents\mndot_project_simulation_environment\controllers\Keyboard_py\GPS_Tracking_with_Kalman_Filter.png"
    
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

    x = np.zeros(9)  # State vector: [pos, vel, ang_vel]
    P = np.eye(9) * 100  # Initial uncertainty
    gps_noise_std_dev = 0.00005  # Standard deviation for GPS noise

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
        rssi_measurement = np.random.normal(-70, 2)  # Simulated RSSI value

        x, P = kalman_filter_update(x, P, noisy_gps, imu_data, rssi_measurement)

        ground_truth_positions.append(np.array(gps.getValues()))  # Actual position from GPS
        measured_positions.append(noisy_gps)  # Noisy GPS readings
        filtered_positions.append(x[:3])  # Kalman filter output

        # Print the positions
        print("Ground Truth:", np.array(gps.getValues()))
        print("Measured GPS:", noisy_gps)
        print("Filtered GPS:", x[:3])

    # After the loop, plot the positions
    plot_positions(ground_truth_positions, measured_positions, filtered_positions)

if __name__ == "__main__":
    main()
