from controller import Robot, Motor, DistanceSensor, Keyboard, GPS, InertialUnit, Accelerometer, Gyro

import numpy as np

TIME_STEP = 64

def kalman_filter_update(x, P, gps_data, imu_data, rssi_data):
    # Define thr matrices F, B, H_combined, Q, R_combined here 
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

def main():
    robot = Robot()

    # Initialize devices using getDevice
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

    # Initialize Kalman filter variables
    x = np.zeros(9)  # State vector: [pos, vel, ang_vel]
    P = np.eye(9) * 100  # Initial uncertainty

    # Define noise parameters
    gps_noise_std_dev = 0.00005  # Standard deviation for GPS noise

    while robot.step(TIME_STEP) != -1:
        # Keyboard control
        key = kb.getKey()
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
        else:
            leftSpeed = 0.0
            rightSpeed = 0.0

        wheels[0].setVelocity(leftSpeed)
        wheels[1].setVelocity(rightSpeed)
        wheels[2].setVelocity(leftSpeed)
        wheels[3].setVelocity(rightSpeed)

        # Get sensor readings
        noisy_gps = np.array(gps.getValues()) + np.random.normal(0, gps_noise_std_dev, 3)
        imu_acc = np.array(acc.getValues())
        imu_gyro = np.array(gyro.getValues())
        imu_data = np.hstack([imu_acc, imu_gyro])
        rssi_measurement = np.random.normal(-70, 2)  # Simulated RSSI value

        # Update Kalman filter
        x, P = kalman_filter_update(x, P, noisy_gps, imu_data, rssi_measurement)

        # Print the filtered position
        print("Filtered Position:", x[:3])

if __name__ == "__main__":
    main()
