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
    if key == ord('W'):
        leftSpeed = 1.0
        rightSpeed = 1.0
    elif key == ord('S'):
        leftSpeed = -1.0
        rightSpeed = -1.0
    elif key == ord('D'):
        leftSpeed = 1.0
        rightSpeed = -1.0
    elif key == ord('A'):
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
    if not is_gps_data_valid(gps_data):
        gps_data = np.array([np.nan, np.nan, np.nan])
    rssi_measurement = np.random.normal(-70, 4)
    return gps_data, imu_data, rssi_measurement

def kalman_filter_update(x, P, gps_data, imu_data, rssi_data, H, R, F, B, Q):
    """Perform a Kalman filter update using the provided measurements and noise characteristics."""
    # Assume 'gps_data' includes position measurements and 'imu_data' includes velocity measurements
    if np.isnan(gps_data).any():
        gps_data = x[:3]  # Predicted position if GPS data is missing
        R[0:3, 0:3] *= 10  # Increase measurement uncertainty

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
    
  
    # Kalman filter initialization
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
        print("Updated State:", x)  # Output the updated state

if __name__ == "__main__":
    main()
