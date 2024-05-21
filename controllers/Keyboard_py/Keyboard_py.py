from controller import Robot, Keyboard, DistanceSensor, GPS, InertialUnit, Motor

TIME_STEP = 64

# Initialize the Robot
robot = Robot()
kb = Keyboard()
kb.enable(TIME_STEP)

# Initialize Distance Sensors
ds_names = ["ds_right", "ds_left"]
ds = [robot.getDevice(name) for name in ds_names]
for sensor in ds:
    sensor.enable(TIME_STEP)

# Initialize GPS
gp = robot.getDevice("global")
gp.enable(TIME_STEP)

# Initialize Inertial Unit
iu = robot.getDevice("imu")
iu.enable(TIME_STEP)

# Initialize Motors
wheel_names = ["wheel1", "wheel2", "wheel3", "wheel4"]
wheels = [robot.getDevice(name) for name in wheel_names]
for wheel in wheels:
    wheel.setPosition(float('inf'))
    wheel.setVelocity(0.0)

# Main loop
left_speed = 0.0
right_speed = 0.0

while robot.step(TIME_STEP) != -1:
    key = kb.getKey()
    
    if key == Keyboard.UP:
        left_speed = 1.0
        right_speed = 1.0
    elif key == Keyboard.DOWN:
        left_speed = -1.0
        right_speed = -1.0
    elif key == Keyboard.RIGHT:
        left_speed = 1.0
        right_speed = -1.0
    elif key == Keyboard.LEFT:
        left_speed = -1.0
        right_speed = 1.0
    else:
        left_speed = 0.0
        right_speed = 0.0

    # Set wheel velocities
    for i in [0, 2]:  # left wheels
        wheels[i].setVelocity(left_speed)
    for i in [1, 3]:  # right wheels
        wheels[i].setVelocity(right_speed)

    # Print sensor values
    print(f"Right Sensor: {ds[0].getValue()}")
    print(f"Left Sensor: {ds[1].getValue()}")

    # Print GPS values
    gps_values = gp.getValues()
    print(f"X : {gps_values[0]}")
    print(f"Y : {gps_values[1]}")
    print(f"Z : {gps_values[2]}")

    # Print Inertial Unit values
    imu_values = iu.getRollPitchYaw()
    print(f"Angle X : {imu_values[0]}")
    print(f"Angle Y : {imu_values[1]}")
    print(f"Angle Z : {imu_values[2]}")
    print("########################")
