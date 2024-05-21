from controller import Robot, Motor, DistanceSensor, Keyboard, GPS, InertialUnit, Accelerometer, Gyro

TIME_STEP = 64

def main():
    robot = Robot()
    kb = Keyboard()

    ds_names = ["ds_right", "ds_left"]
    ds = [robot.getDistanceSensor(name) for name in ds_names]
    for sensor in ds:
        sensor.enable(TIME_STEP)

    gp = robot.getGPS("global")
    gp.enable(TIME_STEP)

    iu = robot.getInertialUnit("imu")
    iu.enable(TIME_STEP)

    acc = robot.getAccelerometer("accelerometer")
    acc.enable(TIME_STEP)

    gy = robot.getGyro("gyro")
    gy.enable(TIME_STEP)

    wheels_names = ["wheel1", "wheel2", "wheel3", "wheel4"]
    wheels = [robot.getMotor(name) for name in wheels_names]
    for wheel in wheels:
        wheel.setPosition(float('inf'))
        wheel.setVelocity(0.0)

    kb.enable(TIME_STEP)
    leftSpeed = 0.0
    rightSpeed = 0.0

    while robot.step(TIME_STEP) != -1:
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

        print("X : ", gp.getValues()[0])
        print("Y : ", gp.getValues()[1])
        print("Z : ", gp.getValues()[2])
        
        print("Angle X : ", iu.getRollPitchYaw()[0])
        print("Angle Y : ", iu.getRollPitchYaw()[1])
        print("Angle Z : ", iu.getRollPitchYaw()[2])

        print("Acc X : ", acc.getValues()[0])
        print("Acc Y : ", acc.getValues()[1])
        print("Acc Z : ", acc.getValues()[2])

        print("Gy X : ", gy.getValues()[0])
        print("Gy Y : ", gy.getValues()[1])
        print("Gy Z : ", gy.getValues()[2])
        print("########################")

if __name__ == "__main__":
    main()
