"""Particle_Filtering controller."""

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for environments without a display
import matplotlib.pyplot as plt
import numpy as np
from controller import Robot, Motor, Keyboard, GPS, Accelerometer, Gyro
from collections import deque
import logging
import scipy.stats


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

def initialize_particles(num_particles, initial_pos, initial_cov):
    particles = np.random.multivariate_normal(initial_pos, initial_cov, num_particles)
    weights = np.ones(num_particles) / num_particles  # Uniform initial weights
    return particles, weights

def predict_particles(particles, u, dt=1, noise_std=[0.05, 0.01]):
    """ Predict the next state of the particles """
    N = len(particles)
    # Add control noise
    noises = np.random.normal(0, noise_std, (N, 2))
    particles[:, 0] += (u[0] + noises[:, 0]) * np.cos(particles[:, 2]) * dt
    particles[:, 1] += (u[0] + noises[:, 0]) * np.sin(particles[:, 2]) * dt
    particles[:, 2] += (u[1] + noises[:, 1]) * dt
    return particles

def update_weights(particles, weights, measurement, R):
    """ Update the weights of each particle based on measurement likelihood """
    distances = np.linalg.norm(particles[:, :2] - measurement, axis=1)
    weights *= scipy.stats.norm(distances, np.sqrt(R)).pdf(0)
    weights += 1.e-300  # avoid divide by zero
    weights /= np.sum(weights)  # normalize
    return weights

def resample_particles(particles, weights):
    """ Resample particles based on weights """
    indices = np.random.choice(range(len(particles)), size=len(particles), p=weights)
    particles = particles[indices]
    weights.fill(1.0 / len(weights))
    return particles, weights

def rmse(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.sqrt(np.mean((predictions - targets) ** 2))

def mae(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.mean(np.abs(predictions - targets))

def mse(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.mean((predictions - targets) ** 2)

def max_error(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    return np.max(np.abs(predictions - targets))

def effective_sample_size(weights):
    weights = np.array(weights)
    weights /= np.sum(weights)
    return 1.0 / np.sum(np.square(weights))

def simulate_robot_movement(x, u, dt=1):
    """ Simulate the real movement of the robot, which will be considered as ground truth """
    noise_level = 0.05  # Noise level in the control input
    noisy_u = u + np.random.randn(*u.shape) * noise_level
    return np.array([
        x[0] + noisy_u[0] * np.cos(x[2]) * dt,
        x[1] + noisy_u[0] * np.sin(x[2]) * dt,
        x[2] + noisy_u[1] * dt
    ])


def plot_metrics(steps, rmse_values, mae_values, mse_values, max_error_values, neff_values):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))  # Adjust the subplot grid
    axes = axes.flatten()
    metrics = [rmse_values, mae_values, mse_values, max_error_values, neff_values]
    titles = ['PF RMSE', 'PF MAE', 'PF MSE', 'PF Max Error', 'PF Effective Sample Size']
    labels = ['RMSE', 'MAE', 'MSE', 'Max Error', 'Neff']

    for ax, metric, title, label in zip(axes, metrics, titles, labels):
        ax.plot(steps, metric, label=label, marker='o', linestyle='-', color='b')
        ax.set_title(title)
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel(label)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('PF_Complete_Performance_Metrics_Over_Time.png')
    plt.close()


def plot_positions(ground_truth, measured, filtered, particles):
    plt.figure(figsize=(10, 8))

    # Plot Ground Truth
    if ground_truth:
        plt.plot([pos[0] for pos in ground_truth], [pos[1] for pos in ground_truth], 'g-', label='Ground Truth')

    # Plot Measured GPS positions
    if measured:
        plt.plot([pos[0] for pos in measured], [pos[1] for pos in measured], 'r:', label='Measured GPS')

    # Plot Filtered positions from Particle Filter
    if filtered:
        plt.plot([pos[0] for pos in filtered], [pos[1] for pos in filtered], 'b--', label='Filtered GPS')

    # Plot Particles
    if particles.size > 0:  # Check if there are particles to plot
        plt.scatter(particles[:, 0], particles[:, 1], color='c', s=10, alpha=0.3, label='Particles')

    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('GPS Tracking with Particle Filter')
    plt.legend()
    plt.grid(True)
    plt.savefig('GPS_Tracking_with_Particle_Filter.png')
    plt.close()


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
        wheel.setPosition(float('inf'))  # Set wheels to continuous rotation mode
        wheel.setVelocity(0)

    num_particles = 500
    initial_pos = np.zeros(3)  # Starting at the origin
    initial_cov = np.eye(3) * 100  # Large initial uncertainty
    particles, weights = initialize_particles(num_particles, initial_pos, initial_cov)

    path_iterator = follow_path(wheels, path)  # Initialize the path iterator

    ground_truth = []
    measurements = []
    filtered_positions = []

    rmse_values = []
    mae_values = []
    mse_values = []
    max_error_values = []
    neff_values = []  # Store Neff values
    steps = []
    step_count = 0

    # Initialize true_x with the initial position
    true_x = np.copy(initial_pos)  # Make sure to copy to avoid modifying initial_pos inadvertently

    while robot.step(TIME_STEP) != -1:
        try:
            next(path_iterator)  # Move to the next step in the path
        except StopIteration:
            break  # End the simulation if the path is complete

        u = np.array([1.0, 0.1])  # Control inputs
        true_x = simulate_robot_movement(true_x, u)  # Update ground truth based on the last true position
        gps_data = true_x[:2] + np.random.randn(2) * 0.1  # Simulated GPS measurement with noise

        particles = predict_particles(particles, u)
        weights = update_weights(particles, weights, gps_data, 0.1)
        particles, weights = resample_particles(particles, weights)

        x_estimated = np.average(particles, weights=weights, axis=0)
        ground_truth.append(true_x)
        measurements.append(gps_data)
        filtered_positions.append(x_estimated[:2])

        logging.info(f"Step {step_count}: Ground Truth: {true_x}, Particle Filter Estimate: {x_estimated}, GPS Data: {gps_data}")
        print(f"Step {step_count}: Ground Truth: {true_x}, GPS Measurement: {gps_data}, Particle Filter Estimate: {x_estimated}")

        if step_count % 10 == 0 and ground_truth and filtered_positions:
            ground_truth_np = np.array(ground_truth)
            filtered_positions_np = np.array(filtered_positions)
            ground_truth_xy = ground_truth_np[:, :2]
            filtered_xy = filtered_positions_np

            rmse_values.append(rmse(filtered_xy, ground_truth_xy))
            mae_values.append(mae(filtered_xy, ground_truth_xy))
            mse_values.append(mse(filtered_xy, ground_truth_xy))
            max_error_values.append(max_error(filtered_xy, ground_truth_xy))
            neff_values.append(effective_sample_size(weights))
            steps.append(step_count)

        step_count += 1

    if steps:
        plot_metrics(steps, rmse_values, mae_values, mse_values, max_error_values, neff_values)

    plot_positions(ground_truth, measurements, filtered_positions, particles[:, :2])

if __name__ == "__main__":
    main()
