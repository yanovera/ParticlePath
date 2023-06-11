import numpy as np
from numpy.linalg import norm
import scipy.stats
import matplotlib.pyplot as plt
# from celluloid import Camera
import copy
from itertools import cycle
import random

from classes import WorldLimits, Point, Particle, Beacon, Agent, OdometerReading, BeaconID
from data import BEACONS_DATA, WAYPOINTS_DATA

WORLD_LIMITS = WorldLimits(x_min=0, y_min=0, x_max=30, y_max=40)
INIT_POS = Point(x=1, y=1)
SIM_TIME = 600
WAYPOINT_TOLERANCE = 0.2
BEACON_RADIUS = 2.0
SPEED = 0.1  # distance unit per time unit

ODOMETER_VAR = 0.1
PROXIMITY_VAR = 0.4


def plot_state(true_trajectory: list[Point], particles: list[Particle], beacons: list[Beacon], map_limits: WorldLimits, estimated_trajectory: list[Point]):
    # Visualizes the state of the particle filter.
    #
    # Displays the particle cloud, mean position and landmarks.

    true_traj_x = []
    true_traj_y = []

    for point in true_trajectory:
        true_traj_x.append(point.x)
        true_traj_y.append(point.y)

    estimated_traj_x = []
    estimated_traj_y = []

    for point in estimated_trajectory:
        estimated_traj_x.append(point.x)
        estimated_traj_y.append(point.y)

    particles_x = []
    particles_y = []

    for particle in particles:
        particles_x.append(particle.x)
        particles_y.append(particle.y)

    # beacon positions
    beacons_x = []
    beacons_y = []
    beacons_id = []

    for beacon in beacons:
        beacons_x.append(beacon.x)
        beacons_y.append(beacon.y)
        beacons_id.append(beacon.id)

    # plot filter state
    plt.clf()
    plt.plot(true_traj_x, true_traj_y, 'g.')
    plt.plot(estimated_traj_x, estimated_traj_y, 'r.')
    plt.plot(particles_x, particles_y, 'm.')
    plt.plot(beacons_x, beacons_y, 'co', markersize=10)
    for beacon in beacons:
        plt.annotate(beacon.id, (beacon.x-0.3, beacon.y-0.6))

    plt.axis(map_limits.to_tuple())

    plt.pause(0.01)


def initialize_particles(num_particles: int, map_limits: WorldLimits) -> list[Particle]:
    # randomly initialize the particles inside the map limits

    particles = []

    for _ in range(num_particles):
        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        x = np.random.uniform(map_limits.x_min, map_limits.x_max)
        y = np.random.uniform(map_limits.y_min, map_limits.y_max)

        particles.append(Particle(x=x, y=y))

    return particles


def move_agent(next_waypoint: Point, agent: Agent, speed: float):
    delta_x = next_waypoint.x - agent.x
    delta_y = next_waypoint.y - agent.y
    distance = agent.distance(next_waypoint)

    agent.x += delta_x / distance * speed
    agent.y += delta_y / distance * speed


def move_particles(particles: list[Particle], odometer_reading: OdometerReading):
    for particle in particles:
        particle.x += odometer_reading.vx
        particle.y += odometer_reading.vy


def eval_sensor_model(sensor_data: dict[BeaconID, (Beacon, float)], particles: list[Particle], noise_variance:float) -> np.array:
    # Computes the observation likelihood of all particles, given the
    # particle and beacons positions and sensor measurements

    weights = []

    scale = np.sqrt(noise_variance)

    for particle in particles:

        prob = 1
        for beacon, distance in sensor_data.values():
            expected_distance = norm((particle.x-beacon.x, particle.y-beacon.y))
            prob_i = scipy.stats.norm.pdf(distance, expected_distance, scale)
            prob = prob * prob_i
        weights.append(prob)

    # normalize weights
    normalizer = sum(weights)
    weights = np.array(weights) / normalizer
    s = sum(weights)

    return weights


def mean_pose(particles) -> Point:
    # calculate the mean pose of a particle set.
    # the mean position is the mean of the particle coordinates

    xs = []
    ys = []

    for particle in particles:
        xs.append(particle.x)
        ys.append(particle.y)

    # calculate average coordinates
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)

    return Point(x=x_mean, y=y_mean)


def resample_particles(particles: list[Particle], weights: np.array):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle weights.

    new_particles = []
    mm = len(particles)
    r = np.random.uniform(0, 1/mm)
    c = weights[0]
    i = 0

    for m in range(mm):
        u = r + m / mm
        while u > c:
            i = i + 1
            c = c + weights[i]
        new_particles.append(particles[i])

    return new_particles


def read_odometer(trajectory: list[Point], noise_variance: float) -> OdometerReading:
    scale = np.sqrt(noise_variance)
    vx = trajectory[-1].x - trajectory[-2].x + np.random.normal(loc=0, scale=scale)
    vy = trajectory[-1].y - trajectory[-2].y + np.random.normal(loc=0, scale=scale)

    return OdometerReading(vx=vx, vy=vy)


def read_sensors(agent: Agent, beacons: list[Beacon], beacon_radius: float, noise_variance: float) -> dict[BeaconID, (Beacon, float)]:
    sensor_reading = {}
    scale = np.sqrt(noise_variance)
    for beacon in beacons:
        distance = agent.distance(beacon)
        distance += np.random.normal(loc=0, scale=scale)
        if distance <= beacon_radius:
            sensor_reading[beacon.id] = (beacon, distance)

    return sensor_reading


def main():

    # add random seed for generating comparable pseudo random numbers
    np.random.seed(123)

    # plot preferences, interactive plotting mode
    # fig = plt.figure()
    plt.title('Particle Filter')
    plt.xlabel('x')
    plt.ylabel('y')
    # camera = Camera(fig)
    plt.axis(WORLD_LIMITS.to_tuple())
    plt.ion()
    plt.show()

    # implementation of a particle filter for robot pose estimation

    # initialize the particles

    particles = initialize_particles(num_particles=1000, map_limits=WORLD_LIMITS)

    waypoints = cycle(WAYPOINTS_DATA)

    next_waypoint = next(waypoints)

    agent = Agent(x=INIT_POS.x, y=INIT_POS.y)

    true_trajectory = [copy.deepcopy(agent)]
    estimated_trajectory = [mean_pose(particles)]

    # run particle filter
    for timestep in range(SIM_TIME):
        # plot the current state
        plot_state(particles=particles, beacons=BEACONS_DATA, map_limits=WORLD_LIMITS, true_trajectory=true_trajectory, estimated_trajectory=estimated_trajectory)
        # camera.snap()

        move_agent(next_waypoint=next_waypoint, agent=agent, speed=SPEED)
        true_trajectory.append(copy.deepcopy(agent))

        if agent.distance(next_waypoint) < WAYPOINT_TOLERANCE:
            next_waypoint = next(waypoints)

        odometer_reading = read_odometer(trajectory=true_trajectory, noise_variance=ODOMETER_VAR)
        sensors_reading = read_sensors(agent=agent, beacons=BEACONS_DATA, beacon_radius=BEACON_RADIUS, noise_variance=PROXIMITY_VAR)

        # predict particles by sampling from motion model with odometry info
        move_particles(particles=particles, odometer_reading=odometer_reading)

        # calculate importance weights according to sensor model
        weights = eval_sensor_model(
            sensor_data=sensors_reading, particles=particles, noise_variance=PROXIMITY_VAR)

        # resample new particle set according to their importance weights
        particles = resample_particles(particles, weights)

        estimated_trajectory.append(mean_pose(particles))

    # save animation as .mp4
    # animation = camera.animate()
    # animation.save('animation.mp4')
    # plt.show('hold')


if __name__ == "__main__":
    main()