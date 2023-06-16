import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import copy
from itertools import cycle

from classes import WorldLimits, Point, Particle, Beacon, Agent, OdometerReading, BeaconID
from data import BEACONS_DATA, WAYPOINTS_DATA

WORLD_LIMITS = WorldLimits(x_min=0, y_min=0, x_max=30, y_max=40)
SIM_STEPS = 300
WAYPOINT_TOLERANCE = 0.2
BEACON_RADIUS = 2.0
SPEED = 1  # distance unit per second
DT = 0.25  # sim time step duration
NUM_PARTICLES = 1000

ODOMETER_VAR = 0.1
PROXIMITY_VAR = 0.4
MOTION_VAR = 0.001
REGULATION_VAR = 0.01


def plot_state(true_trajectory: list[Point], particles: list[Particle], beacons: list[Beacon], map_limits: WorldLimits, estimated_trajectory: list[Point]):
    # Visualizes the state of the particle filter.
    # Displays the particle cloud, beacons, and true vs estimated trajectories

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
    plt.axis('scaled')
    plt.plot(true_traj_x, true_traj_y, 'g.')
    plt.plot(estimated_traj_x, estimated_traj_y, 'r-')
    plt.plot(particles_x, particles_y, 'm.', markersize=0.2)
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


def move_agent(next_waypoint: Point, agent: Agent, speed: float, noise_variance: float):
    scale = np.sqrt(noise_variance)
    delta_x = next_waypoint.x - agent.x
    delta_y = next_waypoint.y - agent.y
    distance = agent.distance(next_waypoint)

    agent.x += DT*(delta_x / distance * speed + np.random.normal(loc=0, scale=scale))
    agent.y += DT*(delta_y / distance * speed + np.random.normal(loc=0, scale=scale))


def move_particles(particles: list[Particle], odometer_reading: OdometerReading, noise_variance: float):
    scale = np.sqrt(noise_variance)
    for particle in particles:
        particle.x += odometer_reading.vx * DT + np.random.normal(loc=0, scale=scale)
        particle.y += odometer_reading.vy * DT + np.random.normal(loc=0, scale=scale)


def eval_weights(sensor_data: dict[BeaconID, (Beacon, float)], particles: list[Particle], noise_variance: float, old_weights: list[float]) -> np.array:
    # Computes the observation likelihood of all particles, given the
    # particle and beacons positions and sensor measurements

    weights = []

    scale = np.sqrt(noise_variance)

    for i, particle in enumerate(particles):
        likelihood = 1.0
        for beacon, distance in sensor_data.values():
            expected_distance = particle.distance(beacon)
            likelihood *= scipy.stats.norm.pdf(distance, expected_distance, scale)
        weights.append(likelihood)  # * old_weights[i])

    # normalize weights
    weights = np.array(weights) / sum(weights)

    return weights


def mean_pose(particles, weights) -> Point:
    # calculates the mean pose of a particle set.

    x_mean = 0
    y_mean = 0
    for i, particle in enumerate(particles):
        x_mean += weights[i] * particle.x
        y_mean += weights[i] * particle.y

    # xs = []
    # ys = []
    # for particle in particles:
    #     xs.append(particle.x)
    #     ys.append(particle.y)
    #
    # # calculate average coordinates
    # x_mean = np.mean(xs)
    # y_mean = np.mean(ys)

    return Point(x=x_mean, y=y_mean)


def resample_particles(particles: list[Particle], weights: np.array):
    # Returns a new set of particles
    new_particles = []
    n = len(particles)
    cdf = np.cumsum(weights)
    i = 0
    for j in range(n):
        starting_point = np.random.uniform(0, 1 / n)
        u = starting_point + j / n  # move along the cdf
        while u > cdf[i]:
            i += 1
        new_particles.append(copy.deepcopy(particles[i]))

    return new_particles


def read_odometer(trajectory: list[Point], noise_variance: float) -> OdometerReading:
    scale = np.sqrt(noise_variance)
    vx = np.random.normal(loc=(trajectory[-1].x - trajectory[-2].x)/DT, scale=scale)
    vy = np.random.normal(loc=(trajectory[-1].y - trajectory[-2].y)/DT, scale=scale)

    return OdometerReading(vx=vx, vy=vy)


def read_sensors(agent: Agent, beacons: list[Beacon], beacon_radius: float, noise_variance: float) -> dict[BeaconID, (Beacon, float)]:
    sensor_reading = {}
    scale = np.sqrt(noise_variance)
    for beacon in beacons:
        distance = np.random.normal(loc=agent.distance(beacon), scale=scale)
        if distance <= beacon_radius:
            sensor_reading[beacon.id] = (beacon, distance)

    return sensor_reading


def main():
    np.random.seed(0)

    plt.title('Particle Filter')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.axis(WORLD_LIMITS.to_tuple())
    plt.ion()
    plt.show()

    # initialize the particles
    particles = initialize_particles(num_particles=NUM_PARTICLES, map_limits=WORLD_LIMITS)

    waypoints = cycle(WAYPOINTS_DATA)

    next_waypoint = next(waypoints)

    agent = Agent(x=next_waypoint.x, y=next_waypoint.y)
    next_waypoint = next(waypoints)

    true_trajectory = [copy.deepcopy(agent)]
    estimated_trajectory = [copy.deepcopy(agent)]

    weights = np.ones(len(particles)) / len(particles)

    # run particle filter
    for timestep in range(SIM_STEPS):
        plot_state(particles=particles, beacons=BEACONS_DATA, map_limits=WORLD_LIMITS, true_trajectory=true_trajectory, estimated_trajectory=estimated_trajectory[6:])

        move_agent(next_waypoint=next_waypoint, agent=agent, speed=SPEED, noise_variance=MOTION_VAR)
        true_trajectory.append(copy.deepcopy(agent))

        if agent.distance(next_waypoint) < WAYPOINT_TOLERANCE:
            next_waypoint = next(waypoints)

        odometer_reading = read_odometer(trajectory=true_trajectory, noise_variance=ODOMETER_VAR)
        sensors_reading = read_sensors(agent=agent, beacons=BEACONS_DATA, beacon_radius=BEACON_RADIUS, noise_variance=PROXIMITY_VAR)

        # predict particles by sampling from motion model with odometry info
        move_particles(particles=particles, odometer_reading=odometer_reading, noise_variance=REGULATION_VAR)

        # calculate importance weights according to sensors readings
        weights = eval_weights(
            sensor_data=sensors_reading, particles=particles, noise_variance=PROXIMITY_VAR, old_weights=weights)

        n_eff = 1 / sum(weights**2)
        if n_eff < len(particles) * 2/3:
            # resample new particle set according to their importance weights
            particles = resample_particles(particles, weights)
            weights = np.ones(len(particles)) / len(particles)

        estimated_trajectory.append(mean_pose(particles, weights))

    plt.show(block=True)


if __name__ == "__main__":
    main()
