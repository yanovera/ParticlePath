import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import copy
from itertools import cycle
from celluloid import Camera
from typing import Optional

from DataTypes import WorldLimits, Point, Particle, Beacon, Agent, OdometerReading, BeaconID, Variances


class ParticleFilter:
    def __init__(self, world_limits: WorldLimits, beacons_data: list[Beacon], waypoints_data: list[Point], freq: int, variances: Variances, waypoint_tolerance: float, beacon_radius: float, num_particles: int, description: str, gather_particles: bool=False):
        self.fig = plt.figure()
        self.world_limits = world_limits
        self.beacons = beacons_data
        self.variances = variances
        self.waypoint_tolerance = waypoint_tolerance
        self.beacon_radius = beacon_radius
        self.num_particles = num_particles
        self.gather_particles = gather_particles
        self.description = description

        self.waypoints_data = waypoints_data

        self.dt = 1 / freq

    def run(self, sim_steps: int, plot: bool = False, seed: int = 0, animation_filename: Optional[str] = None) -> list[float]:
        np.random.seed(seed)
        waypoints = cycle(self.waypoints_data)
        if plot or animation_filename is not None:
            plt.title(f'{self.num_particles} Particles Filter, #{seed}')
            plt.xlabel('x')
            plt.ylabel('y')
            camera = Camera(self.fig)
            plt.axis(self.world_limits.to_tuple())
            plt.ion()
            plt.show()

        # initialize the particles
        particles = self.initialize_particles()

        next_waypoint = next(waypoints)
        agent = Agent(x=next_waypoint.x, y=next_waypoint.y)
        next_waypoint = next(waypoints)

        true_trajectory = [copy.deepcopy(agent)]
        estimated_trajectory = [copy.deepcopy(agent)]

        weights = np.ones(len(particles)) / len(particles)

        # run particle filter
        for timestep in range(sim_steps):
            if animation_filename is not None:
                camera.snap()
            elif plot:
                plt.clf()

            if plot:
                self.plot_state(particles=particles,
                                true_trajectory=true_trajectory,
                                estimated_trajectory=estimated_trajectory[3:])

            self.move_agent(next_waypoint=next_waypoint,
                            agent=agent)
            true_trajectory.append(copy.deepcopy(agent))

            if agent.distance(next_waypoint) < self.waypoint_tolerance:
                next_waypoint = next(waypoints)

            odometer_reading = self.read_odometer(trajectory=true_trajectory)
            sensors_reading = self.read_sensors(agent=agent)

            # predict particles by sampling from motion model with odometry info
            self.move_particles(particles=particles, odometer_reading=odometer_reading)
            if self.gather_particles:
                for beacon_data in sensors_reading.values():
                    self.move_particles_to_circle(particles=particles, center=beacon_data[0], radius=self.beacon_radius*2)

            # calculate importance weights according to sensors readings
            weights = self.eval_weights(sensor_data=sensors_reading,
                                        particles=particles,
                                        old_weights=weights)

            n_eff = 1 / sum(weights ** 2)
            if n_eff < len(particles) * 2 / 3:
                # resample new particle set according to their importance weights
                particles = self.resample_particles(particles, weights)
                weights = np.ones(len(particles)) / len(particles)

            estimated_trajectory.append(self.mean_pose(particles, weights))

        if animation_filename is not None:
            # save animation as .mp4
            camera.snap()
            animation = camera.animate()
            animation.save(animation_filename+'.mp4')
        elif plot:
            plt.show(block=True)

        squared_errors = [true_point.distance(estimated_point)**2 for true_point, estimated_point in zip(true_trajectory, estimated_trajectory)]
        return squared_errors

    def initialize_particles(self) -> list[Particle]:
        # randomly initialize the particles inside the map limits

        particles = []

        for _ in range(self.num_particles):
            # draw x,y and theta coordinate from uniform distribution
            # inside map limits
            x = np.random.uniform(self.world_limits.x_min, self.world_limits.x_max)
            y = np.random.uniform(self.world_limits.y_min, self.world_limits.y_max)

            particles.append(Particle(x=x, y=y))

        return particles

    def plot_state(self, true_trajectory: list[Point], particles: list[Particle],
                   estimated_trajectory: list[Point]):
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

        for beacon in self.beacons:
            beacons_x.append(beacon.x)
            beacons_y.append(beacon.y)
            beacons_id.append(beacon.id)

        # plot filter state
        plt.axis('scaled')
        plt.plot(true_traj_x, true_traj_y, 'g.')
        plt.plot(estimated_traj_x, estimated_traj_y, 'r-')
        plt.plot(particles_x, particles_y, 'm.', markersize=0.2)
        plt.plot(beacons_x, beacons_y, 'co', markersize=10)
        for beacon in self.beacons:
            plt.annotate(beacon.id, (beacon.x - 0.3, beacon.y - 0.6))

        plt.axis(self.world_limits.to_tuple())

        plt.pause(0.101)

    def move_agent(self, next_waypoint: Point, agent: Agent):
        scale = np.sqrt(self.variances.motion)
        delta_x = next_waypoint.x - agent.x
        delta_y = next_waypoint.y - agent.y
        distance = agent.distance(next_waypoint)

        agent.x += self.dt*(delta_x / distance + np.random.normal(loc=0, scale=scale))
        agent.y += self.dt*(delta_y / distance + np.random.normal(loc=0, scale=scale))

    def read_odometer(self, trajectory: list[Point]) -> OdometerReading:
        scale = np.sqrt(self.variances.odometer)
        vx = np.random.normal(loc=(trajectory[-1].x - trajectory[-2].x) / self.dt, scale=scale)
        vy = np.random.normal(loc=(trajectory[-1].y - trajectory[-2].y) / self.dt, scale=scale)

        return OdometerReading(vx=vx, vy=vy)

    def read_sensors(self, agent: Agent) -> dict[BeaconID, (Beacon, float)]:
        sensor_reading = {}
        scale = np.sqrt(self.variances.proximity)
        for beacon in self.beacons:
            distance = np.random.normal(loc=agent.distance(beacon), scale=scale)
            if distance <= self.beacon_radius:
                sensor_reading[beacon.id] = (beacon, distance)

        return sensor_reading

    def move_particles(self, particles: list[Particle], odometer_reading: OdometerReading):
        scale = np.sqrt(self.variances.regulation)
        for particle in particles:
            particle.x += odometer_reading.vx * self.dt + np.random.normal(loc=0, scale=scale)
            particle.y += odometer_reading.vy * self.dt + np.random.normal(loc=0, scale=scale)

    def eval_weights(self, sensor_data: dict[BeaconID, (Beacon, float)], particles: list[Particle],
                     old_weights: list[float]) -> np.array:
        # Computes the observation likelihood of all particles, given the
        # particle and beacons positions and sensor measurements

        weights = []

        scale = np.sqrt(self.variances.proximity)

        for i, particle in enumerate(particles):
            likelihood = 1.0
            for beacon, distance in sensor_data.values():
                expected_distance = particle.distance(beacon)
                likelihood *= scipy.stats.norm.pdf(distance, expected_distance, scale)
            weights.append(likelihood * old_weights[i])

        # normalize weights
        normalizer = sum(weights)
        if normalizer > 0:
            weights = np.array(weights) / normalizer
        else:
            weights = np.ones(len(particles)) / len(particles)

        return weights

    @staticmethod
    def mean_pose(particles, weights) -> Point:
        # calculates the mean pose of a particle set.

        x_mean = 0
        y_mean = 0
        for i, particle in enumerate(particles):
            x_mean += weights[i] * particle.x
            y_mean += weights[i] * particle.y

        return Point(x=x_mean, y=y_mean)

    @staticmethod
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

    @staticmethod
    def move_particles_to_circle(particles: list[Particle], center: Point, radius: float):
        for particle in particles:
            dx = particle.x - center.x
            dy = particle.y - center.y
            distance = np.linalg.norm([dx, dy])
            if distance > radius:
                ratio = distance / radius
                particle.x = dx / ratio + center.x
                particle.y = dy / ratio + center.y

