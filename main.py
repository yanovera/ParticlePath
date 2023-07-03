from DataTypes import WorldLimits, Variances, Beacon, Point
from ParticleFilter import ParticleFilter
import numpy as np
import matplotlib.pyplot as plt
import pickle

WORLD_LIMITS = WorldLimits(x_min=0, y_min=0, x_max=30, y_max=40)
SIM_STEPS = 250
WAYPOINT_TOLERANCE = 0.2
BEACON_RADIUS = 2.0
SPEED = 1.0  # distance unit per time unit
FREQ = 4  # sampling frequency per time unit
SAVE_ANIMATION = False
NUM_PARTICLES = [50, 100, 200, 400]
NUM_PARTICLES_GATHERING = [50, 250]
NUM_RUNS = 100
PLOT_EVERY_RUN = False
SAVE_RESULTS = False
READ_RESULTS = False
DICT_READ_FILE = 'results.txt'
DICT_SAVE_FILE = 'results2.txt'
SEED = 0

VARIANCES = Variances(odometer=0.1, proximity=0.4, motion=0.001, regulation=0.001)

BEACONS_DATA = [Beacon(id=1, x=3.5, y=5),
                Beacon(id=2, x=6.5, y=5),
                Beacon(id=3, x=6.8, y=15),
                Beacon(id=4, x=9.5, y=13.5),
                Beacon(id=5, x=13.5, y=20),
                Beacon(id=6, x=16.5, y=20),
                Beacon(id=7, x=21, y=26.5),
                Beacon(id=8, x=23.2, y=25),
                Beacon(id=9, x=23.5, y=35),
                Beacon(id=10, x=26.5, y=35)
                ]

WAYPOINTS_DATA = [Point(x=5, y=3),
                  Point(x=5, y=12),
                  Point(x=25, y=28),
                  Point(x=25, y=37),
                  Point(x=22.5, y=37),
                  Point(x=7.5, y=3)
                  ]


def main():
    particle_filters: list[ParticleFilter] = []
    for n in NUM_PARTICLES:
        particle_filters.append(ParticleFilter(beacon_radius=BEACON_RADIUS,
                                beacons_data=BEACONS_DATA,
                                freq=FREQ,
                                speed=SPEED,
                                variances=VARIANCES,
                                waypoints_data=WAYPOINTS_DATA,
                                waypoint_tolerance=WAYPOINT_TOLERANCE,
                                world_limits=WORLD_LIMITS,
                                num_particles=n,
                                description=f'{n} particles',
                                gather_particles=False))
    for n in NUM_PARTICLES_GATHERING:
        particle_filters.append(ParticleFilter(beacon_radius=BEACON_RADIUS,
                                beacons_data=BEACONS_DATA,
                                freq=FREQ,
                                speed=SPEED,
                                variances=VARIANCES,
                                waypoints_data=WAYPOINTS_DATA,
                                waypoint_tolerance=WAYPOINT_TOLERANCE,
                                world_limits=WORLD_LIMITS,
                                num_particles=n,
                                description=f'{n} particles, with gathering',
                                gather_particles=True))
    mse: dict[int: list[float]] = {}
    if READ_RESULTS:
        mse = read_dict(DICT_READ_FILE)
    for pf in particle_filters:
        se = np.empty((0, SIM_STEPS + 1))
        for i in range(NUM_RUNS):
            print(f'performing run #{i+1} of {NUM_RUNS} for N={pf.num_particles}')
            se = np.vstack([se, pf.run(sim_steps=SIM_STEPS, seed=i+SEED, plot=PLOT_EVERY_RUN)])
        mse[pf.description] = np.average(se, axis=0)

    fig, ax = plt.subplots(1)

    for key, value in mse.items():
        time_indices = [k/FREQ for k in range(SIM_STEPS+1)]
        ax.semilogy(time_indices, value, label=key)

    ax.legend()

    plt.title(f'MSE over time, {NUM_RUNS} runs')
    plt.xlabel('time')
    plt.ylabel(r'm$^2$')

    plt.show()

    if SAVE_RESULTS:
        save_dict(dict_to_save=mse, filename=DICT_SAVE_FILE)


def save_dict(dict_to_save: dict, filename: str):
    file = open(filename, "wb")

    pickle.dump(dict_to_save, file)

    file.close()


def read_dict(filename: str) -> dict:
    with open(filename, "rb") as handle:
        data = handle.read()

    return pickle.loads(data)


if __name__ == "__main__":
    main()
