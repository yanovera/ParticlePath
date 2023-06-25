from DataTypes import WorldLimits, Variances, Beacon, Point
from ParticleFilter import ParticleFilter
import numpy as np
import matplotlib.pyplot as plt
import csv

WORLD_LIMITS = WorldLimits(x_min=0, y_min=0, x_max=30, y_max=40)
SIM_STEPS = 177
WAYPOINT_TOLERANCE = 0.2
BEACON_RADIUS = 2.0
SPEED = 1.0  # distance unit per time unit
FREQ = 4  # sampling frequency per time unit
SAVE_ANIMATION = False
NUM_PARTICLES = [100, 200, 400, 800]
NUM_RUNS = 100
PLOT_EVERY_RUN = False
SAVE_RESULTS = True

VARIANCES = Variances(odometer=0.1, proximity=0.4, motion=0.001, regulation=0.01)

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
                                num_particles=n))
    mse: dict[int: list[float]] = {}
    se = np.empty((0, SIM_STEPS+1))
    for pf in particle_filters:
        for i in range(NUM_RUNS):
            print(f'performing run #{i+1} of {NUM_RUNS} for N={pf.num_particles}')
            se = np.vstack([se, pf.run(sim_steps=SIM_STEPS, seed=i, plot=PLOT_EVERY_RUN)])
        mse[pf.num_particles] = np.average(se, axis=0)

    fig, ax = plt.subplots(1)

    for key, value in mse.items():
        time_indices = [k/FREQ for k in range(SIM_STEPS+1)]
        ax.semilogy(time_indices, value, label=f'{str(key)} particles')

    ax.legend()

    plt.title(f'MSE over time, {NUM_RUNS} runs')
    plt.xlabel('time')
    plt.ylabel(r'm$^2$')

    plt.show()

    if SAVE_RESULTS:
        save_dict(dict_to_save=mse, filename='results.csv')


def save_dict(dict_to_save: dict, filename: str):
    with open(filename, "w", newline="") as fp:
        # Create a writer object
        writer = csv.DictWriter(fp, fieldnames=dict_to_save.keys())

        # Write the header row
        writer.writeheader()

        # Write the data rows
        writer.writerow(dict_to_save)
        print('Done writing dict to a csv file')


if __name__ == "__main__":
    main()
