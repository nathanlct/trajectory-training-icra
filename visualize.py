import sys

from trajectory.visualize.plot_energy_model import main as plot_energy_model


if __name__ == '__main__':
    if 'energy_model' in sys.argv:
        plot_energy_model()