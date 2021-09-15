import matplotlib.pyplot as plt
import numpy as np

from trajectory.env.energy_models import PFM2019RAV4


def main():
    speeds = np.arange(0, 30, 0.1)
    accels = np.arange(-3, 1.5, 0.05)

    energy_model = PFM2019RAV4()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X, Y = np.meshgrid(speeds, accels)
    Z = np.zeros_like(X)
    for idx, _ in np.ndenumerate(Z):
        Z[idx] = energy_model.get_instantaneous_fuel_consumption(Y[idx], X[idx], grade=0)

    surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()