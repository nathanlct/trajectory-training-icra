from matplotlib import colors
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def plot_time_space_diagram(emissions_path, save_path):
    # load emissions
    df = pd.read_csv(emissions_path)

    # compute line segment ends by shifting dataframe by 1 row
    df[['next_pos', 'next_time']] = df.groupby('id')[['position', 'time']].shift(-1)
    # remove nans from data
    df = df[df['next_time'].notna()]
    # generate segments for line collection
    segs = df[['time', 'position', 'next_time', 'next_pos']].values.reshape((len(df), 2, 2))

    # create figure
    fig, ax = plt.subplots()
    ax.set_xlim(df['time'].min(), df['time'].max())
    ax.set_ylim(df['position'].min(), df['position'].max())
    norm = plt.Normalize(df['speed'].min(), df['speed'].max())
    cdict = {
        'red': ((0, 0, 0), (0.2, 1, 1), (0.6, 1, 1), (1, 0, 0)),
        'green': ((0, 0, 0), (0.2, 0, 0), (0.6, 1, 1), (1, 1, 1)),
        'blue': ((0, 0, 0), (0.2, 0, 0), (0.6, 0, 0), (1, 0, 0))
    }
    cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

    # plot line segments
    lc = LineCollection(segs, cmap=cmap) #, norm=norm)
    lc.set_array(df['speed'].values)
    lc.set_linewidth(1)
    ax.add_collection(lc)
    ax.autoscale()

    # add colorbar
    axcb = fig.colorbar(lc)

    # set title and axes labels
    ax.set_title('Time-space diagram on trajectory env')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    axcb.set_label('Velocity (m/s)')

    # save
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == '__main__':
	emissions_path, save_path = sys.argv[1], sys.argv[2]
	plot_time_space_diagram(emissions_path, save_path)
