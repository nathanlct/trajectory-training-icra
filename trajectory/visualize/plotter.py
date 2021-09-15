from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path # needed?
from stable_baselines3.common.logger import Figure


class Plotter(object):
    def __init__(self, *save_dir):
        self.save_dir = Path(*save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.plot_data = []

        self.subplots = False

    def subplot(self, title=None, xlabel=None, ylabel=None, grid=False, legend=False):
        self.subplots = True
        self.plot_data.append({
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'grid': grid,
            'legend': legend,
            'plots': [],
        })
        return self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, tb):
        self.subplots = False

    def plot(self, x, y=None, label=None, title=None, xlabel=None, ylabel=None, grid=False, legend=False, linewidth=1.0):
        if y is None:
            x, y = list(range(len(x))), x
        if self.subplots:
            self.plot_data[-1]['plots'].append({
                'x': x,
                'y': y,
                'label': label,
                'linewidth': linewidth,
            })
        else:
            self.plot_data.append({
                'title': title,
                'xlabel': xlabel,
                'ylabel': ylabel,
                'grid': grid,
                'legend': legend,
                'plots': [{
                    'x': x,
                    'y': y,
                    'label': label,
                    'linewidth': linewidth,
                }],
            })
    
    def heatmap (self, array, xlabel='', ylabel='', xticks=None, yticks=None, xticklabels=None, yticklabels=None,
                 title=None, file_name='heatmap', cbarlabel=None, cbar_kw={}):
        # Integrate it in with plot_data
        fig, ax = plt.subplots()
        im = ax.imshow(array)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)

        # ... and label them with the respective list entries
        if xlabel is not None:
            ax.set_xticklabels(xticklabels)
        if ylabel is not None:
            ax.set_yticklabels(yticklabels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom") 

        ax.set_title(title)
        fig.tight_layout()
        save_path = self.save_dir / (file_name + '.png')
        plt.savefig(save_path)

    def makefig(self, dpi=100):
        # figsize in inches, dpi = dots (pixels) per inches
        fig, axes = plt.subplots(len(self.plot_data), 
            figsize=(15, 2 * len(self.plot_data)), dpi=dpi)
        if len(self.plot_data) == 1:
            axes = [axes]
        for ax, data in zip(axes, self.plot_data):
            for plot in data['plots']:
                ax.plot(plot['x'], plot['y'], label=plot['label'], linewidth=plot['linewidth'])
            ax.set_title(data['title'])
            ax.set_xlabel(data['xlabel'])
            ax.set_ylabel(data['ylabel'])
            ax.set_xlim(np.min([plot['x'] for plot in data['plots']]), 
                        np.max([plot['x'] for plot in data['plots']]))
            if data['grid']: ax.grid()
            if data['legend']: ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        return fig

    def save(self, file_name, log=None):
        fig = self.makefig()
        save_path = self.save_dir / (file_name + '.png')
        fig.savefig(save_path)
        plt.close(fig)
        self.plot_data.clear()

        if log:
            print(f'{log if type(log) is str else ""}Written {save_path}')


class TensorboardPlotter(Plotter):
    def __init__(self, logger):
        self.logger = logger
        self.plot_data = []
        self.subplots = False

    def save(self, log_name):
        fig = self.makefig(dpi=100)
        self.logger.record(log_name, Figure(fig, close=True), exclude=('stdout', 'log', 'json', 'csv'))
        plt.close(fig)
        self.plot_data.clear()


class BarPlot(object):
    def __init__(self):
        self.plot_data = defaultdict(lambda: defaultdict(dict))
        self.labels = set()
        self.groups = set()
        self.subplot_metadata = dict()

    def add(self, top, bottom=0, group='group', label='label', subplot=0):
        self.plot_data[subplot][group][label] = (round(top - bottom, 2), round(bottom, 2))
        self.groups.add(group)
        self.labels.add(label)

    def configure_subplot(self, subplot, title, ylabel):
        self.subplot_metadata[subplot] = {
            'title': title,
            'ylabel': ylabel,
        }
        
    def save(self, *save_path):
        fig, axes = plt.subplots(len(self.plot_data), figsize=(len(list(self.groups)), 6*len(self.plot_data)))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for i, (subplot, data) in enumerate(self.plot_data.items()):
            ax = axes[i]

            groups = sorted(list(self.groups))
            x = np.arange(len(groups))
            bar_width = 0.35
            for i, label in enumerate(sorted(list(self.labels))):
                tops, bottoms = list(zip(*[data[group][label] for group in groups]))
                rects = ax.bar(x - bar_width * i, tops, bar_width, label=label, align='edge', bottom=bottoms)
                ax.bar_label(rects, padding=3, label_type='edge')

            if subplot in self.subplot_metadata:
                ax.set_ylabel(self.subplot_metadata[subplot]['ylabel'])
                ax.set_title(self.subplot_metadata[subplot]['title'])

            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            ax.legend()

        # create save dir if it doesn't exist
        save_path = Path(*save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # save figure
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        print(f'Wrote {save_path}')

