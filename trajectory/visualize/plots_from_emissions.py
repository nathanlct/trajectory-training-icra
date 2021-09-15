"""
Generate emissions for all trajectories with IDM

    python simulate.py --av_controller idm --av_kwargs "dict(noise=0)" --all_trajectories --gen_emissions

Generate emissions for all trajectories with an RL controller
    
    python simulate.py --cp_path checkpoints/gamma=0.99_gae_lambda=0.9_env_num_concat_states=1_env_horizon=1000/checkpoints/800.zip --av_controller rl --all_trajectories --gen_emissions

Then use this script to generate metrics and plot stuff

    python visualize/plots_from_emissions.py {path_to_rl_emissions_folder} {path_to_idm_emissions_folder}

eg.

    python visualize/plots_from_emissions.py emissions/23Jun21_15h16m06s/ emissions/23Jun21_15h20m11s/

Each input folder should contain the same number of emissions files, from the same trajectories
and from only one AV controller.

Generated folders:
- idm baseline: emissions/23Jun21_15h16m06s/
- rl controller w/ accel output (cra): emissions/23Jun21_15h20m11s/
- rl controller wrapped in fs (sadifs2): emissions/06Jul21_16h18m30s/
"""
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from plotter import BarPlot
import sys


# load all emissions files
data = defaultdict(dict)
emissions_folders = map(Path, sys.argv[1:])
for path in emissions_folders:
    for csv_path in path.glob('*.csv'):
        split = csv_path.stem.replace('rl_fs', 'rl-fs').split('_')
        av_controller = split[1].replace('rl-fs', 'rl_fs')
        traj_name = '_'.join(split[2:])
        data[traj_name][av_controller] = pd.read_csv(csv_path)

traj2idx = {traj: i for i, traj in enumerate(data.keys())}

# create plotter
now = datetime.now().strftime('%d%b%y_%Hh%Mm%Ss')

plot = BarPlot()


mpgs_by_speed = defaultdict(lambda: defaultdict(list))
steps_by_speed = defaultdict(lambda: defaultdict(list))

for traj_name, dfs in data.items():
    for av_controller, df in dfs.items():
        data_by_vid = dict(tuple(df.groupby(df['id'], as_index=False)))

        traj_id = [vid for vid in data_by_vid.keys() if 'leader' in vid.split('_')][0]
        av_ids = [vid for vid in data_by_vid.keys() if 'av' in vid.split('_')]
        human_ids = [vid for vid in data_by_vid.keys() if 'human' in vid.split('_')]

        # plot traj length
        distance = data_by_vid[traj_id]['total_distance_traveled'].iloc[-1]
        plot.add(top=distance, group=traj2idx[traj_name], label=av_controller, subplot='traj_length')

        # plot traj duration
        duration = data_by_vid[traj_id]['time'].iloc[-1]
        plot.add(top=duration, group=traj2idx[traj_name], label=av_controller, subplot='traj_duration')

        # plot leader speed range
        leader_speeds = data_by_vid[traj_id]['speed']
        plot.add(top=leader_speeds.max(), bottom=leader_speeds.min(), group=traj2idx[traj_name], label=av_controller, subplot='leader_speeds')

        # plot av headway range
        av_headways_min = [data_by_vid[av_id]['headway'].min() for av_id in av_ids]
        av_headways_max = [data_by_vid[av_id]['headway'].max() for av_id in av_ids]
        plot.add(top=max(av_headways_max), bottom=min(av_headways_min), group=traj2idx[traj_name], label=av_controller, subplot='avs_headways')

        # plot av accel range
        av_accels_min = [data_by_vid[av_id]['target_accel_no_noise_with_failsafe'].min() for av_id in av_ids]
        av_accels_max = [data_by_vid[av_id]['target_accel_no_noise_with_failsafe'].max() for av_id in av_ids]
        plot.add(top=max(av_accels_max), bottom=min(av_accels_min), group=traj2idx[traj_name], label=av_controller, subplot='avs_accels')

        # plot avg mpg
        mpg_avs = [data_by_vid[vid]['avg_mpg'].iloc[-1] for vid in av_ids]
        mpg_humans = [data_by_vid[vid]['avg_mpg'].iloc[-1] for vid in human_ids]

        plot.add(top=np.mean(mpg_avs), group=traj2idx[traj_name], label=av_controller, subplot='mpg_avs')
        plot.add(top=np.mean(mpg_humans), group=traj2idx[traj_name], label=av_controller, subplot='mpg_humans')
        plot.add(top=np.mean(mpg_humans + mpg_avs), group=traj2idx[traj_name], label=av_controller, subplot='mpg_avs_humans')

        # plot mpg by speed
        ranges = [0, 10, 20, 30, 40]
        speeds = data_by_vid[traj_id]['speed']
        speeds = speeds.reset_index(drop=True)
        speed_ranges = dict(tuple(speeds.groupby(pd.cut(speeds, ranges, labels=ranges[1:]))))
        speed_idx = {k: v.index for k, v in speed_ranges.items()}
        
        for max_speed in ranges[1:]:
            if len(speed_idx[max_speed]) > 1000:
                mpg_avs = [(data_by_vid[vid]['speed'].iloc[speed_idx[max_speed]].sum() / 1609.34) /
                           (data_by_vid[vid]['instant_energy_consumption'].iloc[speed_idx[max_speed]].sum() / 3600 + 1e-6)
                           for vid in av_ids]
                mpg_humans = [(data_by_vid[vid]['speed'].iloc[speed_idx[max_speed]].sum() / 1609.34) /
                              (data_by_vid[vid]['instant_energy_consumption'].iloc[speed_idx[max_speed]].sum() / 3600 + 1e-6)
                              for vid in human_ids]

                mpgs_by_speed[max_speed][av_controller].append(np.mean(mpg_humans + mpg_avs))
                steps_by_speed[max_speed][av_controller].append(len(speed_idx[max_speed]))

                plot.add(top=np.mean(mpg_humans + mpg_avs), group=traj2idx[traj_name], label=av_controller, subplot=f'mpg_avs_humans_speed_{max_speed-10}_{max_speed}')
            else:
                plot.add(top=0, group=traj2idx[traj_name], label=av_controller, subplot=f'mpg_avs_humans_speed_{max_speed-10}_{max_speed}')

plot.configure_subplot(subplot='traj_length', title='Length of the trajectory', ylabel='Length (m)')
plot.configure_subplot(subplot='traj_duration', title='Duration of the trajectory', ylabel='Duration (s)')

plot.configure_subplot(subplot='leader_speeds', title='Speed range of leader car', ylabel='Speed (m/s)')
plot.configure_subplot(subplot='avs_headways', title='Headway range of AVs', ylabel='Headway (m)')
plot.configure_subplot(subplot='avs_accels', title='Acceleration range of AVs (w/ failsafes)', ylabel='Acceleration (m/s$^2$)')

plot.configure_subplot(subplot='mpg_avs', title='AVs MPG', ylabel='Average MPG')
plot.configure_subplot(subplot='mpg_humans', title='Humans MPG', ylabel='Average MPG')
plot.configure_subplot(subplot='mpg_avs_humans', title='AVs + Humans MPG', ylabel='Average MPG')

for max_speed in [10, 20, 30, 40]:
    plot.configure_subplot(subplot=f'mpg_avs_humans_speed_{max_speed-10}_{max_speed}', title=f'AVs + Humans MPG by speed ({max_speed-10}m/s to {max_speed}m/s)', ylabel='Average MPG')

for max_speed, av_controllers in mpgs_by_speed.items():
    for av_controller, mpgs in av_controllers.items():
        print(f'{av_controller} \t {max_speed-10}m/s to {max_speed}m/s \t avg mpg: {round(np.mean(mpgs), 2)}\t', np.sum(steps_by_speed[max_speed][av_controller]), 'steps')

print()
for max_speed, av_controllers in mpgs_by_speed.items():
    improv = round((np.mean(av_controllers['rl_fs']) / np.mean(av_controllers['idm']) - 1) * 100, 1)
    print(f'{max_speed-10}m/s to {max_speed}m/s \t energy improvement: {improv}%')
