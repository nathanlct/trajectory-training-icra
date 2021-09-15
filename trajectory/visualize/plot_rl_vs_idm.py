import pandas as pd
from plotter import Plotter
import time
import numpy as np


rl_path = 'emissions/09Jul21_09h40m25s/emissions.csv'
idm_path = 'emissions/09Jul21_09h44m28s/emissions.csv'


# load data
rl_data = pd.read_csv(rl_path)
idm_data = pd.read_csv(idm_path)

# group data by vid
rl_data = dict(tuple(rl_data.groupby(rl_data['id'], as_index=False)))
idm_data = dict(tuple(idm_data.groupby(idm_data['id'], as_index=False)))

veh_ids = {
    'rl': {
        'traj': [vid for vid in rl_data.keys() if 'leader' in vid][0],
        'av': [vid for vid in rl_data.keys() if 'av' in vid][0],
        'humans': [vid for vid in rl_data.keys() if 'human' in vid],
    }, 
    'idm': {
        'traj': [vid for vid in idm_data.keys() if 'leader' in vid][0],
        'av': [vid for vid in idm_data.keys() if 'av' in vid][0],
        'humans': [vid for vid in idm_data.keys() if 'human' in vid],
    }
}

# traj_id = [vid for vid in data_by_vid.keys() if 'leader' in vid.split('_')][0]
# av_ids = [vid for vid in data_by_vid.keys() if 'av' in vid.split('_')]
# human_ids = [vid for vid in data_by_vid.keys() if 'human' in vid.split('_')]

# plot data
curr_time = time.strftime("%Y%m%d-%H%M%S")
plotter = Plotter(f'figs/rl_vs_idm/{curr_time}/')

with plotter.subplot(title='Trajectory leader speed', xlabel='Time (s)', ylabel='Velocity (m/s)', grid=True, legend=True):
    plotter.plot(rl_data[veh_ids['rl']['traj']]['time'], rl_data[veh_ids['rl']['traj']]['speed'])

with plotter.subplot(title='AV desired speed vs speed (RL FS)', xlabel='Time (s)', ylabel='Velocity (m/s)', grid=True, legend=True):
    av_id = veh_ids['rl']['av']
    plotter.plot(rl_data[av_id]['time'], rl_data[av_id]['speed'], label='Speed')
    plotter.plot(rl_data[av_id]['time'], rl_data[av_id]['vdes'], label='Desired speed (v_des)')

with plotter.subplot(title='AV position', xlabel='Time (s)', ylabel='Position (m)', grid=True, legend=True):
    plotter.plot(rl_data[veh_ids['rl']['av']]['time'], rl_data[veh_ids['rl']['av']]['position'], label='RL FS')
    plotter.plot(idm_data[veh_ids['idm']['av']]['time'], idm_data[veh_ids['idm']['av']]['position'], label='IDM')

with plotter.subplot(title='AV headway', xlabel='Time (s)', ylabel='Headway (m)', grid=True, legend=True):
    plotter.plot(rl_data[veh_ids['rl']['av']]['time'], rl_data[veh_ids['rl']['av']]['headway'], label='RL FS')
    plotter.plot(idm_data[veh_ids['idm']['av']]['time'], idm_data[veh_ids['idm']['av']]['headway'], label='IDM')

with plotter.subplot(title='AV speed', xlabel='Time (s)', ylabel='AcceleSpeedration (m/s)', grid=True, legend=True):
    plotter.plot(rl_data[veh_ids['rl']['av']]['time'], rl_data[veh_ids['rl']['av']]['speed'], label='RL FS')
    plotter.plot(idm_data[veh_ids['idm']['av']]['time'], idm_data[veh_ids['idm']['av']]['speed'], label='IDM')

with plotter.subplot(title='AV accel', xlabel='Time (s)', ylabel='Acceleration (m/s$^2$)', grid=True, legend=True):
    plotter.plot(rl_data[veh_ids['rl']['av']]['time'], rl_data[veh_ids['rl']['av']]['accel'], label='RL FS')
    plotter.plot(idm_data[veh_ids['idm']['av']]['time'], idm_data[veh_ids['idm']['av']]['accel'], label='IDM')

with plotter.subplot(title='AV metrics difference (IDM minus RL FS)', xlabel='Time (s)', ylabel='Difference x100', grid=True, legend=True):
    plotter.plot(idm_data[veh_ids['idm']['av']]['time'], 
        100 * (idm_data[veh_ids['idm']['av']]['speed'] - rl_data[veh_ids['rl']['av']]['speed']), label='Speed')
    plotter.plot(idm_data[veh_ids['idm']['av']]['time'], 
        100 * (idm_data[veh_ids['idm']['av']]['accel'] - rl_data[veh_ids['rl']['av']]['accel']), label='Acceleration')

with plotter.subplot(title='AV instant energy consumption', xlabel='Time (s)', ylabel='Nrj (some interesting unit)', grid=True, legend=True):
    plotter.plot(rl_data[veh_ids['rl']['av']]['time'], rl_data[veh_ids['rl']['av']]['instant_energy_consumption'], label='RL FS')
    plotter.plot(idm_data[veh_ids['idm']['av']]['time'], idm_data[veh_ids['idm']['av']]['instant_energy_consumption'], label='IDM')

with plotter.subplot(title='AV speed difference (leader speed - AV speed)', xlabel='Time (s)', ylabel='Speed (m/s)', grid=True, legend=True):
    plotter.plot(rl_data[veh_ids['rl']['av']]['time'], rl_data[veh_ids['rl']['av']]['speed_difference'], label='RL FS')
    plotter.plot(idm_data[veh_ids['idm']['av']]['time'], idm_data[veh_ids['idm']['av']]['speed_difference'], label='IDM')

with plotter.subplot(title='AV average MPG so far', xlabel='Time (s)', ylabel='MPG', grid=True, legend=True):
    plotter.plot(rl_data[veh_ids['rl']['av']]['time'][1000:], rl_data[veh_ids['rl']['av']]['avg_mpg'][1000:], label='RL FS')
    plotter.plot(idm_data[veh_ids['idm']['av']]['time'][1000:], idm_data[veh_ids['idm']['av']]['avg_mpg'][1000:], label='IDM')

with plotter.subplot(title='Average MPG over the whole trajectory', xlabel='Time (s)', ylabel='MPG', grid=True, legend=True):
    avg_mpg_rl_av = rl_data[veh_ids['rl']['av']]['avg_mpg'].iloc[-1]
    avg_mpg_rl_humans = np.mean([rl_data[human_id]['avg_mpg'].iloc[-1] for human_id in veh_ids['rl']['humans']])
    avg_mpg_idm_av = idm_data[veh_ids['idm']['av']]['avg_mpg'].iloc[-1]
    avg_mpg_idm_humans = np.mean([idm_data[human_id]['avg_mpg'].iloc[-1] for human_id in veh_ids['idm']['humans']])
    plotter.plot([0, 1], [avg_mpg_rl_av] * 2, label=f'RL - AV mpg ({round(avg_mpg_rl_av, 3)})')
    plotter.plot([0, 1], [avg_mpg_rl_humans] * 2, label=f'RL - humans mpg ({round(avg_mpg_rl_humans, 3)})')
    plotter.plot([0, 1], [avg_mpg_idm_av] * 2, label=f'IDM - AV mpg ({round(avg_mpg_idm_av, 3)})')
    plotter.plot([0, 1], [avg_mpg_idm_humans] * 2, label=f'IDM - humans mpg ({round(avg_mpg_idm_humans, 3)})')

with plotter.subplot(title='AV running MPG over the last 100s', xlabel='Time (s)', ylabel='MPG', grid=True, legend=True):
    rl_data_running_sum = rl_data[veh_ids['rl']['av']].rolling(1000).sum()
    rl_mpgs = (rl_data_running_sum['total_miles'] / rl_data_running_sum['total_gallons'] + 1e-6)
    plotter.plot(rl_data[veh_ids['rl']['av']]['time'], rl_mpgs, label='RL FS')

    idm_data_running_sum = idm_data[veh_ids['idm']['av']].rolling(1000).sum()
    idm_mpgs = (idm_data_running_sum['total_miles'] / idm_data_running_sum['total_gallons'] + 1e-6)
    plotter.plot(idm_data[veh_ids['idm']['av']]['time'], idm_mpgs, label='IDM')

with plotter.subplot(title='AV running total miles traveled over the last 100s', xlabel='Time (s)', ylabel='Distance (miles)', grid=True, legend=True):
    rl_data_running_sum = rl_data[veh_ids['rl']['av']].rolling(1000).sum()
    plotter.plot(rl_data[veh_ids['rl']['av']]['time'], rl_data_running_sum['total_miles'], label='RL FS')
    idm_data_running_sum = idm_data[veh_ids['idm']['av']].rolling(1000).sum()
    plotter.plot(idm_data[veh_ids['idm']['av']]['time'], idm_data_running_sum['total_miles'], label='IDM')

with plotter.subplot(title='AV running total gallons consumed over the last 100s', xlabel='Time (s)', ylabel='Energy consumption (gallons)', grid=True, legend=True):
    rl_data_running_sum = rl_data[veh_ids['rl']['av']].rolling(1000).sum()
    plotter.plot(rl_data[veh_ids['rl']['av']]['time'], rl_data_running_sum['total_gallons'], label='RL FS')
    idm_data_running_sum = idm_data[veh_ids['idm']['av']].rolling(1000).sum()
    plotter.plot(idm_data[veh_ids['idm']['av']]['time'], idm_data_running_sum['total_gallons'], label='IDM')

plotter.save('plot', log='\t')