import itertools
import numpy as np
import pandas as pd
from pathlib import Path
import random
import sys
import time
import os
from collections import defaultdict

import trajectory.config as tc
from trajectory.env.simulation import Simulation
from trajectory.env.utils import get_bearing, get_driving_direction, get_valid_lat_long, \
                      lat_long_distance, moving_sum, pairwise, counter
from trajectory.visualize.plotter import Plotter


class DataLoader(object):
    def __init__(self):
        self.trajectories = [{
                'path': fp,
                'timestep': round(data['Time'][1] - data['Time'][0], 3),
                'duration': round(data['Time'].max() - data['Time'].min(), 3),
                'size': len(data['Time']),
                'times': np.array(data['Time']) - data['Time'][0],
                'positions': np.array(data['DistanceGPS']),
                'velocities': np.array(data['Velocity']) / 3.6,
                'accelerations': np.array(data['Acceleration'])
            } for fp, data in self.get_raw_data()]

    def get_raw_data(self):
        file_paths = list(Path(os.path.join(
            tc.PROJECT_PATH, 'dataset/icra_final')).glob('**/*.csv'))
        data = map(pd.read_csv, file_paths)
        return zip(file_paths, data)

    def get_all_trajectories(self):
        random.shuffle(self.trajectories)
        return iter(self.trajectories)

    def get_trajectories(self, chunk_size=None, count=None):
        for _ in counter(count):
            traj = random.sample(self.trajectories, k=1)[0]
            if chunk_size is None:
                yield dict(traj)
            start_idx = random.randint(0, traj['size'] - chunk_size)
            traj_chunk = {
                k: traj[k][start_idx:start_idx+chunk_size]
                for k in ['times', 'positions', 'velocities', 'accelerations']
            }
            traj_chunk.update({
                'path': traj['path'],
                'timestep': traj['timestep'],
                'duration': round(np.max(traj_chunk['times']) - np.min(traj_chunk['times']), 3),
                'size': len(traj_chunk['times']),
            })
            yield traj_chunk


###################################################################################################

def smooth_data(target, n, mu = 0.5):
    for i in range(n):
        target = mu/2*np.append(target[0], target[:-1]) + (1-mu)*target + mu/2*np.append(target[1:], target[-1])
    return target

def _preprocess_data():
    """Preprocess the data in dataset/data_v2 into dataset/data_v2_preprocessed."""
    file_paths = list(Path('../dataset/data_v2').glob('**/*.csv'))
    for fp in file_paths:
        print(f'Reading {fp}')

        # load
        df = pd.read_csv(fp, index_col=0)
        df = df.reset_index(drop=True)
        # sometimes there are missing timesteps
        # fix that by doing linear interpolation on the missing timesteps
        dt = 0.1
        for i in range(1, len(df['Time'])):
            timestep = df['Time'][i] - df['Time'][i-1]
            if abs(timestep - dt) > 1e-3:
                missing = int(round(timestep / dt, 3))
                for k in range(1, missing):
                    idx = round(i - 1 / (k + 1), 3)
                    df.loc[idx] = df.loc[i - 1] + (df.loc[i] - df.loc[i - 1]) * k / missing
                    df.loc[idx]['Time'] = round(df.loc[idx]['Time'], 3)
        df = df.sort_index().reset_index(drop=True)
        assert(all(abs(t1 - t0 - timestep) < 1e-3 for t0, t1 in pairwise(df['Time'])))

        # smooth accelerations
        df['Acceleration'] = smooth_data(np.array(df['Acceleration']), n=50, mu=0.9)

        # compute total ego distances traveled from GPS coordinates
        distances = itertools.accumulate(
            pairwise(zip(df['LatitudeGPS'], df['LongitudeGPS'])),
            func=lambda dist, x: dist + lat_long_distance(*x), 
            initial=0
        )
        df['DistanceGPS'] = list(distances)

        # keep only westbound trajectories, and cut out the on and off ramp parts
        df['MileMarker'] = 74.3 - df['DistanceGPS']

        bears = []
        for i in range(len(df['LatitudeGPS'])-1):
            j = np.deg2rad(df['LongitudeGPS'][i])
            k = np.deg2rad(df['LongitudeGPS'][i+1])
            m = np.deg2rad(df['LatitudeGPS'][i])
            n = np.deg2rad(df['LatitudeGPS'][i+1])
            bears.append(get_bearing(j,m,k,n))
        bears.append(0)
        df['Bearing'] = bears

        directions = []
        for i in range(len(df['Bearing'])):
            directions.append(get_driving_direction(df['Bearing'][i]))
        df['DrivingDir'] = directions
        
        valid_points = []
        for i in range(len(df['LatitudeGPS'])):
            lat = df['LatitudeGPS'][i]
            long = df['LongitudeGPS'][i]
            valid_points.append(get_valid_lat_long(lat,long))
        df['ValidPt'] = valid_points

        switches = np.abs(np.diff(df['ValidPt']))
        switching_points = np.argwhere(switches==1)
        num_switches = len(switching_points)
        
        i = 0
        k = 0
        while i < num_switches:
            #Find points at which it switches out of valid space:
            start_point = switching_points[i][0]
            end_point = switching_points[i+1][0]
            
            #get the direction it's going in:
            direction = df['DrivingDir'][start_point]

            if direction == 'West':
                # extract sub-trajectory
                sub_df = df.iloc[start_point:end_point]
            
                # save trajectory
                file_path = str(fp).replace('data_v2', 'data_v2_preprocessed')
                file_path = file_path.replace('.csv', f'_{k}_{end_point - start_point}.csv')
                sub_df.to_csv(file_path, encoding='utf-8', index=True, index_label='index')
                print(f'Wrote {file_path}')

                k += 1

            i += 2


if __name__ == '__main__':
    data_loader = DataLoader()
    curr_time = time.strftime("%Y%m%d-%H%M%S")

    if 'preprocess' in sys.argv:
        print('Preprocessing trajectory files')
        _preprocess_data()

    if 'plot_raw' in sys.argv:  # plot all columns in the CSVs
        plotter = Plotter('figs/dataset/raw')
        print('Plotting raw trajectory data')
        for fp, data in data_loader.get_raw_data():
            for key in data:
                plotter.plot(data[key], title=key, grid=True)
            plotter.save(fp.stem, log='\t')

    if 'plot_data' in sys.argv:  # plot ego trajectories (positions, speeds, accels)
        print('Plotting processed trajectory data')
        plotter = Plotter(f'figs/dataset/processed/{curr_time}')
        for traj in data_loader.get_all_trajectories():
            plotter.plot(traj['times'], traj['positions'], title='Ego positions', 
                xlabel='time (s)', ylabel='position (m)', grid=True)
            plotter.plot(traj['times'], traj['velocities'], title='Ego velocities', 
                xlabel='time (s)', ylabel='speed (m/s)', grid=True)
            plotter.plot(traj['times'], traj['accelerations'], title='Ego accelerations', 
                xlabel='time (s)', ylabel='accel (m/$s^2$)', grid=True)
            plotter.save(traj['path'].stem, log='\t')

    if 'plot_sims' in sys.argv:  # plot mpg values for the trajectories, using different types of platoons
        print('Computing MPG values for different platoons following the trajectories')
        # Contains data from each trajectory
        all_mpg_params = []
        for traj in data_loader.get_all_trajectories():
            # desired constraints: s0 > 7, noise = 0, 10 < headway < 150, T > 0.4
            all_av_params = []
            # Add IDM search params
            if 'grid_search' in sys.argv:
                a_sweep = np.around(np.arange(0.5, 2.4, 0.1), decimals=1)
                b_sweep = np.around(np.arange(1.3, 3.0, 0.1), decimals=1)
                # a_sweep = np.around(np.arange(0.8, 1.0, 0.1), decimals=1)
                # b_sweep = np.around(np.arange(1.6, 1.8, 0.1), decimals=1)
                for a in a_sweep:
                    for b in b_sweep:
                        all_av_params.append((5, 'idm', dict(v0=35, T=1, a=a, b=b, delta=4, s0=7, noise=0)))

                v_des_sweep = np.arange(5, 21)
                # v_des_sweep = np.arange(5, 7)
                for v in v_des_sweep:
                    all_av_params.append((5, 'fs', dict(v_des=v)))
            else:
                all_av_params.append((5, 'idm', dict(v0=35, T=1, a=1.3, b=2.0, delta=4, s0=7, noise=0)))

            # For each trajectory, stores key-values of car params-MPG by controller
            mpg_params = defaultdict(dict)
            fs_plotter = Plotter('figs/dataset/', curr_time)
            for num_idms, controller, av_params in all_av_params:
                plotter = Plotter(f'figs/dataset/{curr_time}/', traj['path'].stem)
                print(traj['path'])
                sim = Simulation(timestep=traj['timestep'])
                sim.add_vehicle(controller='trajectory',
                    trajectory=zip(traj['positions'], traj['velocities'], traj['accelerations']))
                sim.add_vehicle(controller=controller, gap=20, **av_params)
                for _ in range(num_idms):
                    sim.add_vehicle(controller='idm', gap=20, **dict(v0=35, T=1, a=1.3, b=2.0, delta=4, s0=2, noise=0.3))
                sim.run()

                # plot
                with plotter.subplot(title='Positions', xlabel='Time (s)', ylabel='Position (m)', grid=True, legend=True):
                    for vid, vdata in sim.data_by_vehicle.items():
                        plotter.plot(vdata['time'], vdata['position'], label=vid)
                with plotter.subplot(title='Velocities', xlabel='Time (s)', ylabel='Speed (m/s)', grid=True, legend=True):
                    for vid, vdata in sim.data_by_vehicle.items():
                        plotter.plot(vdata['time'], vdata['speed'], label=vid)
                with plotter.subplot(title='Accelerations', xlabel='Time (s)', ylabel='Accel (m/s$^2$)', grid=True, legend=True):
                    for vid, vdata in sim.data_by_vehicle.items():
                        plotter.plot(vdata['time'], vdata['accel'], label=vid)
                with plotter.subplot(title='Bumper-to-bumper gaps (to leader)', xlabel='Time (s)', ylabel='Gap (m)', grid=True, legend=True):
                    for vid, vdata in sim.data_by_vehicle.items():
                        plotter.plot(vdata['time'], vdata['headway'], label=vid)
                with plotter.subplot(title='Velocity differences (to leader)', xlabel='Time (s)', ylabel='Speed diff (m/s)', grid=True, legend=True):
                    for vid, vdata in sim.data_by_vehicle.items():
                        plotter.plot(vdata['time'], vdata['speed_difference'], label=vid)
                with plotter.subplot(title='Running MPGs over the last 100s', xlabel='Time (s)', ylabel='MPG average', grid=True, legend=True):
                    for vid, vdata in sim.data_by_vehicle.items():
                        speeds = moving_sum(vdata['speed'], chunk_size=1000)
                        energies = moving_sum(vdata['instant_energy_consumption'], chunk_size=1000)
                        mpgs = (speeds / 1609.34) / (energies / 3600 + 1e-6)
                        plotter.plot(vdata['time'][999:], mpgs, label=vid)
                with plotter.subplot(title='MPGs (average doesn\'t account for trajectory vehicles)', xlabel='Time (s)', ylabel='Miles per gallon', grid=True, legend=True):
                    mpgs = []
                    for vid, vdata in sim.data_by_vehicle.items():
                        mpg = (np.sum(vdata['speed']) / 1609.34) / (np.sum(vdata['instant_energy_consumption']) / 3600 + 1e-6)
                        plotter.plot([0, 1], [mpg] * 2, label=f'{vid} ({round(mpg, 2)})')
                        if 'trajectory' not in vid:
                            mpgs.append(mpg)
                    avg_mpg = np.mean(mpgs)
                    plotter.plot([0, 1], [avg_mpg] * 2, label=f'average ({round(avg_mpg, 2)})', linewidth=3.0)
                
                mpg_params[controller][str(av_params)] = avg_mpg

                params_str = '_'.join([f'{k}={v}' for k, v in av_params.items()])
                plotter.save(f'platoon_{num_idms}{controller}_{params_str}_{round(avg_mpg, 2)}mpg', log='\t')
            
            if 'idm' in mpg_params:
                heat = np.empty((0,len(b_sweep)), int)
                for j in np.arange(len(a_sweep) * len(b_sweep), 0, -len(b_sweep)):
                    heat = np.append(heat, np.array([list(mpg_params['idm'].values())[j-len(b_sweep): j]]), axis=0)

                y_ticks = range(0, len(a_sweep)) if len(a_sweep) % 2 == 0 else range(0, len(a_sweep))
                x_ticks = range(0, len(b_sweep))
                yticklabels = np.around(a_sweep, decimals=1)[::-1]
                xticklabels = np.around(b_sweep, decimals=1)
    
                plotter.heatmap(heat, xlabel='', ylabel='', xticks=x_ticks, yticks=y_ticks,
                                xticklabels= xticklabels, yticklabels=yticklabels,
                                title="Heatmap of IDM a and b parameters", cbarlabel='MPG')

            all_mpg_params.append(mpg_params)
            score = dict()
            for elem in list(mpg_params.values()):
                score.update(elem)
            key = max(score, key=lambda k: score.get(k))
            print(f'Top score is {score[key]} mpg with params: {str(key)}')
            print("All results:")
            for k, v in score.items():
                print(k, v)

        with fs_plotter.subplot(title='MPGs over v_des sweep', xlabel='v_des (m/s)', ylabel='Miles per gallon', grid=True, legend=True):
            for traj in all_mpg_params:
                for controller, m_param in traj.items():
                    if controller == 'fs':
                        fs_plotter.plot([elem['v_des'] for elem in [eval(x) for x in list(m_param.keys())]], list(m_param.values()))

            fs_plotter.save('v_des_sweep')
        

    if 'small_chunks' in sys.argv:  # compute different metrics on a lot of small chunks of trajectories
        print('Running simulations with small chunks of trajectories')
        
        low_speed_mpgs = []
        high_speed_mpgs = []
        for i, traj in enumerate(data_loader.get_trajectories(chunk_size=600, count=100)):
            print(i)
            idm_params = dict(v0=35, T=1, a=1.3, b=2.0, delta=4, s0=2, noise=0.3)
            sim = Simulation(timestep=traj['timestep'])
            sim.add_vehicle(controller='trajectory',
                trajectory=zip(traj['positions'], traj['velocities'], traj['accelerations']))
            for _ in range(5):
                sim.add_vehicle(controller='idm', gap=20, **idm_params)
            sim.run()

            mpgs = [(np.sum(vdata['speeds']) / 1609.34) / (np.sum(vdata['instant_energy_consumptions']) / 3600 + 1e-6)
                    for vid, vdata in sim.data_by_vehicle.items()
                    if 'trajectory' not in vid]
            avg_speed = np.mean(sim.data_by_vehicle['0_trajectory']['speeds'])
            if avg_speed > 18:
                high_speed_mpgs += mpgs
            else:
                low_speed_mpgs += mpgs
        print(f'\tLow speed average MPG: {np.mean(low_speed_mpgs)}')
        print(f'\tHigh speed average MPG: {np.mean(high_speed_mpgs)}')
