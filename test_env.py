import os

from trajectory.env.trajectory_env import *
from trajectory.env.accel_controllers import IDMController, TimeHeadwayFollowerStopper

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from trajectory.env.failsafes import safe_velocity

PLOT_HEATMAPS = False
PLOT_PLOTS = False
LOAD_CONFIG = True

if __name__ == '__main__':
    controller = 'idm'

    custom_cp_path = None

    metrics = defaultdict(list)

    os.makedirs('figs/test_env/', exist_ok=True)

    # config_path = os.path.join('/'.join(custom_cp_path.split('/')[:-2]) ,'env_config.json')
    # if os.path.exists(config_path) and LOAD_CONFIG:
    #     with open(config_path, 'r') as fp:
    #         env_config = json.load(fp)
    if True:
        env_config = DEFAULT_ENV_CONFIG
        env_config.update({
            'horizon': 300,
            'max_headway': 160,
            'use_fs': False,
            'augment_vf': True,
            # how close we need to be at the end to get the reward
            'closing_gap': .85,
            # if we get closer then this time headway we are forced to break with maximum decel
            'minimal_time_headway': 1.0,
            'whole_trajectory': True,
            'num_concat_states': 1,
            'num_idm_cars': 10,
            'minimal_headway': 10.0
        })

    env = TrajectoryEnv(env_config)
    idm = IDMController(a=env.max_accel, b=env.max_decel, noise=0.0)
    fs = TimeHeadwayFollowerStopper(max_accel=env.max_accel, max_deaccel=env.max_decel)

    if custom_cp_path is not None:
        from stable_baselines3 import PPO
        # from stable
        model = PPO.load(custom_cp_path, env=env) #, custom_objects={'policy_class': ActorCriticPolicy})

    state = env.reset()

    s = env.get_base_state()
    fs.v_des = s['leader_speed']

    dmax = 0

    done = False
    total_reward = 0
    total_distance = 0

    times = [0]
    positions = {
        'leader': [env.leader_positions[env.traj_idx]],
        'av': [env.av['pos']],
        **{f'idm_{k}': [env.idm_followers[k]['pos']] for k in range(len(env.idm_followers))}
    }
    speeds = {
        'leader': [env.leader_speeds[env.traj_idx]],
        'av': [env.av['speed']],
        **{f'idm_{k}': [env.idm_followers[k]['speed']] for k in range(len(env.idm_followers))}
    }
    accels = {
        'av': [0],
        **{f'idm_{k}': [0] for k in range(len(env.idm_followers))}
    }
    energy = {
        'av': [0],
        **{f'idm_{k}': [0] for k in range(len(env.idm_followers))}
    }
    distance = {
        'av': [0],
        **{f'idm_{k}': [0] for k in range(len(env.idm_followers))}
    }
    mpg = {
        'av': 0,
        **{f'idm_{k}': 0 for k in range(len(env.idm_followers))}
    }

    low_speed_mpg = {
        'av': 0,
        **{f'idm_{k}': 0 for k in range(len(env.idm_followers))}
    }

    high_speed_mpg = {
        'av': 0,
        **{f'idm_{k}': 0 for k in range(len(env.idm_followers))}
    }



    i = 0
    while not done:
        # if i % 1000 == 0:
        #     print(i)
        i += 1
        if controller == 'idm':
            s = env.get_base_state()
            accel = idm.get_accel(s['speed'], s['leader_speed'], s['headway'], env.time_step)
        elif controller == 'fs_leader':
            s = env.get_base_state()
            dmax = max(dmax, abs(fs.v_des - s['leader_speed']))  # dmax = 1.3933480995153147
            fs.v_des = s['leader_speed']
            accel = fs.get_accel(s['speed'], s['leader_speed'], s['headway'], env.time_step)
        elif controller == 'fs_leader_every':
            s = env.get_base_state()
            if i % 30 == 0:
                fs.v_des =  s['leader_speed']
            accel = fs.get_accel(s['speed'], s['leader_speed'], s['headway'], env.time_step)
        elif controller == 'fs_leader_incremental':
            s = env.get_base_state()
            delta_max = 0.05
            fs.v_des += min(max(s['leader_speed'] - fs.v_des, -delta_max), delta_max)
            accel = fs.get_accel(s['speed'], s['leader_speed'], s['headway'], env.time_step)
        elif controller == 'fs_leader_late':
            s = env.get_base_state()
            delta_max = 1
            fs.v_des = min(max(fs.v_des, s['leader_speed'] - delta_max), s['leader_speed'] + delta_max)
            accel = fs.get_accel(s['speed'], s['leader_speed'], s['headway'], env.time_step)
        elif controller == 'fs_fixed':
            s = env.get_base_state()
            fs.v_des = 50
            accel = fs.get_accel(s['speed'], s['leader_speed'], s['headway'], env.time_step)
        elif controller == 'constant_accel':
            accel = 1.0
        elif controller == 'custom_cp_path':
            accel, _ = model.predict(state, deterministic=True)
        else:
            raise ValueError

        state, reward, done, _ = env.step(accel)
        total_reward += reward

        times.append(times[-1] + env.time_step)

        positions['leader'].append(env.leader_positions[env.traj_idx])
        positions['av'].append(env.av['pos'])
        for k in range(len(env.idm_followers)):
            positions[f'idm_{k}'].append(env.idm_followers[k]['pos'])

        speeds['leader'].append(env.leader_speeds[env.traj_idx])
        speeds['av'].append(env.av['speed'])
        for k in range(len(env.idm_followers)):
            speeds[f'idm_{k}'].append(env.idm_followers[k]['speed'])

        accels['av'].append(env.av['last_accel'])
        for k in range(len(env.idm_followers)):
            accels[f'idm_{k}'].append(env.idm_followers[k]['last_accel'])

        distance['av'].append(speeds['av'][-1] * env.time_step)
        energy['av'].append(env.energy_model.get_instantaneous_fuel_consumption(accels['av'][-1], speeds['av'][-1], grade=0))
        for k in range(len(env.idm_followers)):
            distance[f'idm_{k}'].append(speeds[f'idm_{k}'][-1] * env.time_step)
            energy[f'idm_{k}'].append(env.energy_model.get_instantaneous_fuel_consumption(accels[f'idm_{k}'][-1],
                                                                                      speeds[f'idm_{k}'][-1], grade=0))

    metrics['rewards'].append(total_reward)
    mpg['av'] = (np.sum(distance['av']) / 1609.34) / (np.sum(energy['av']) / 3600 + 1e-6) / env.time_step
    for k in range(len(env.idm_followers)):
        mpg[f'idm_{k}'] = (np.sum(distance[f'idm_{k}']) / 1609.34) / (np.sum(energy[f'idm_{k}']) / 3600 + 1e-6) / env.time_step

    # compute low speed mpg
    low_speeds = np.array(speeds['av']) < 20.0
    high_speeds = np.array(speeds['av']) > 20.0
    low_speed_mpg['av'] = (np.sum(np.array(distance['av'])[low_speeds]) / 1609.34) / \
                          (np.sum(np.array(energy['av'])[low_speeds]) / 3600 + 1e-6) / env.time_step
    for k in range(len(env.idm_followers)):
        low_speed_mpg[f'idm_{k}'] = (np.sum(np.array(distance[f'idm_{k}'])[low_speeds]) / 1609.34) / \
                                    (np.sum(np.array(energy[f'idm_{k}'])[low_speeds]) / 3600 + 1e-6) / env.time_step

    high_speed_mpg['av'] = (np.sum(np.array(distance['av'])[high_speeds]) / 1609.34) / \
                           (np.sum(np.array(energy['av'])[high_speeds]) / 3600 + 1e-6) / env.time_step
    for k in range(len(env.idm_followers)):
        high_speed_mpg[f'idm_{k}'] = (np.sum(np.array(distance[f'idm_{k}'])[high_speeds]) / 1609.34) / \
                                    (np.sum(np.array(energy[f'idm_{k}'])[high_speeds]) / 3600 + 1e-6) / env.time_step


    # compute avg mpg
    avg_rwd = sum(metrics['rewards']) / len(metrics['rewards'])
    avg_mpg = np.sum([val for val in mpg.values()]) / len(mpg)
    avg_mpg_low = np.sum([val for val in low_speed_mpg.values()]) / len(mpg)
    avg_mpg_high = np.sum([val for val in high_speed_mpg.values()]) / len(mpg)

    print(f'Avg rwd ', round(avg_rwd, 2))
    print(f'Avg mpg ', avg_mpg)
    print(f'Avg mpg low ', avg_mpg_low)
    print(f'Avg mpg high ', avg_mpg_high)

    print(f' per car mpg ', mpg)
    print(f' slow mpg ', low_speed_mpg)
    print(f' high speed mpg ', high_speed_mpg)

    N = len(times)

    headways = {
        'leader': [-1] * N,
        'av': [positions['leader'][i] - positions['av'][i] for i in range(N)],
        'idm_0': [positions['av'][i] - positions['idm_0'][i] for i in range(N)],
        **{ f'idm_{k}': [positions[f'idm_{k-1}'][i] - positions[f'idm_{k}'][i] for i in range(N)] for k in range(1, len(env.idm_followers)) },
    }

    headways_diffs = {
        'leader': [-1] * N,
        'av': [-1] * N,
        'idm_0': [headways['av'][i] - headways['idm_0'][i] for i in range(N)],
        **{ f'idm_{k}': [headways[f'idm_{k-1}'][i] - headways[f'idm_{k}'][i] for i in range(N)] for k in range(1, len(env.idm_followers)) },
    }

    max_pos = np.max(positions['leader'])
    relative_positions = {
        k: [positions[k][i] - max_pos * i / N for i in range(N)] for k in positions
    }

    if PLOT_PLOTS:
        for k in range(12):
            start = k * (N // 12)
            end = (k + 1) * (N // 12)

            for label, data in [('position', positions),
                                ('speed', speeds),
                                ('accel', accels),
                                ('headway', headways),
                                ('headways_diff', headways_diffs),
                                ('relative_position', relative_positions),]:
                plt.figure()
                for veh in data:
                    plt.plot(times[start:end], data[veh][start:end], label=veh)
                plt.xlabel('Time (s)')
                plt.ylabel(label)
                plt.legend()
                if controller == 'custom_cp_path':
                    log_name = custom_cp_path.split('/')[-3]
                    save_path = f'figs/test_env/{controller}/{log_name}/{label}'
                else:
                    save_path = f'figs/test_env/{controller}/{label}'
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f'{save_path}/{k}.png')
                plt.close()

    if PLOT_HEATMAPS:
        # also write some accel heatmaps
        for failsafe in [True, False]:
            for ego_speed in range(5, 30, 2):
                lead_speed_diff_range = np.arange(-10, 10.5, 0.5)
                headway_range = np.arange(0, 101, 1)
                lead_speed_diffs, headways = np.meshgrid(lead_speed_diff_range, headway_range)
                accels = np.zeros_like(lead_speed_diffs)
                for i in range(lead_speed_diffs.shape[0]):
                    for j in range(lead_speed_diffs.shape[1]):
                        leader_speed = ego_speed + lead_speed_diffs[i,j]
                        headway = headways[i,j]

                        state_dict = {}
                        for i in range(env.num_concat_states):
                            state_dict.update({'vdes_{}'.format(i): 10.0,
                                               'speed_{}'.format(i): ego_speed,
                                               'leader_speed_{}'.format(i): leader_speed[i, j],
                                               'headway_{}'.format(i): headways[i, j]})
                        state = env.normalize_state(state_dict)
                        if env.augment_vf:
                            augment_vf_shape = int(env.observation_space.low.shape[0] / 2)
                            state = np.concatenate((state, np.zeros(augment_vf_shape)))
                        accel, _ = model.predict(state, deterministic=True)
                        # accel = idm.get_accel(ego_speed, leader_speed, headway, env.time_step)


                        if failsafe:
                            time_step = 0.1
                            max_accel = 1.5
                            max_decel = 3.0
                            v_safe = safe_velocity(ego_speed, leader_speed, headway, max_decel, time_step)
                            v_next = accel * time_step + ego_speed
                            if v_next > v_safe:
                                accel = np.clip((v_safe - ego_speed) / time_step, -np.abs(max_decel), max_accel)

                        accels[-1-i,j] = accel
                extent = np.min(lead_speed_diff_range), np.max(lead_speed_diff_range), np.min(headway_range), np.max(headway_range)
                figure = plt.figure() #figsize=(3,3))
                # figure.tight_layout()
                subplot = figure.add_subplot()
                im = subplot.imshow(accels, extent=extent, cmap=plt.cm.RdBu, interpolation='bilinear', vmin=-3, vmax=1.5)
                extent = im.get_extent()
                subplot.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/1.0)
                figure.colorbar(im, ax=subplot)
                subplot.set_xlabel('Leader speed delta (m/s)')
                subplot.set_ylabel('Headway (m)')
                # figure.tight_layout()
                failsafe_str = 'with failsafe' if failsafe else 'no failsafe'
                subplot.set_title(f'Ego speed {ego_speed}m/s ({failsafe_str})')
                log_name = custom_cp_path.split('/')[-3]
                save_path = f'figs/test_env/{controller}/{log_name}/accel_heatmaps'
                os.makedirs(save_path, exist_ok=True)
                failsafe_str = '_failsafe' if failsafe else ''
                plt.savefig(f'{save_path}/ego_speed_{ego_speed}{failsafe_str}.png')
                plt.close() 