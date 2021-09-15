from collections import defaultdict
from datetime import datetime
import importlib
import json
import numpy as np
from pathlib import Path
import os
import re

import trajectory.config as tc
from args import parse_args_simulate
from trajectory.callbacks import TensorboardCallback
from trajectory.env.trajectory_env import DEFAULT_ENV_CONFIG, TrajectoryEnv
from trajectory.env.utils import get_first_element
from trajectory.visualize.plotter import Plotter


# parse command line arguments
args = parse_args_simulate()

# load AV controller
if 'rl' in args.av_controller.lower():
    # load config file
    cp_path = Path(args.cp_path)
    with open(cp_path.parent.parent / 'configs.json', 'r') as fp:
        configs = json.load(fp)
    env_config = DEFAULT_ENV_CONFIG
    env_config.update(configs['env_config'])

    if args.av_controller == 'rl_fs':
        env_config['av_controller'] = 'rl_fs'


    # retrieve algorithm
    alg_module, alg_class = re.match("<class '(.+)\.([a-zA-Z\_]+)'>", configs['algorithm']).group(1, 2)
    assert (alg_module.split('.')[0] in ['stable_baselines3', 'algos'])
    algorithm = getattr(importlib.import_module(alg_module), alg_class)

    # load checkpoint into model
    model = algorithm.load(cp_path)

    print(f'\nLoaded model checkpoint at {cp_path}\n')
    if args.verbose:
        print(f'trained for {model.num_timesteps} timesteps')
        print(f'algorithm = {alg_module}.{alg_class}')
        print(f'observation space = {model.observation_space}')
        print(f'action space = {model.action_space}')
        print(f'policy = {model.policy_class}')
        print(f'\n{model.policy}\n')

    get_action = lambda state: model.predict(state, deterministic=True)[0]

else:
    env_config = DEFAULT_ENV_CONFIG
    env_config.update({
        'use_fs': False,
        'discrete': False,
        'av_controller': args.av_controller,
        'av_kwargs': args.av_kwargs,
        'human_controller': 'idm',
        'human_kwargs': args.human_kwargs,
    })

env_config.update({
    'platoon': args.platoon,
    'whole_trajectory': True,
    'fixed_traj_path': (os.path.join(tc.PROJECT_PATH, args.traj_path)
                        if not args.all_trajectories and args.traj_path != 'None' else None),
})

if args.horizon is not None:
    env_config.update({
        'whole_trajectory': False,
        'horizon': args.horizon,
    })

if args.s3:
    assert (args.platoon == 'scenario1')
    assert (args.all_trajectories == False)

# create env
test_env = TrajectoryEnv(config=env_config, _simulate=True)

# execute controller on traj
mpgs = defaultdict(list)
now = datetime.now().strftime('%d%b%y_%Hh%Mm%Ss')

while True:
    try:
        state = test_env.reset()
    except StopIteration:
        print('Done.')
        break

    print('Using trajectory', test_env.traj['path'], '\n')
    done = False
    test_env.start_collecting_rollout()
    while not done:
        if 'rl' in args.av_controller:
            if False:
                state = test_env.get_state(av_idx=0)
                state[3:] = [0,0,0]
                output = model.predict(state, deterministic=True)
                print('model input', state)
                print('model output', output)
            action = [
                get_first_element(model.predict(test_env.get_state(av_idx=i), deterministic=True))
                for i in range(len(test_env.avs))
            ]
        else:
            action = 0  # do not change (controllers should be implemented via Vehicle objects)
        state, reward, done, infos = test_env.step(action)
    test_env.stop_collecting_rollout()

    for veh in test_env.sim.vehicles:
        mpg = test_env.sim.data_by_vehicle[veh.name]['avg_mpg'][-1]
        mpgs[veh.name].append(mpg)

    all_mpgs = []
    for k, v in mpgs.items():
        print(k, np.mean(v), np.std(v))
        if 'av' in k or 'human' in k:
            all_mpgs.append(np.mean(v))
    print('Avg mpg of AV + humans: ', np.mean(all_mpgs))

    # generate_emissions
    if args.all_trajectories:
        traj_name = Path(test_env.traj['path']).stem
        emissions_path = f'emissions/{now}/emissions_{args.av_controller}_{traj_name}.csv'
    else:
        emissions_path = None

    if args.gen_emissions:
        print('Generating emissions...')
        metadata = {
            'is_baseline': args.s3_baseline,
            'author': args.s3_author,
            'strategy': args.s3_strategy,
        }
        test_env.gen_emissions(emissions_path=emissions_path, upload_to_leaderboard=args.s3, 
                               additional_metadata=metadata)

    if args.gen_metrics:
        tb_callback = TensorboardCallback(eval_freq=0, eval_at_end=True)  # temporary shortcut
        rollout_dict = tb_callback.get_rollout_dict(test_env)

        # plot stuff
        print()
        now = datetime.now().strftime('%d%b%y_%Hh%Mm%Ss')
        plotter = Plotter(f'figs/simulate/{now}')
        for group, metrics in rollout_dict.items():
            for k, v in metrics.items():
                plotter.plot(v, title=k, grid=True, linewidth=1.0)
            plotter.save(group, log='\t')
        print()

        # print stuff
        print('\nMetrics:')
        episode_reward = np.sum(rollout_dict['training']['rewards'])
        av_mpg = rollout_dict['sim_data_av']['avg_mpg'][-1]
        print('\tepisode_reward', episode_reward)
        print('\tav_mpg', av_mpg)
        for penalty in ['crash', 'low_headway_penalty', 'large_headway_penalty', 'low_time_headway_penalty']:
            has_penalty = int(any(rollout_dict['custom_metrics'][penalty]))
            print(f'\thas_{penalty}', has_penalty)

        for (name, array) in [
            ('reward', rollout_dict['training']['rewards']),
            ('headway', rollout_dict['sim_data_av']['headway']),
            ('speed_difference', rollout_dict['sim_data_av']['speed_difference']),
            ('instant_energy_consumption', rollout_dict['sim_data_av']['instant_energy_consumption']),
        ]:
            print(f'\tmin_{name}', np.min(array))
            print(f'\tmax_{name}', np.max(array))

    if not args.all_trajectories:
        break


