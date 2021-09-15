from collections import defaultdict
from datetime import datetime
import gym
from gym.spaces import Discrete, Box
import numpy as np
import pandas as pd
from pathlib import Path
import re
import time
import uuid

from trajectory.data_loader import DataLoader
from trajectory.env.simulation import Simulation
from trajectory.env.utils import get_first_element, upload_to_s3
from trajectory.visualize.platoon_mpg import plot_platoon_mpg
from trajectory.visualize.time_space_diagram import plot_time_space_diagram


# env params that will be used except for params explicitely set in the command-line arguments
DEFAULT_ENV_CONFIG = {
    'horizon': 1000,
    'min_headway': 7.0,
    'max_headway': 120.0,
    'whole_trajectory': False,
    'discrete': False,
    'num_actions': 7,
    'use_fs': False,
    # extra observations for the value function
    'augment_vf': True,
    # if we get closer then this time headway we are forced to break with maximum decel
    'minimal_time_headway': 1.0,
    # if false, we only include the AVs mpg in the calculation
    'include_idm_mpg': False,
    'num_concat_states': 1,
    'num_steps_per_sim': 1,
    # platoon (combination of avs and humans following the leader car)
    'platoon': 'av human*5',
    # controller to use for the AV (available options: rl, idm, fs)
    'av_controller': 'rl',
    'av_kwargs': '{}',
    # human controller & params
    'human_controller': 'idm',
    'human_kwargs': '{}',
    # set to use one specific trajectory
    'fixed_traj_path': None,
}

# platoon presets that can be passed to the "platoon" env param
PLATOON_PRESETS= {
    # scenario 1: 4 AVs with human cars inbetween, some of which are sensing cars used to collect metrics on
    'scenario1': 'human#sensor human*5 (human#sensor human*5 av human*5)*4 human#sensor human*5 human#sensor',
}


class TrajectoryEnv(gym.Env):
    def __init__(self, config, _simulate=False, _verbose=True):
        super().__init__()

        # extract params from config
        self.config = DEFAULT_ENV_CONFIG
        self.config.update(config)
        for k, v in self.config.items():
            setattr(self, k, v)
        self.collect_rollout = False
        self.simulate = _simulate
        self.log_time_counter = time.time()

        if self.use_fs:
            assert self.av_controller == 'rl'
            assert not self.discrete
            self.av_controller = 'rl_fs'

        # instantiate generator of dataset trajectories
        self.data_loader = DataLoader()
        if self.whole_trajectory:
            self.trajectories = self.data_loader.get_all_trajectories()
        else:
            self.trajectories = self.data_loader.get_trajectories(chunk_size=self.horizon * self.num_steps_per_sim)
        self.traj = None

        # create simulation
        self.create_simulation()

        self._verbose = _verbose
        if self._verbose:
            print('\nRunning experiment with the following platoon:', ' '.join([v.name for v in self.sim.vehicles]))
            print(f'with av controller {self.av_controller} ({self.av_kwargs})')
            print(f'with human controller {self.human_controller} ({self.human_kwargs})\n')
            if not self.simulate and len([v for v in self.sim.vehicles if v.kind == 'av']) > 1:
                raise ValueError('Training is only supported with 1 AV in the platoon.')

        # define action space
        if self.discrete:
            self.action_space = Discrete(self.num_actions)
            self.action_set = np.linspace(-1, 1, self.num_actions)
        else:
            if self.use_fs:
                self.action_space = Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
            else:
                self.action_space = Box(low=-3.0, high=1.5, shape=(1,), dtype=np.float32)

        # get number of states
        n_states = len(self.get_base_state())
        n_additional_vf_states = len(self.get_base_additional_vf_state())
        if self.augment_vf:
            n_additional_vf_states = 0
        assert (n_additional_vf_states <= n_states)

        # create buffer to concatenate past states
        n_states *= self.num_concat_states
        self.past_states = np.zeros(n_states)

        # define observation space
        n_obs = n_states
        if self.augment_vf:
            n_obs *= 2  # additional room for vf states
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

    def get_base_state(self, av_idx=None):
        """Dict of state_name: (state_value, state_normalization_scale)"""
        av = self.avs[av_idx if av_idx is not None else 0]
        state = {
            'speed': (av.speed, 40.0),
            'leader_speed': (av.get_leader_speed(), 40.0),
            'headway': (av.get_headway(), 100.0),
        }

        return state

    def get_base_additional_vf_state(self):
        """Dict of state_name: (state_value, state_normalization_scale)
        This state will only be used if augment_vf is set"""
        vf_state = {
            'time': (self.sim.step_counter, self.horizon),
            'avg_miles': (np.mean([self.sim.get_data(veh, 'total_miles')[-1] for veh in self.mpg_cars]), 50.0),
            'avg_gallons': (np.mean([self.sim.get_data(veh, 'total_gallons')[-1] + 1e-6 for veh in self.mpg_cars]), 100.0),
        }

        return vf_state

    def get_state(self, _store_state=False, av_idx=None):
        if av_idx is not None and _store_state == True:
            raise ValueError('Training with several AVs is not supported.')

        if _store_state or av_idx is not None:
            # preprend new state to the saved past states
            state = [value / scale for value, scale in self.get_base_state(av_idx=av_idx).values()]
            self.past_states = np.roll(self.past_states, len(state))
            self.past_states[:len(state)] = state

        if self.augment_vf:
            additional_vf_state = [value / scale for value, scale in self.get_base_additional_vf_state().values()]
            additional_vf_state += [0] * (len(self.past_states) - len(additional_vf_state))
            state = np.concatenate((self.past_states, additional_vf_state))
        else:
            state = self.past_states

        return state

    def create_simulation(self):
        # collect the next trajectory
        if self.fixed_traj_path is not None and self.traj is None:
            self.traj = next(
                t for t in self.data_loader.trajectories
                if str(t['path']).split("/")[-1]
                == self.fixed_traj_path.split("/")[-1])

        if not self.fixed_traj_path:
            self.traj = next(self.trajectories)

        # create a simulation object
        self.time_step = self.traj['timestep']
        self.sim = Simulation(timestep=self.time_step)

        # populate simulation with a trajectoy leader
        self.sim.add_vehicle(controller='trajectory', kind='leader',
            trajectory=zip(self.traj['positions'], self.traj['velocities'], self.traj['accelerations']))

        # parse platoons
        if self.platoon in PLATOON_PRESETS:
            print(f'Setting scenario preset "{self.platoon}"')
            self.platoon = PLATOON_PRESETS[self.platoon]

        # replace (subplatoon)*n into subplatoon ... subplatoon (n times)
        replace1 = lambda match: ' '.join([match.group(1)] * int(match.group(2)))
        self.platoon = re.sub(r'\(([a-z0-9\s\*\#]+)\)\*([0-9]+)', replace1, self.platoon)
        # parse veh#tag1...#tagk*n into (veh, [tag1, ..., tagk], n)
        self.platoon_lst = re.findall(r'([a-z]+)((?:\#[a-z]+)*)(?:\*?([0-9]+))?', self.platoon)

        # spawn vehicles
        self.avs = []
        self.humans = []
        for vtype, vtags, vcount in self.platoon_lst:
            for _ in range(int(vcount) if vcount else 1):
                tags = vtags.split('#')[1:]
                if vtype == 'av':
                    self.avs.append(
                        self.sim.add_vehicle(controller=self.av_controller, kind='av', tags=tags, gap=20, **eval(self.av_kwargs))
                    )
                elif vtype == 'human':
                    self.humans.append(
                        self.sim.add_vehicle(controller=self.human_controller, kind='human', tags=tags, gap=20, **eval(self.human_kwargs))
                    )
                else:
                    raise ValueError(f'Unknown vehicle type: {vtype}. Allowed types are "human" and "av".')

        # define which vehicles are used for the MPG reward
        self.mpg_cars = self.avs + (self.humans if self.include_idm_mpg else [])

        # initialize one data collection step
        self.sim.collect_data()

    def reset(self):
        self.create_simulation()
        return self.get_state(_store_state=True)

    def step(self, actions):
        # additional trajectory data that will be plotted in tensorboard
        metrics = {}

        # apply action to AV
        if self.av_controller == 'rl':
            if type(actions) not in [list, np.ndarray]:
                actions = [actions]
            for av, action in zip(self.avs, actions):
                accel = self.action_set[action] if self.discrete else float(action)
                metrics['rl_accel_before_failsafe'] = accel
                accel = av.set_accel(accel)  # returns accel with failsafes applied
                metrics['rl_accel_after_failsafe'] = accel
        elif self.av_controller == 'rl_fs':
            # RL with FS wrapper
            if type(actions) not in [list, np.ndarray]:
                actions = [actions]
            for av, action in zip(self.avs, actions):
                vdes_command = av.speed + float(action) * self.time_step
                metrics['rl_accel'] = float(action)
                metrics['vdes_command'] = vdes_command
                metrics['vdes_delta'] = float(action) * self.time_step
                av.set_vdes(vdes_command)  # set v_des = v_av + accel * dt

        # execute one simulation step
        end_of_horizon = not self.sim.step()

        # print progress every 5s if running from simulate.py
        if self.simulate and (end_of_horizon or time.time() - self.log_time_counter > 5.0):
            steps, max_steps = self.sim.step_counter, self.traj['size']
            print(f'Progress: {round(steps / max_steps * 100, 1)}% ({steps}/{max_steps} env steps)')
            self.log_time_counter = time.time()

        # compute reward & done
        h = self.avs[0].get_headway()
        th = self.avs[0].get_time_headway()

        reward = 0

        # prevent crashes
        crash = (h <= 0)
        if crash:
            reward -= 50.0

        # forcibly prevent the car from getting too small or large headways
        headway_penalties = {
            'low_headway_penalty': h < self.min_headway,
            'large_headway_penalty': h > self.max_headway,
            'low_time_headway_penalty': th < self.minimal_time_headway}
        if any(headway_penalties.values()):
            reward -= 2.0

        reward -= np.mean([max(self.sim.get_data(veh, 'instant_energy_consumption')[-1], 0) for veh in self.mpg_cars]) / 10.0
        if self.av_controller == 'rl':
            reward -= 0.002 * accel ** 2
        reward += 1

        # give average MPG reward at the end
        # if end_of_horizon:
        #     mpgs = [self.sim.get_data(veh, 'avg_mpg')[-1] for veh in self.mpg_cars]
        #     reward += np.mean(mpgs)

        # log some metrics
        metrics['crash'] = int(crash)
        for k, v in headway_penalties.items():
            metrics[k] = int(v)

        # get next state & done
        next_state = self.get_state(_store_state=True)
        done = (end_of_horizon or crash)
        infos = { 'metrics': metrics }

        if self.collect_rollout:
            self.collected_rollout['actions'].append(get_first_element(actions))
            self.collected_rollout['base_states'].append(self.get_base_state())
            self.collected_rollout['base_states_vf'].append(self.get_base_additional_vf_state())
            self.collected_rollout['rewards'].append(reward)
            self.collected_rollout['dones'].append(done)
            self.collected_rollout['infos'].append(infos)

        return next_state, reward, done, infos

    def start_collecting_rollout(self):
        self.collected_rollout = defaultdict(list)
        self.collect_rollout = True

    def stop_collecting_rollout(self):
        self.collot_rollout = False

    def get_collected_rollout(self):
        return self.collected_rollout

    def gen_emissions(self, emissions_path='emissions', upload_to_leaderboard=True, additional_metadata={}):
        # create emissions dir if it doesn't exist
        if emissions_path is None: 
            emissions_path = 'emissions'
        emissions_path = Path(emissions_path)
        if emissions_path.suffix == '.csv':
            dir_path = emissions_path.parent
        else:                
            now = datetime.now().strftime('%d%b%y_%Hh%Mm%Ss')
            dir_path = Path(emissions_path, now)
            emissions_path = dir_path / 'emissions.csv'
        dir_path.mkdir(parents=True, exist_ok=True)
        self.emissions_path = emissions_path
        self.dir_path = dir_path

        # generate emissions dict
        self.emissions = defaultdict(list)
        for veh in self.sim.vehicles:
            for k, v in self.sim.data_by_vehicle[veh.name].items():
                self.emissions[k] += v

        # sort and save emissions file
        pd.DataFrame(self.emissions) \
            .sort_values(by=['time', 'id']) \
            .to_csv(emissions_path, index=False, float_format="%g")
        print(f'Saved emissions file at {emissions_path}')

        if upload_to_leaderboard:
            # get date & time in appropriate format
            now = datetime.now()
            date_now = now.date().isoformat()
            time_now = now.time().isoformat()

            # create metadata file
            source_id = f'flow_{uuid.uuid4().hex}'
            is_baseline = str(additional_metadata.get('is_baseline', False))
            submitter_name = additional_metadata.get('author', 'blank')
            strategy = additional_metadata.get('strategy', 'blank')
            metadata = pd.DataFrame({
                'source_id': [source_id],
                'submission_time': [time_now],
                'network': ['Single-Lane Trajectory'],
                'is_baseline': [is_baseline],
                'submitter_name': [submitter_name],
                'strategy': [strategy],
                'version': ['3.0'],
                'on_ramp': ['False'],
                'penetration_rate': ['x'],
                'road_grade': ['False'],
                'is_benchmark': ['False'],
            })
            metadata_path = dir_path / 'metadata.csv'
            metadata.to_csv(metadata_path, index=False)

            # custom emissions for leaderboard
            def change_id(vid):
                if vid is None: return None
                elif '#sensor' in vid: return f'sensor_{vid}'
                elif 'human' in vid: return f'human_{vid}'
                elif 'av' in vid: return f'av_{vid}'
                elif 'trajectory' in vid: return f'human_{vid}'
                else: raise ValueError(f'Unknown vehicle type: {vid}')

            self.emissions['id'] = list(map(change_id, self.emissions['id']))
            self.emissions['leader_id'] = list(map(change_id, self.emissions['leader_id']))
            self.emissions['follower_id'] = list(map(change_id, self.emissions['follower_id']))
            self.emissions['x'] = self.emissions['position']
            self.emissions['y'] = [0] * len(self.emissions['x'])
            self.emissions['leader_rel_speed'] = self.emissions['speed_difference']
            self.emissions['road_grade'] = [0] * len(self.emissions['x'])
            self.emissions['edge_id'] = ['edge0'] * len(self.emissions['x'])
            self.emissions['lane_id'] = [0] * len(self.emissions['x'])
            self.emissions['distance'] = self.emissions['total_distance_traveled']
            self.emissions['relative_position'] = self.emissions['total_distance_traveled']
            self.emissions['realized_accel'] = self.emissions['realized_accel']
            self.emissions['target_accel_with_noise_with_failsafe'] = self.emissions['accel']
            self.emissions['target_accel_no_noise_no_failsafe'] = self.emissions['target_accel_no_noise_no_failsafe']
            self.emissions['target_accel_with_noise_no_failsafe'] = self.emissions['target_accel_with_noise_no_failsafe']
            self.emissions['target_accel_no_noise_with_failsafe'] = self.emissions['target_accel_no_noise_with_failsafe']
            self.emissions['source_id'] = [source_id] * len(self.emissions['x'])
            self.emissions['run_id'] = ['run_0'] * len(self.emissions['x'])

            emissions_df = pd.DataFrame(self.emissions).sort_values(by=['time', 'id'])
            emissions_df = emissions_df[['time', 'id', 'x', 'y', 'speed', 'headway',
                'leader_id', 'follower_id', 'leader_rel_speed', 'target_accel_with_noise_with_failsafe',
                'target_accel_no_noise_no_failsafe', 'target_accel_with_noise_no_failsafe',
                'target_accel_no_noise_with_failsafe', 'realized_accel', 'road_grade',
                'edge_id', 'lane_id', 'distance', 'relative_position', 'source_id', 'run_id']]
            leaderboard_emissions_path = dir_path / 'emissions_leaderboard.csv'
            emissions_df.to_csv(leaderboard_emissions_path, index=False)

            platoon_mpg_path = dir_path / 'platoon_mpg.png'
            print(f'Generating platoon MPG plot at {platoon_mpg_path}')
            plot_platoon_mpg(emissions_path, save_path=platoon_mpg_path)

            tsd_path = dir_path / 'time_space_diagram.png'
            print(f'Generating time-space diagram plot at {tsd_path}')
            plot_time_space_diagram(emissions_path, save_path=tsd_path)

            print()

            # upload data to S3

            # metadata
            upload_to_s3(
                'circles.data.pipeline',
                f'metadata_table/date={date_now}/partition_name={source_id}_METADATA/{source_id}_METADATA.csv',
                metadata_path, log=True
            )
            # emissions
            upload_to_s3(
                'circles.data.pipeline',
                f'fact_vehicle_trace/date={date_now}/partition_name={source_id}/{source_id}.csv',
                leaderboard_emissions_path,
                {'network': metadata['network'][0],
                 'is_baseline': metadata['is_baseline'][0],
                 'version': metadata['version'][0],
                 'on_ramp': metadata['on_ramp'][0],
                 'road_grade': metadata['road_grade'][0]},
                log=True
            )
            # platoons MPG plot
            upload_to_s3(
                'circles.data.pipeline',
                f'platoon_mpg/date={date_now}/partition_name={source_id}/{source_id}.png',
                platoon_mpg_path, log=True
            )
            # time-space diagram
            upload_to_s3(
                'circles.data.pipeline',
                f'time_space_diagram/date={date_now}/partition_name={source_id}/{source_id}.png',
                tsd_path, log=True
            )

