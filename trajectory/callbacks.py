import matplotlib
matplotlib.use('agg')

import boto3
from collections import defaultdict, deque
import math
import numpy as np
import os
from pathlib import Path
import random
from stable_baselines3.common.callbacks import BaseCallback
import time

from trajectory.env.trajectory_env import TrajectoryEnv
from trajectory.env.utils import duration_to_str, get_first_element
from trajectory.visualize.plotter import TensorboardPlotter


class TensorboardCallback(BaseCallback):
    """Callback for plotting additional metrics in tensorboard."""
    def __init__(self, eval_freq, eval_at_end):
        super().__init__()

        self.eval_freq = eval_freq
        self.eval_at_end = eval_at_end
        self.rollout = 0

    def _on_training_start(self):
        self.env = self.training_env.envs[0]

    def _on_training_end(self):
        if self.eval_at_end and (self.eval_freq is None or (self.rollout - 1) % self.eval_freq != 0):
            self.log_rollout_dict('idm_eval', self.run_eval(av_controller='idm'))
            self.log_rollout_dict('fs_eval', self.run_eval(av_controller='fs'))
            self.log_rollout_dict('rl_eval', self.run_eval(av_controller='rl'))

    def _on_rollout_start(self):
        if self.eval_freq is not None and self.rollout % self.eval_freq == 0:
            self.env.start_collecting_rollout()

    def _on_rollout_end(self):
        if self.eval_freq is not None and self.rollout % self.eval_freq == 0:
            self.env.stop_collecting_rollout()

            self.log_rollout_dict('rollout', self.get_rollout_dict(self.env))

            self.log_rollout_dict('idm_eval', self.run_eval(av_controller='idm'))
            self.log_rollout_dict('fs_eval', self.run_eval(av_controller='fs'))
            self.log_rollout_dict('rl_eval', self.run_eval(av_controller='rl'))
        
        self.rollout += 1
    
    def log_rollout_dict(self, base_name, rollout_dict):
        plotter = TensorboardPlotter(self.logger)
        for group, metrics in rollout_dict.items():
            for k, v in metrics.items():
                plotter.plot(v, title=k, grid=True, linewidth=2.0)
            plotter.save(f'{base_name}/{base_name}_{group}')

        episode_reward = np.sum(rollout_dict['training']['rewards'])
        av_mpg = rollout_dict['sim_data_av']['avg_mpg'][-1]
        self.logger.record(f'{base_name}/{base_name}_episode_reward', episode_reward)
        self.logger.record(f'{base_name}/{base_name}_av_mpg', av_mpg)
        for penalty in ['crash', 'low_headway_penalty', 'large_headway_penalty', 'low_time_headway_penalty']:
            has_penalty = int(any(rollout_dict['custom_metrics'][penalty]))
            self.logger.record(f'{base_name}/{base_name}_has_{penalty}', has_penalty)

        for (name, array) in [
            ('reward', rollout_dict['training']['rewards']),
            ('headway', rollout_dict['sim_data_av']['headway']),
            ('speed_difference', rollout_dict['sim_data_av']['speed_difference']),
            ('instant_energy_consumption', rollout_dict['sim_data_av']['instant_energy_consumption']),
        ]:
            self.logger.record(f'{base_name}/{base_name}_min_{name}', np.min(array))
            self.logger.record(f'{base_name}/{base_name}_max_{name}', np.max(array))

    def get_rollout_dict(self, env):
        collected_rollout = env.get_collected_rollout()

        rollout_dict = defaultdict(lambda: defaultdict(list))

        rollout_dict['training']['rewards'] = collected_rollout['rewards']
        rollout_dict['training']['dones'] = collected_rollout['dones']
        rollout_dict['training']['actions'] = collected_rollout['actions']

        if 'metrics' in collected_rollout['infos'][0]:
            for info in collected_rollout['infos']:
                for k, v in info['metrics'].items():
                    rollout_dict['custom_metrics'][k].append(v)
        
        for base_state in collected_rollout['base_states']:
            for k, v in base_state.items():
                    rollout_dict['base_state'][k].append(v[0])
        for base_state_vf in collected_rollout['base_states_vf']:
            for k, v in base_state_vf.items():
                    rollout_dict['base_state'][f'vf_{k}'].append(v[0])

        for veh in env.sim.vehicles:
            if veh.kind == 'av':
                for k, v in env.sim.data_by_vehicle[veh.name].items():
                    rollout_dict['sim_data_av'][k] = v

        return rollout_dict

    def run_eval(self, av_controller):
        # set seed so that the different evaluated controllers use the same trajectory
        random.seed(self.rollout)
        np.random.seed(self.rollout)

        # create test env
        config = dict(self.env.config)
        config['whole_trajectory'] = True
        if av_controller != 'rl':
            config['use_fs'] = False
            config['discrete'] = False
        config['av_controller'] = av_controller
        if av_controller == 'idm':
            config['av_kwargs'] = 'dict(v0=45,noise=0)'
        test_env = TrajectoryEnv(config=config, _verbose=False)

        # execute controller on traj
        state = test_env.reset()
        done = False
        test_env.start_collecting_rollout()
        while not done:
            if av_controller == 'rl':
                action = get_first_element(self.model.predict(state, deterministic=True))
            else:
                action = 0
            state, reward, done, infos = test_env.step(action)
        test_env.stop_collecting_rollout()

        return self.get_rollout_dict(test_env)

    def _on_step(self):
        return True


class CheckpointCallback(BaseCallback):
    """Callback for saving a model every `save_freq` rollouts."""
    def __init__(self, save_freq=10, save_path='./checkpoints', save_at_end=False, s3_bucket=None, exp_logdir=None):
        super(CheckpointCallback, self).__init__()

        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_at_end = save_at_end

        self.iter = 0

        self.s3_bucket = s3_bucket
        self.exp_logdir = Path(exp_logdir)

        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_rollout_end(self):
        self.iter += 1

        if self.save_freq is not None and self.iter % self.save_freq == 0:
            self.write_checkpoint()

    def _on_training_end(self):
        if (self.save_freq is None or self.iter % self.save_freq != 0) and self.save_at_end:
            self.write_checkpoint()

    def write_checkpoint(self):
        path = self.save_path / str(self.iter)
        self.model.save(path)
        print(f'Saved model checkpoint to {path}.zip')

        if self.s3_bucket is not None and self.exp_logdir is not None:
            s3 = boto3.resource('s3').Bucket(self.s3_bucket)
            for root, _, file_names in os.walk(self.exp_logdir):
                for file_name in file_names:
                    file_path = Path(root, file_name)
                    file_path_s3 = file_path.relative_to(self.exp_logdir.parent.parent)
                    s3.upload_file(str(file_path), str(file_path_s3))
            print(f'Uploaded exp logdir to s3://{self.s3_bucket}/{self.exp_logdir.relative_to(self.exp_logdir.parent.parent)}')

    def _on_step(self):
        return True


class LoggingCallback(BaseCallback):
    """Callback for logging additional information."""
    def __init__(self, log_metrics=False, grid_search_config={}):
        super(LoggingCallback, self).__init__()

        self.grid_search_config = grid_search_config
        self.log_metrics = log_metrics

        self.ep_info_buffer = deque(maxlen=100)

    def _on_rollout_start(self):
        self.logger.record('time/time_this_iter', duration_to_str(time.time() - self.iter_t0))

        self.rollout_t0 = time.time()
        self.iter_t0 = time.time()

    def _on_rollout_end(self):
        # log current training progress
        if hasattr(self.model, 'n_steps'):
            # for PPO
            timesteps_per_iter = self.training_env.num_envs * self.model.n_steps
            total_iters = math.ceil(self.locals['total_timesteps'] / timesteps_per_iter)
        else:
            # for TD3
            timesteps_per_iter = self.training_env.num_envs * self.model.train_freq.frequency
            if len(self.locals['total_timesteps']) > 0:
                total_iters = math.ceil(self.locals['total_timesteps'][0] / timesteps_per_iter)
            else:
                total_iters = 1

        total_timesteps_rounded = timesteps_per_iter * total_iters
        progress_fraction = self.num_timesteps / total_timesteps_rounded

        if self.log_metrics:
            self.logger.record('time/timesteps', self.num_timesteps)
            self.logger.record('time/iters', self.num_timesteps // timesteps_per_iter)
                
        self.logger.record('time/goal_timesteps', total_timesteps_rounded)
        self.logger.record('time/goal_iters', total_iters)
        self.logger.record('time/training_progress', f'{round(100 * progress_fraction, 1)}%')

        t = time.time()
        self.logger.record('time/time_since_start', duration_to_str(t - self.training_t0))
        self.logger.record('time/time_this_rollout', duration_to_str(t - self.rollout_t0))
        time_left = (t - self.training_t0) / progress_fraction
        self.logger.record('time/estimated_time_left', duration_to_str(time_left))
        self.logger.record('time/timesteps_per_second', round(self.num_timesteps / (t - self.training_t0), 1))

        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", np.mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        
        if self.log_metrics:
            self.print_metrics()

    def _on_training_start(self):
        self.training_t0 = time.time()
        self.iter_t0 = time.time()

    def _on_training_end(self):
        if self.log_metrics:
            self.print_metrics()

    def print_metrics(self):
        metrics = self.logger.get_log_dict()
        gs_str = ', '.join([f'{k} = {v}' for k, v in self.grid_search_config.items()])
        print(f'\nEnd of rollout for grid search: {gs_str}')

        key2str = {}
        tag = None
        for (key, value) in sorted(metrics.items()):
            if isinstance(value, float):
                value_str = f'{value:<10.5g}'
            elif isinstance(value, int) or isinstance(value, str):
                value_str = str(value)
            else:
                continue  # plt figure

            if key.find('/') > 0: 
                tag = key[: key.find('/') + 1]
                key2str[tag] = ''
            if tag is not None and tag in key:
                key = str("   " + key[len(tag) :])

            key2str[key] = value_str

        key_width = max(map(len, key2str.keys()))
        val_width = max(map(len, key2str.values()))

        dashes = '-' * (key_width + val_width + 7)
        lines = [dashes]
        for key, value in key2str.items():
            key_space = ' ' * (key_width - len(key))
            val_space = ' ' * (val_width - len(value))
            lines.append(f'| {key}{key_space} | {value}{val_space} |')
        lines.append(dashes)
        print('\n'.join(lines))

    def _on_step(self):
        if (infos := self.locals['infos'][0].get('episode')) is not None:
            self.ep_info_buffer.extend([infos])
        return True
