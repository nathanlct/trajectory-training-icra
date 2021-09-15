from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import register_policy
from stable_baselines3.ppo import PPO
from stable_baselines3.td3 import TD3

from trajectory.algos.ppo.policies import PopArtActorCriticPolicy, SplitActorCriticPolicy
from trajectory.algos.ppo.ppo import PPO as AugmentedPPO
from trajectory.algos import CustomTD3Policy
from trajectory.callbacks import CheckpointCallback, LoggingCallback, TensorboardCallback
from trajectory.env.trajectory_env import DEFAULT_ENV_CONFIG, TrajectoryEnv
from trajectory.env.utils import dict_to_json

register_policy("PopArtMlpPolicy", PopArtActorCriticPolicy)


def run_experiment(config):
    # create exp logdir
    gs_logdir = config['gs_logdir']
    gs_logdir.mkdir(parents=True, exist_ok=True)

    # create env config
    env_config = dict(DEFAULT_ENV_CONFIG)
    env_config.update({
        'horizon': config['env_horizon'],
        'max_headway': config['env_max_headway'],
        'discrete': config['env_discrete'],
        'num_actions': config['env_num_actions'],
        'use_fs': config['use_fs'],
        'augment_vf': config['augment_vf'],
        'minimal_time_headway': config['env_minimal_time_headway'],
        'include_idm_mpg': config['env_include_idm_mpg'],
        'num_concat_states': config['env_num_concat_states'],
        'num_steps_per_sim': config['env_num_steps_per_sim'],
        'platoon': config['env_platoon'],
        'human_kwargs': config['env_human_kwargs'],
    })

    # create env
    multi_env = make_vec_env(TrajectoryEnv, n_envs=config['n_envs'], env_kwargs=dict(config=env_config))

    # create callbacks
    callbacks = []        
    if not config['no_eval']:
        callbacks.append(TensorboardCallback(
            eval_freq=config['eval_frequency'],
            eval_at_end=True))
    callbacks += [
        LoggingCallback(
            grid_search_config=config['gs_config'],
            log_metrics=True),
        CheckpointCallback(
            save_path=gs_logdir / 'checkpoints',
            save_freq=config['cp_frequency'],
            save_at_end=True,
            s3_bucket='trajectory.env' if config['s3'] else None,
            exp_logdir=config['exp_logdir'],),
    ]
    callbacks = CallbackList(callbacks)

    # create train config
    if config['algorithm'].lower() == 'ppo':
        algorithm = AugmentedPPO if config['augment_vf'] else PPO
        policy = SplitActorCriticPolicy if config['augment_vf'] else PopArtActorCriticPolicy

        train_config = {
            'policy_kwargs': {
                'net_arch': [{
                    'pi': [config['hidden_layer_size']] * config['network_depth'],
                    'vf': [config['hidden_layer_size']] * config['network_depth'],
                }],
            },
            'learning_rate': config['lr'],
            'n_steps': config['n_steps'],
            'batch_size': config['batch_size'],
            'n_epochs': config['n_epochs'],
            'gamma': config['gamma'],
            'gae_lambda': config['gae_lambda'],
            'clip_range': 0.2,
            'clip_range_vf': 50,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
    elif config['algorithm'].lower() == 'td3':
        algorithm = TD3
        policy = CustomTD3Policy if config['augment_vf'] else 'MlpPolicy'

        train_config = {
            'gamma': 0.99,
            'learning_rate': 0.0003,
            'buffer_size': 1000000,
            'learning_starts': 100,
            'train_freq': 100,
            'gradient_steps': 100,
            'batch_size': 128,
            'tau': 0.005,
            'policy_delay': 2,
            'action_noise': None,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
        }
    else:
        raise ValueError(f'Unknown algorithm: {config["algorithm"]}')

    train_config.update({
        'env': multi_env,
        'tensorboard_log': gs_logdir,
        'verbose': 0,  # 0 no output, 1 info, 2 debug
        'seed': None,  # only concerns PPO and not the environment
        'device': 'cpu',  # 'cpu', 'cuda', 'auto'
        'policy': policy,
    })

    # create learn config
    learn_config = {
        'total_timesteps': config['iters'] * config['n_steps'] * config['n_envs'],
        'callback': callbacks,
    }

    # save configs
    configs = {
        'algorithm': algorithm,
        'env_config': env_config,
        'train_config': train_config,
        'learn_config': learn_config
    }
    dict_to_json(configs, gs_logdir / 'configs.json')

    # create model and start training
    model = algorithm(**train_config)
    model.learn(**learn_config)
