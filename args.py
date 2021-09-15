import argparse


def parse_args_train():
    parser = argparse.ArgumentParser(description='Train on the trajectory env.')

    # exp params
    parser.add_argument('--expname', type=str, default='test',
        help='Name for the experiment.')
    parser.add_argument('--logdir', type=str, default='./log',
        help='Experiment logs, checkpoints and tensorboard files will be saved under {logdir}/{expname}_[current_time]/.')
    parser.add_argument('--n_processes', type=int, default=1,
        help='Number of processes to run in parallel. Useful when running grid searches.'
             'Can be more than the number of available CPUs.')
    parser.add_argument('--s3', default=False, action='store_true',
        help='If set, experiment data will be uploaded to s3://trajectory.env/. '
             'AWS credentials must have been set in ~/.aws in order to use this.')
    
    parser.add_argument('--iters', type=int, default=1, nargs='+',
        help='Number of iterations (rollouts) to train for.'
             'Over the whole training, {iters} * {n_steps} * {n_envs} environment steps will be sampled.')
    parser.add_argument('--n_steps', type=int, default=640, nargs='+',
        help='Number of environment steps to sample in each rollout in each environment.'
             'This can span over less or more than the environment horizon.'
             'Ideally should be a multiple of {batch_size}.')
    parser.add_argument('--n_envs', type=int, default=1, nargs='+',
        help='Number of environments to run in parallel.')

    parser.add_argument('--cp_frequency', type=int, default=10,
        help='A checkpoint of the model will be saved every {cp_frequency} iterations.'
             'Set to None to not save no checkpoints during training.'
             'Either way, a checkpoint will automatically be saved at the end of training.')
    parser.add_argument('--eval_frequency', type=int, default=10,
        help='An evaluation of the model will be done and saved to tensorboard every {eval_frequency} iterations.'
             'Set to None to run no evaluations during training.'
             'Either way, an evaluation will automatically be done at the start and at the end of training.')
    parser.add_argument('--no_eval', default=False, action='store_true',
        help='If set, no evaluation (ie. tensorboard plots) will be done.')

    # training params
    parser.add_argument('--algorithm', type=str, default='PPO', nargs='+',
        help='RL algorithm to train with. Available options: PPO, TD3.')

    parser.add_argument('--hidden_layer_size', type=int, default=32, nargs='+',
        help='Hidden layer size to use for the policy and value function networks.'
             'The networks will be composed of {network_depth} hidden layers of size {hidden_layer_size}.')
    parser.add_argument('--network_depth', type=int, default=2, nargs='+',
        help='Number of hidden layers to use for the policy and value function networks.'
             'The networks will be composed of {network_depth} hidden layers of size {hidden_layer_size}.')

    parser.add_argument('--lr', type=float, default=3e-4, nargs='+',
        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, nargs='+',
        help='Minibatch size.')
    parser.add_argument('--n_epochs', type=int, default=10, nargs='+',
        help='Number of SGD iterations per training iteration.')
    parser.add_argument('--gamma', type=float, default=0.99, nargs='+',
        help='Discount factor.')
    parser.add_argument('--gae_lambda', type=float, default=0.99, nargs='+',
        help=' Factor for trade-off of bias vs. variance for Generalized Advantage Estimator.')

    parser.add_argument('--augment_vf', type=int, default=1, nargs='+',
        help='If true, the value function will be augmented with some additional states.')

    # env params
    parser.add_argument('--env_num_concat_states', type=int, default=1, nargs='+',
        help='This many past states will be concatenated. If set to 1, it\'s just the current state. '
             'This works only for the base states and not for the additional vf states.')
    parser.add_argument('--env_discrete', type=int, default=0, nargs='+',
        help='If true, the environment has a discrete action space.')
    parser.add_argument('--use_fs', type=int, default=0, nargs='+',
        help='If true, use a FollowerStopper wrapper.')
    parser.add_argument('--env_include_idm_mpg', type=int, default=0, nargs='+',
        help='If true, the mpg is calculated averaged over the AV and the 5 IDMs behind.')
    parser.add_argument('--env_horizon', type=int, default=1000, nargs='+',
        help='Sets the training horizon.')
    parser.add_argument('--env_max_headway', type=int, default=120, nargs='+',
        help='Sets the headway above which we get penalized.')
    parser.add_argument('--env_minimal_time_headway', type=float, default=1.0, nargs='+',
        help='Sets the time headway below which we get penalized.')
    parser.add_argument('--env_num_actions', type=int, default=7, nargs='+',
        help='If discrete is set, the action space is discretized by 1 and -1 with this many actions')
    parser.add_argument('--env_num_steps_per_sim', type=int, default=1, nargs='+',
        help='We take this many sim-steps per environment step i.e. this lets us taking steps bigger than 0.1')

    parser.add_argument('--env_platoon', type=str, default='av human*5', nargs='+',
        help='Platoon of vehicles following the leader. Can contain either "human"s or "av"s. '
             '"(av human*2)*2" can be used as a shortcut for "av human human av human human". '
             'Vehicle tags can be passed with hashtags, eg "av#tag" "human#tag*3"')
    parser.add_argument('--env_human_kwargs', type=str, default='{}', nargs='+',
        help='Dict of keyword arguments to pass to the IDM platoon cars controller.')

    args = parser.parse_args()
    return args


def parse_args_simulate():
    parser = argparse.ArgumentParser(description='Simulate a trained controller or baselines on the trajectory env.')

    parser.add_argument('--cp_path', type=str, default=None,
        help='Path to a saved model checkpoint. '
             'Checkpoint must be a .zip file and have a configs.json file in its parent directory.')
    parser.add_argument('--verbose', default=False, action='store_true',
        help='If set, print information about the loaded controller when {av_controller} is "rl".')
    parser.add_argument('--gen_emissions', default=False, action='store_true',
        help='If set, a .csv emission file will be generated.')
    parser.add_argument('--gen_metrics', default=False, action='store_true',
        help='If set, some figures will be generated and some metrics printed.')
    parser.add_argument('--s3', default=False, action='store_true',
        help='If set, a the emission file and metadata will be uploaded to S3 leaderboard.')
    parser.add_argument('--s3_baseline', default=False, action='store_true',
        help='If set, the data will be uploaded to S3 as a baseline.')
    parser.add_argument('--s3_author', type=str, default='blank',
        help='Submitter name that will be used when uploading to S3.')
    parser.add_argument('--s3_strategy', type=str, default='blank',
        help='Strategy name that will be used when uploading to S3.')

    parser.add_argument('--horizon', type=int, default=None,
        help='Number of environment steps to simulate. If None, use a whole trajectory.')
    parser.add_argument('--traj_path', type=str, default=None,
        help='Set to a .csv path of a trajectory to use a specific trajectory. '
             'If set to None, a random trajectory is used.')
    parser.add_argument('--platoon', type=str, default='av human*5',
        help='Platoon of vehicles following the leader. Can contain either "human"s or "av"s. '
             '"(av human*2)*2" can be used as a shortcut for "av human human av human human". '
             'Vehicle tags can be passed with hashtags, eg "av#tag" "human#tag*3". '
             'Available presets: "scenario1".')
    parser.add_argument('--av_controller', type=str, default='idm',
        help='Controller to control the AV(s) with. Can be either one of "rl", "idm" or "fs".')
    parser.add_argument('--av_kwargs', type=str, default='{}',
        help='Kwargs to pass to the AV controller, as a string that will be evaluated into a dict. '
             'For instance "{\'a\':1, \'b\': 2}" or "dict(a=1, b=2)" for IDM.')
    parser.add_argument('--human_controller', type=str, default='idm',
        help='Controller to control the humans(s) with. Can be either one of "idm" or "fs".')
    parser.add_argument('--human_kwargs', type=str, default='{}',
        help='Kwargs to pass to the human vehicles, as a string that will be evaluated into a dict. '
             'For instance "{\'a\':1, \'b\': 2}" or "dict(a=1, b=2)" for IDM.')

    parser.add_argument('--all_trajectories', default=False, action='store_true',
        help='If set, the script will be ran for all the trajectories in the dataset.')

    args = parser.parse_args()
    return args


def parse_args_savio():
    parser = argparse.ArgumentParser(
        description='Run an experiment on Savio.',
        epilog=f'Example usage: python savio.py --jobname test --mail user@coolmail.com "echo hello world"')

    parser.add_argument('command', type=str, help='Command to run the experiment.')
    parser.add_argument('--jobname', type=str, default='test',
        help='Name for the job.')
    parser.add_argument('--logdir', type=str, default='slurm_logs',
        help='Logdir for experiment logs.')
    parser.add_argument('--mail', type=str, default=None,
        help='Email address where to send experiment status (started, failed, finished).'
             'Leave to None to receive no emails.')
    parser.add_argument('--partition', type=str, default='savio',
        help='Partition to run the experiment on.')
    parser.add_argument('--account', type=str, default='ac_mixedav',
        help='Account to use for running the experiment.')
    parser.add_argument('--time', type=str, default='24:00:00',
        help='Maximum running time of the experiment in hh:mm:ss format, maximum 72:00:00.')

    args = parser.parse_args()
    return args