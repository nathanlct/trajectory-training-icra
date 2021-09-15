from datetime import datetime
import itertools
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys

from args import parse_args_train
from trajectory.env.utils import dict_to_json, partition
from setup_exp import run_experiment


if __name__ == '__main__':
    # fix for macOS
    if platform.system() == 'Darwin':
        multiprocessing.set_start_method('spawn')

    # read command line arguments
    args = parse_args_train()

    # create exp logdir
    now = datetime.now()
    now_date = now.strftime('%d%b%y')
    now_time = now.strftime('%Hh%Mm%Ss')
    exp_logdir = Path(args.logdir, now_date, f'{args.expname}_{now_time}')
    exp_logdir.mkdir(parents=True, exist_ok=True)
    print(f'\nCreated experiment logdir at {exp_logdir}')

    # write params.json
    git_branches = subprocess.check_output(['git', 'branch']).decode('utf8')
    git_branch = next(filter(lambda s: s.startswith('*'), git_branches.split('\n')), '?')[2:]
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf8').split()[0]
    whoami = subprocess.check_output(['whoami']).decode('utf8').split()[0]
    
    exp_dict = {
        'full_command': 'python ' + ' '.join(sys.argv),
        'timestamp': datetime.timestamp(datetime.now()),
        'user': whoami,
        'git_branch': git_branch,
        'git_commit': git_commit,
        'n_cpus': multiprocessing.cpu_count(),
        'args': vars(args),
    }

    dict_to_json(exp_dict, exp_logdir / 'params.json')

    # parse command line args to separate grid search args from regular args
    fixed_config, gs_config = partition(
        vars(args).items(),
        pred=lambda kv: type(kv[1]) is list and len(kv[1]) > 1
    )

    # turn args that are a list of one element into just that element
    fixed_config = dict(map(
        lambda kv: (kv[0], kv[1][0]) if type(kv[1]) is list else kv, 
        fixed_config))

    # compute cartesian product of grid search params
    try:
        gs_keys, gs_values = list(zip(*gs_config))
        grid_searches_raw = itertools.product(*gs_values)
        grid_searches = map(lambda gs: dict(zip(gs_keys, gs)), grid_searches_raw)
    except ValueError:
        grid_searches = [{}]

    # generate all configs
    configs = [{'gs_str': (gs_str := '_'.join([f'{k}={v}' for k, v in gs.items()])),
                'gs_logdir': exp_logdir / gs_str,
                'gs_config': gs,
                'exp_logdir': exp_logdir,
                **fixed_config, 
                **gs} for gs in grid_searches]

    # print config and grid searches
    print('\nRunning experiment with the following config:\n')
    print('\n'.join([f'\t{k} = {v}' for k, v in fixed_config.items()]))
    if (n := len(configs)) > 1:
        print(f'\nwith a total of {n} grid searches across the following parameters:\n')
        print('\n'.join([f'\t{k} = {v}' for k, v in zip(gs_keys, gs_values)]))
    print()

    # save git diff to account for uncommited changes
    ps = subprocess.Popen(('git', 'diff', 'HEAD'), stdout=subprocess.PIPE)
    git_diff = subprocess.check_output(('cat'), stdin=ps.stdout).decode('utf8')
    ps.wait()
    if len(git_diff) > 0:
        with open(exp_logdir / 'git_diff.txt', 'w') as fp:
            print(git_diff, file=fp)

    # run experiments
    if len(configs) == 1:
        run_experiment(configs[0])
    else:
        with multiprocessing.Pool(processes=(n := fixed_config['n_processes'])) as pool:
            print(f'Starting training with {n} parallel processes')
            pool.map(run_experiment, configs)
        pool.close()
        pool.join()

    print(f'\nTraining terminated\n\t{exp_logdir}')