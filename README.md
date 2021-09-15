# Trajectory Training

## Installation

```
git clone https://github.com/nathanlct/trajectory_training.git
cd trajectory_training
conda env create -f environment.yml
conda activate trajectory
```

## Train a controller

```
python train.py --expname test --s3 --iters 200 --algorithm PPO --lr 3e-4 --n_epochs 10 --env_platoon 'av human*5'
```

Run `python train.py -h` for a description of all available args. 

Note that a grid search can be ran over most args by specifying several values, for instance `--lr 1e-4 5e-4 1e-5 --gamma 0.9 0.99` will run a total of 3 x 2 = 6 grid searches. 

## Evaluate a controller

RL controller trained using `train.py`:

```
python simulate.py --cp_path checkpoints/.../n.zip --av_controller rl --gen_emissions --gen_metrics --platoon scenario1
```

Baseline controller, eg. IDM or FS:

```
python simulate.py --av_controller idm|fs --gen_emissions --gen_metrics --platoon scenario1
```

Run `python simulate.py -h` for a description of all available args.

To send a controller through the leaderboard pipeline, use `--gen_emissions --s3 --s3_author {your_name} --s3_strategy {controller_name}`.

**Steps to evaluate your custom controller**

Define a vehicle class in `env/vehicles.py` (following the same format as `IDMVehicle` or `FSVehicle` for instance). Once created, go in `env/simulation.py`, import your vehicle class at the top of the file, then go to the `add_vehicle` method and add a mapping from your controller name to the vehicle class in the `vehicle_class` dict. Finally, you should be able to run `python simulate.py --av_controller {your_controller_name}`.

## Visualize results

Running `python simulate.py` with the `--gen_emissions` flag will generate a `.csv` emission file from which you can extract metrics that are interesting to you. Some available scripts are:

- `python visualize/time_space_diagram.py {path_to_emissions.csv}` to generate a time-space diagram
- `python visualize/platoon_mpg.py {path_to_emissions.csv}` to generate a platoon MPG graph (valid if you ran `simulate.py` with the `--platoon scenario1` flag)
- `python visualize/render.py {path_to_emissions.csv}` to render your controller in a Pygame window (not functional right now)

Additionally, a good number of plots and metrics are generated when running `simulate.py` with the `--gen_metrics` flag. 

Note that the behavior of one controller may largely differ from trajectory to trajectory. `simulate.py` defaults to using a custom `--seed {integer}` so that the same trajectory is used across several runs, and when uploading to the leaderboard. Change the seed to run on a different trajectory, or use the `--all_trajectories` flag to run over all available trajectories. If using that flag, you can check out `visualize/plots_from_emissions.py` to plot some data across all trajectories, possibly comparing two controllers. 
