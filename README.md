# ICRA supplements

## Training details

Both the policy and the value function are 4-layer fully-connected neural networks with 64 hidden units at each layer and hyperbolic tangent non-linearities. We use a learning rate of 3 * 10^-4, a training batch size of 25600, minibatch size of 5120, 10 SGD iterations, and train for 800 iterations over a horizon of 1000 steps.
We set the discount factor to gamma=0.99 and the GAE value to lambda=0.9. Other PPO parameters are left to their default values. We ran grid searches over several hyper-parameters, each on 1 CPU, and training for the parameters mentioned above converged in about 12 hours on a single CPU.

In our simulator, we use a time-step of 0.1s; this is the time-step real world is standardized to. For each rollout,  we sample random 1000 step (100s) long portions of I24 trajectories and use that to simulate the leading vehicle. Behind it we place an AV, followed by 5 cars following an IDM model. Further details on the simulator update procedure and the car following model can be found in the code. 

## Energy model

We have removed the energy model we used during training from this repository as it has not been publicly released yet. 

## Dataset

The whole dataset that we release can be found at: https://vanderbilt.app.box.com/s/z3nignz1cgm16dy56r1mqg9ycds40tkz

The pruned trained dataset that we used can be found in `dataset/icra`. 

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
