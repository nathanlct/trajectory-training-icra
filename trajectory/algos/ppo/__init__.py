# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common.policies import (
    register_policy,
)
from trajectory.algos.ppo.policies import SplitActorCriticPolicy

MlpPolicySplit = SplitActorCriticPolicy

register_policy("MlpPolicySplit", SplitActorCriticPolicy)
