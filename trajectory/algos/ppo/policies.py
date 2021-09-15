from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union

import torch as th

from stable_baselines3.common.utils import get_device

from trajectory.algos.pop_art import PopArt

from functools import partial

import numpy as np
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.type_aliases import Schedule


from stable_baselines3.common.policies import (
    ActorCriticPolicy
)

class PopArtActorCriticPolicy(ActorCriticPolicy):
  """
  Policy class for actor-critic algorithms (has both policy and value prediction).
  Used by A2C, PPO and the likes.

  :param observation_space: Observation space
  :param action_space: Action space
  :param lr_schedule: Learning rate schedule (could be constant)
  :param net_arch: The specification of the policy and value networks.
  :param activation_fn: Activation function
  :param ortho_init: Whether to use or not orthogonal initialization
  :param use_sde: Whether to use State Dependent Exploration or not
  :param log_std_init: Initial value for the log standard deviation
  :param full_std: Whether to use (n_features x n_actions) parameters
      for the std instead of only (n_features,) when using gSDE
  :param sde_net_arch: Network architecture for extracting features
      when using gSDE. If None, the latent features from the policy will be used.
      Pass an empty list to use the states as features.
  :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
      a positive standard deviation (cf paper). It allows to keep variance
      above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
  :param squash_output: Whether to squash the output using a tanh function,
      this allows to ensure boundaries when using gSDE.
  :param features_extractor_class: Features extractor to use.
  :param features_extractor_kwargs: Keyword arguments
      to pass to the features extractor.
  :param normalize_images: Whether to normalize images or not,
       dividing by 255.0 (True by default)
  :param optimizer_class: The optimizer to use,
      ``th.optim.Adam`` by default
  :param optimizer_kwargs: Additional keyword arguments,
      excluding the learning rate, to pass to the optimizer
  """

  def _build(self, lr_schedule: Schedule) -> None:
    """
    Create the networks and the optimizer.

    :param lr_schedule: Learning rate schedule
        lr_schedule(1) is the initial learning rate
    """
    self._build_mlp_extractor()

    latent_dim_pi = self.mlp_extractor.latent_dim_pi

    # Separate features extractor for gSDE
    if self.sde_net_arch is not None:
      self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
        self.features_dim, self.sde_net_arch, self.activation_fn
      )

    if isinstance(self.action_dist, DiagGaussianDistribution):
      self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        latent_dim=latent_dim_pi, log_std_init=self.log_std_init
      )
    elif isinstance(self.action_dist, StateDependentNoiseDistribution):
      latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
      self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
      )
    elif isinstance(self.action_dist, CategoricalDistribution):
      self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
    elif isinstance(self.action_dist, MultiCategoricalDistribution):
      self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
    elif isinstance(self.action_dist, BernoulliDistribution):
      self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
    else:
      raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

    self.value_net = PopArt(self.mlp_extractor.latent_dim_vf, 1)
    # Init weights: use orthogonal initialization
    # with small initial weight for the output
    if self.ortho_init:
      # TODO: check for features_extractor
      # Values from stable-baselines.
      # features_extractor/mlp values are
      # originally from openai/baselines (default gains/init_scales).
      module_gains = {
        self.features_extractor: np.sqrt(2),
        self.mlp_extractor: np.sqrt(2),
        self.action_net: 0.01,
        self.value_net: 1,
      }
      for module, gain in module_gains.items():
        module.apply(partial(self.init_weights, gain=gain))

    # Setup optimizer with initial learning rate
    self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


class SplitActorCriticPolicy(ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction but value function
    and actor are separate).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.policy_extractor = SingleMlpExtractor(
            int(self.features_dim / 2), desired_key="pi",
          net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device
        )

        self.vf_extractor = SingleMlpExtractor(
          self.features_dim, desired_key="vf",
          net_arch=self.net_arch, activation_fn=self.activation_fn, device=self.device
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.policy_extractor.latent_dim_pi

        # Separate features extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = PopArt(self.vf_extractor.latent_dim_pi, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.policy_extractor: np.sqrt(2),
                self.vf_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation. Second obs is the extra information for the vf function.
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        policy_obs, _ = th.split(obs, int(obs.shape[1] / 2), dim=1)
        latent_pi, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(self.vf_extractor(obs))
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi = self.policy_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_sde

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        latent_pi, latent_sde = self._get_latent(observation)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution.get_actions(deterministic=deterministic)

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        policy_obs, _ = th.split(obs, int(obs.shape[1] / 2), dim=1)
        features = self.extract_features(policy_obs)
        latent_pi = self.policy_extractor(policy_obs.float())

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_sde

    def evaluate_actions(self, obs: th.Tensor,
                         actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        policy_obs, _ = th.split(obs, int(obs.shape[1] / 2), dim=1)
        latent_pi, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(self.vf_extractor(obs))
        return values, log_prob, distribution.entropy()


class SingleMlpExtractor(nn.Module):
  """
  Constructs an MLP that receives observations as an input and outputs a latent representation
  The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
  of them are shared between the policy network and the value network. It is assumed to be a list with the following
  structure:

  1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
     If the number of ints is zero, there will be no shared layers.
  2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
     It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
     If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

  For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
  network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
  would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
  would be specified as [128, 128].

  Adapted from Stable Baselines.

  :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
  :param net_arch: The specification of the policy and value networks.
      See above for details on its formatting.
  :param activation_fn: The activation function to use for the networks.
  :param desired_key: either pi (to build the policy) or vf to build the value function
  :param device:
  """

  def __init__(
          self,
          feature_dim: int,
          net_arch: List[Union[int, Dict[str, List[int]]]],
          activation_fn: Type[nn.Module],
          desired_key: str,
          device: Union[th.device, str] = "auto",
  ):
    super(SingleMlpExtractor, self).__init__()
    device = get_device(device)
    policy_net, value_net = [], []
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    last_layer_dim_shared = feature_dim

    assert isinstance(net_arch[0][desired_key], list), "Error: net_arch[-1]['{}'] must contain a list of integers.".format(desired_key)
    policy_only_layers = net_arch[0][desired_key]

    last_layer_dim_pi = last_layer_dim_shared

    # Build the non-shared part of the network
    for idx, pi_layer_size in enumerate(zip_longest(policy_only_layers)):
      if pi_layer_size is not None:
        assert isinstance(pi_layer_size[0], int), "Error: net_arch[-1]['pi'] must only contain integers."
        policy_net.append(nn.Linear(last_layer_dim_pi, pi_layer_size[0]))
        policy_net.append(activation_fn())
        last_layer_dim_pi = pi_layer_size[0]

    # Save dim, used to create the distributions
    self.latent_dim_pi = last_layer_dim_pi

    # Create networks
    # If the list of layers is empty, the network will just act as an Identity module
    self.policy_net = nn.Sequential(*policy_net).to(device)

  def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """
    :return: latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    return self.policy_net(features)