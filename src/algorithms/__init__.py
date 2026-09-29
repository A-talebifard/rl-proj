"""
SIoT-RL algorithms package — each RL algorithm in its own standalone file.

    networks.py : shared MLP trunk / ActorCritic head / action masking
    dqn.py      : Deep Q-Network  (replay buffer + target network)
    ppo.py      : Proximal Policy Optimization (clipped surrogate + GAE)
    a2c.py      : Advantage Actor-Critic (bootstrapped TD-error advantage)

Every module is directly runnable on a tiny masked random MDP:

    python src/algorithms/dqn.py
    python src/algorithms/ppo.py
    python src/algorithms/a2c.py

and the three agents are trained under identical conditions (same trunk,
gamma, masking) so performance differences reflect the learning rule.
"""
from .networks import MLP, ActorCritic, masked_categorical
from .dqn import DQNAgent
from .ppo import PPOAgent
from .a2c import A2CAgent

__all__ = ['MLP', 'ActorCritic', 'masked_categorical',
           'DQNAgent', 'PPOAgent', 'A2CAgent']
