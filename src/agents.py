"""
Deep Reinforcement Learning agents for SIoT supply chain topology design.

v3 note — the three algorithms now live in their own standalone files
under ``src/algorithms/`` (one file per algorithm, each runnable on its
own with ``python src/algorithms/<algo>.py``):

    algorithms/dqn.py   Deep Q-Network   (replay buffer + target network)
    algorithms/ppo.py   PPO              (clipped surrogate + GAE)
    algorithms/a2c.py   A2C              (advantage actor-critic)
    algorithms/networks.py               shared trunk / masking utilities

This module remains as a thin compatibility layer re-exporting the agents,
so the experiment pipeline (run_pipeline.py, experiments.py) and any v1/v2
imports keep working unchanged.
"""
from algorithms.networks import MLP, ActorCritic, masked_categorical   # noqa: F401
from algorithms.dqn import DQNAgent                                    # noqa: F401
from algorithms.ppo import PPOAgent                                    # noqa: F401
from algorithms.a2c import A2CAgent                                    # noqa: F401

__all__ = ['MLP', 'ActorCritic', 'masked_categorical',
           'DQNAgent', 'PPOAgent', 'A2CAgent']
