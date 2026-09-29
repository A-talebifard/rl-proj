"""
Shared neural-network building blocks for the SIoT-RL agents.

Every RL algorithm in this package (dqn.py / ppo.py / a2c.py) is trained
under identical conditions: the same MLP trunk (128-128, orthogonal init),
the same discount factor gamma = 0.95, and the same invalid-action masking
scheme, so that observed performance differences are attributable to the
learning rule and not to the capacity of the function approximator.

Contents
--------
MLP               : fully-connected trunk shared by all agents.
ActorCritic       : shared actor (policy logits) + critic (state value) head.
masked_categorical: Categorical distribution restricted to valid actions.
"""
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared network trunk
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Fully-connected network  in_dim -> hidden... -> out_dim.

    Orthogonal weight init (gain sqrt(2)) + zero biases: the standard
    initialization for stable deep-RL training.
    """

    def __init__(self, in_dim, out_dim, hidden=(128, 128)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    """One shared body with two heads: policy logits pi and value V(s).

    Used by the policy-gradient algorithms (PPO, A2C). The policy head uses
    gain=0.01 orthogonal init so the initial policy is near-uniform over
    valid actions (small initial logits).
    """

    def __init__(self, in_dim, out_dim, hidden=(128, 128)):
        super().__init__()
        self.body = MLP(in_dim, hidden[-1], hidden[:-1])
        self.pi = nn.Linear(hidden[-1], out_dim)
        self.v = nn.Linear(hidden[-1], 1)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, x):
        h = torch.relu(self.body(x))
        return self.pi(h), self.v(h).squeeze(-1)


def masked_categorical(logits, mask):
    """Categorical distribution over the valid (mask=True) actions only.

    Invalid actions receive a -1e9 logit, so their probability is
    numerically zero regardless of the raw network output. This is how the
    agents respect the SIoT environment's action legality (an 'add' is only
    legal when the edge is absent, a 'remove' only when it is present).
    """
    logits = logits.masked_fill(~mask, -1e9)
    return torch.distributions.Categorical(logits=logits)
