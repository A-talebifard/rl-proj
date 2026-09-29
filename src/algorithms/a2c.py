"""
Algorithm 3/3 — Advantage Actor-Critic (A2C) for SIoT topology design.

Classical synchronous actor-critic: one network holds both a policy
(actor) pi(a|s; theta) and a state-value (critic) V(s; theta_v). The
advantage of an action is estimated with the bootstrapped TD error

    A(s, a)  =  Q(s, a) - V(s)   ~   delta  =  R + gamma * V(s') - V(s)

giving the policy-gradient and value losses

    L_pi     = -E[ log pi(a|s) * A(s,a) ]
    L_v      =  E[ (R + gamma*V(s') - V(s))^2 ]
    L        =  L_pi + c_v * L_v - c_e * H[pi]

Unlike PPO, A2C updates from every freshly collected rollout with *no*
clipping and no importance-sampling ratio — its updates are aggressive
and immediate. On this environment that translates into a committed
cost-cutting policy (balanced add/remove mix) and the fastest convergence
to a good topology, at the price of higher update variance.

Hyper-parameters (CONFIG['agents']['A2C']): lr 5e-4, gamma 0.95, rollout
32 steps, entropy coef 0.01, value coef 0.5.

Run standalone demo:   python src/algorithms/a2c.py
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .networks import ActorCritic, masked_categorical   # package import
except ImportError:                                         # direct script run:
    from networks import ActorCritic, masked_categorical    #   python src/algorithms/a2c.py


class A2CAgent:
    """Synchronous advantage actor-critic with invalid-action masking."""

    def __init__(self, state_size, action_size, lr=5e-4, gamma=0.95,
                 entropy_coef=0.01, value_coef=0.5, rollout=32,
                 device='cpu', seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = torch.device(device)
        self.gamma, self.rollout = gamma, rollout
        self.entropy_coef, self.value_coef = entropy_coef, value_coef
        self.model = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.storage = []

    def act(self, state, mask, greedy=False):
        """Sample (or take the mode of) the masked categorical policy."""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
            logits, value = self.model(s)
            dist = masked_categorical(logits, m)
            a = dist.mode if greedy else dist.sample()
            return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    def remember(self, s, a, r, s2, d, logp, val, mask):
        """Store one on-policy transition (with log-prob and V(s) at collection time)."""
        self.storage.append((s, a, r, s2, d, logp, val, mask))

    def train_step(self):
        """One actor-critic update over a full rollout (or None if too short)."""
        if len(self.storage) < self.rollout:
            return None
        data = self.storage
        self.storage = []
        s, a, r, s2, d, logp_old, val_old, mask = map(np.array, zip(*data))
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        d = torch.FloatTensor(d).to(self.device)
        mask = torch.BoolTensor(mask).to(self.device)

        # TD targets: R_t + gamma * V(s_{t+1}) (bootstrapped)
        with torch.no_grad():
            s2_t = torch.FloatTensor(s2).to(self.device)
            _, v_next = self.model(s2_t)
            target = r + self.gamma * v_next * (1 - d)
        logits, values = self.model(s)
        dist = masked_categorical(logits, mask)
        logp = dist.log_prob(a)
        # delta = R + gamma V(s') - V(s);  A(s,a) ~ Q(s,a) - V(s)
        advantage = (target - values).detach()
        pi_loss = -(logp * advantage).mean()
        v_loss = self.value_coef * (target - values).pow(2).mean()
        ent = dist.entropy().mean()
        loss = pi_loss + v_loss - self.entropy_coef * ent
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return float(loss.item())


# ---------------------------------------------------------------------------
# Standalone demo — a tiny masked random MDP, runnable without the SIoT env
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('A2C standalone smoke test (random masked MDP, seed 0)')
    rng = np.random.default_rng(0)
    S, A = 9, 40

    def make_step(s, a):
        s2 = rng.normal(size=S)
        r = float(np.sin(s[0] + a / 10.0))
        return s2, r

    agent = A2CAgent(S, A, seed=0)
    s = rng.normal(size=S)
    mask = rng.random(A) > 0.25
    losses, ret = [], 0.0
    for ep in range(120):
        for _ in range(20):
            a, logp, val = agent.act(s, mask)
            s2, r = make_step(s, a)
            m2 = rng.random(A) > 0.25
            agent.remember(s, a, r, s2, False, logp, val, mask)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            ret += r
            s, mask = s2, m2
    g, _, _ = agent.act(s, mask, greedy=True)
    print(f'  episodes=120  updates={len(losses)}  '
          f'last-10 loss mean={np.mean(losses[-10:]):.4f}')
    print(f'  greedy action on final state: {g}')
    print('  OK — rollout buffer flushed:', len(agent.storage) == 0)
