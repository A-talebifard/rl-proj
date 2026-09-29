"""
Algorithm 2/3 — Proximal Policy Optimization (PPO) for SIoT topology design.

On-policy policy-gradient method with a *clipped* surrogate objective. The
agent collects a rollout of transitions with the current policy, then
improves it for several epochs of minibatch SGD while explicitly limiting
how far each update may move the policy away from the data-collecting one:

    ratio_t     = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
    L^CLIP      = E_t[ min( ratio_t * A_t ,
                            clip(ratio_t, 1-eps, 1+eps) * A_t ) ]

Advantages are estimated with Generalized Advantage Estimation (GAE):

    delta_t     = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t^GAE     = sum_l (gamma*lambda)^l * delta_{t+l}

and the total loss adds the value regression and an entropy bonus:

    L = L^CLIP + 0.5 * c_v * (R_t - V(s_t))^2 - c_e * H[pi]

PPO is famously sample-hungry at this scale: after 150 episodes x 20 steps
= 3,000 on-policy samples its clipped updates keep the policy close to the
initial near-uniform distribution (residual action entropy ~88% of the
maximum), which is exactly the under-commitment diagnosed in the analysis
sections of this project.

Hyper-parameters (CONFIG['agents']['PPO']): lr 3e-4, gamma 0.95, clip 0.2,
4 epochs per update, minibatch 64, GAE lambda 0.95, entropy coef 0.01,
value coef 0.5.

Run standalone demo:   python src/algorithms/ppo.py
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .networks import ActorCritic, masked_categorical   # package import
except ImportError:                                         # direct script run:
    from networks import ActorCritic, masked_categorical    #   python src/algorithms/ppo.py


class PPOAgent:
    """Clipped-surrogate PPO with GAE and invalid-action masking."""

    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.95,
                 clip=0.2, epochs=4, batch_size=64, gae_lambda=0.95,
                 device='cpu', seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = torch.device(device)
        self.gamma, self.clip = gamma, clip
        self.epochs, self.batch_size = epochs, batch_size
        self.lam = gae_lambda
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
        """One PPO update over the collected rollout (or None if too short)."""
        if len(self.storage) < self.batch_size:
            return None
        data = self.storage
        self.storage = []
        s, a, r, s2, d, logp_old, val_old, mask = map(np.array, zip(*data))
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        d = torch.FloatTensor(d).to(self.device)
        mask = torch.BoolTensor(mask).to(self.device)
        logp_old = torch.FloatTensor(logp_old).to(self.device)

        with torch.no_grad():
            _, v = self.model(s)
            # GAE advantages
            adv = torch.zeros_like(r)
            lastgaelam = 0.0
            for t in reversed(range(len(r))):
                next_v = 0.0 if d[t] else v[t + 1].item() if t + 1 < len(r) else 0.0
                delta = r[t] + self.gamma * next_v - v[t]
                lastgaelam = delta + self.lam * (1 - d[t]) * lastgaelam
                adv[t] = lastgaelam
            ret = adv + v

        losses = []
        idx = np.arange(len(r))
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, len(idx), self.batch_size):
                j = idx[start:start + self.batch_size]
                logits, values = self.model(s[j])
                dist = masked_categorical(logits, mask[j])
                logp = dist.log_prob(a[j])
                ratio = torch.exp(logp - logp_old[j])
                surr1 = ratio * adv[j]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[j]
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = 0.5 * (ret[j] - values[j]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pi_loss + 0.5 * v_loss - 0.01 * ent
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else None


# ---------------------------------------------------------------------------
# Standalone demo — a tiny masked random MDP, runnable without the SIoT env
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('PPO standalone smoke test (random masked MDP, seed 0)')
    rng = np.random.default_rng(0)
    S, A = 9, 40

    def make_step(s, a):
        s2 = rng.normal(size=S)
        r = float(np.sin(s[0] + a / 10.0))
        return s2, r

    agent = PPOAgent(S, A, seed=0)
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
