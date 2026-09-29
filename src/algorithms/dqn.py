"""
Algorithm 1/3 — Deep Q-Network (DQN) for SIoT supply-chain topology design.

Value-based deep reinforcement learning. DQN approximates the optimal
action-value function

    Q*(s, a; w)  ~  E[ R_t + gamma * max_a' Q*(s', a'; w^-) ]

with a single MLP and stabilizes training with two classical tricks:

1. Experience replay  — transitions (s, a, r, s', done, mask') are stored
   in a FIFO buffer and each learner step samples a random mini-batch,
   which breaks the temporal correlation of on-policy data and reuses every
   observed transition ~ (buffer re-visits) times. In this study DQN
   reuses each transition ~128x on average, extracting the most signal per
   environment sample.

2. Target network — a slowly-updated copy w^- of the online weights w
   provides stable regression targets that only change every
   `target_update` steps, preventing the "chasing its own target" drift.

Loss (per the project's model document):

    L(w) = ( Q_target - Q(s, a; w) )^2 ,         Q_target = r + gamma * max_a' Q(s', a'; w^-)

exploration is epsilon-greedy with an exponential schedule
epsilon <- max(eps_min, epsilon * eps_decay) per episode, and invalid
actions are excluded both from exploration and from the argmax (masking).

Hyper-parameters (CONFIG['agents']['DQN']): lr 1e-3, gamma 0.95,
epsilon 1.0 -> 0.05 (decay 0.995), replay buffer 20k, batch 128,
target sync every 10 learner steps.

Run standalone demo:   python src/algorithms/dqn.py
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .networks import MLP            # package import (algorithms.dqn)
except ImportError:                      # direct script run:
    from networks import MLP             #   python src/algorithms/dqn.py


class DQNAgent:
    """Epsilon-greedy DQN with replay buffer + periodic target sync."""

    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 buffer_size=20000, batch_size=128, target_update=10,
                 device='cpu', seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = torch.device(device)
        self.state_size, self.action_size = state_size, action_size
        self.gamma, self.lr = gamma, lr
        self.epsilon, self.epsilon_min = epsilon, epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size, self.target_update = batch_size, target_update
        self.policy = MLP(state_size, action_size).to(self.device)
        self.target = MLP(state_size, action_size).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.step_count = 0

    def act(self, state, mask, greedy=False):
        """epsilon-greedy action over VALID actions only."""
        if not greedy and random.random() < self.epsilon:
            valid = np.where(mask)[0]
            return int(random.choice(valid))
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            m = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
            q = self.policy(s).masked_fill(~m, -1e9)
            return int(q.argmax(dim=1).item())

    def remember(self, s, a, r, s2, d, m2):
        """Store one transition (with next-state action mask) in the buffer."""
        self.buffer.append((s, a, r, s2, float(d), m2))

    def train_step(self):
        """One MSE regression step on a replayed mini-batch (or None)."""
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, d, m2 = map(np.array, zip(*batch))
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).to(self.device)
        m2 = torch.BoolTensor(m2).to(self.device)

        q_cur = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(s2).masked_fill(~m2, -1e9).max(1)[0]
            q_tgt = r + (1 - d) * self.gamma * q_next
        # L = (Q_target - Q(state, a; w))^2   (MSE, per the model document)
        loss = nn.functional.mse_loss(q_cur, q_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    def decay_epsilon(self):
        """Exponential exploration schedule (called once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------------------------
# Standalone demo — a tiny masked random MDP, runnable without the SIoT env
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('DQN standalone smoke test (random masked MDP, seed 0)')
    rng = np.random.default_rng(0)
    S, A = 9, 40

    def make_step(s, a):
        s2 = rng.normal(size=S)
        r = float(np.sin(s[0] + a / 10.0))          # deterministic-ish reward
        return s2, r

    agent = DQNAgent(S, A, seed=0)
    s = rng.normal(size=S)
    mask = rng.random(A) > 0.25                     # ~75% of actions valid
    losses, ret = [], 0.0
    for ep in range(120):
        for _ in range(20):
            a = agent.act(s, mask)
            s2, r = make_step(s, a)
            m2 = rng.random(A) > 0.25
            agent.remember(s, a, r, s2, False, m2)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            ret += r
            s, mask = s2, m2
        agent.decay_epsilon()
    g = agent.act(s, mask, greedy=True)
    print(f'  episodes=120  learner steps={agent.step_count}  '
          f'last-50 loss mean={np.mean(losses[-50:]):.4f}')
    print(f'  epsilon after schedule: {agent.epsilon:.3f}  '
          f'greedy action on final state: {g}')
    print('  OK — replay buffer size', len(agent.buffer))
