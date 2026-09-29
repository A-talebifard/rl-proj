"""
Experiment runner for the SIoT-RL supply chain project (v3).

Pipeline:
  1. TRAIN  : DQN / PPO / A2C trained on identical SIoT environments
              (same seeds => same initial graph for every algorithm).
              v3: the reward is SERVICE-AWARE (adds w_S * dFOF) and each
              algorithm is trained on 10 independent seeds.
  2. TEST   : greedy rollout of each trained policy; ALL training seeds
              are evaluated (mean +/- std, n = 10/algo) with significance
              tests (Kruskal-Wallis / Welch / Mann-Whitney).
  3. STRESS : two progressive disruption scenarios, 10 stages each:
                A) node failure probability  (+0.05 per stage, cumulative
                   on a FIXED affected node set)
                B) retailer demand           (+10%  per stage)
              At every stage SIoT-RL is re-executed (adaptation) and the
              network is re-evaluated; a static (no-adaptation) baseline is
              recorded as well.
  4. ANALYZE: policy-behavior logging, convergence statistics,
              robustness slopes/AUC (see analysis.py).

NOTE (protocol comparability): the best-topology SELECTION score
(combined_score) is intentionally unchanged across v1/v2/v3 — it is the
pure cost-risk objective. Only the TRAINING reward changed in v3, so any
change in the reported FOF is attributable to the service-aware reward.
"""
import numpy as np
import pandas as pd
from scipy import stats as sps

from environment import SIoTRLEnvironment, LAYER_LABEL
from agents import DQNAgent, PPOAgent, A2CAgent

try:
    import config as _config
except Exception:
    _config = None

ALGOS = ['DQN', 'PPO', 'A2C']

_CFG_AGENTS = (_config.CONFIG['agents'] if _config else {
    'DQN': {'lr': 1e-3}, 'PPO': {'lr': 3e-4}, 'A2C': {'lr': 5e-4}})


def make_agent(algo, state_size, action_size, seed, device='cpu'):
    """Build an agent with hyper-parameters taken from the central CONFIG."""
    if algo == 'DQN':
        c = _CFG_AGENTS['DQN']
        return DQNAgent(state_size, action_size, lr=c['lr'], gamma=c['gamma'],
                        epsilon=c['epsilon_start'],
                        epsilon_min=c['epsilon_min'],
                        epsilon_decay=c['epsilon_decay'],
                        buffer_size=c['buffer_size'], batch_size=c['batch_size'],
                        target_update=c['target_update'], device=device,
                        seed=seed)
    if algo == 'PPO':
        c = _CFG_AGENTS['PPO']
        return PPOAgent(state_size, action_size, lr=c['lr'], gamma=c['gamma'],
                        clip=c['clip'], epochs=c['epochs'],
                        batch_size=c['batch_size'],
                        gae_lambda=c['gae_lambda'], device=device, seed=seed)
    if algo == 'A2C':
        c = _CFG_AGENTS['A2C']
        return A2CAgent(state_size, action_size, lr=c['lr'], gamma=c['gamma'],
                        entropy_coef=c['entropy_coef'],
                        value_coef=c['value_coef'], rollout=c['rollout'],
                        device=device, seed=seed)
    raise ValueError(algo)


def combined_score(Z1, Z2, Z1_0, Z2_0):
    """Weighted objective relative to the initial graph (lower = better)."""
    return 0.6 * Z1 / max(Z1_0, 1e-9) + 0.4 * Z2 / max(Z2_0, 1e-9)


def train_agent(algo, env, episodes=200, seed=0, device='cpu', log_every=25):
    """Train one algorithm; returns agent, history and best topology."""
    agent = make_agent(algo, env.state_size, env.action_size, seed, device)
    env.reset()
    Z1_0, Z2_0 = env.scn.calculate_objectives()

    hist = {'episode': [], 'reward': [], 'Z1': [], 'Z2': [], 'loss': [],
            'confidence': [], 'edges': []}
    best = {'score': np.inf, 'snapshot': None, 'metrics': None,
            'episode': -1}

    for ep in range(episodes):
        state = env.reset()
        mask = env.action_mask()
        ep_reward, last_loss = 0.0, None
        for step in range(env.max_steps):
            if algo == 'DQN':
                a = agent.act(state, mask)
                next_state, r, done, info = env.step(a)
                agent.remember(state, a, r, next_state, done,
                               env.action_mask())
                last_loss = agent.train_step()
            else:
                a, logp, val = agent.act(state, mask)
                next_state, r, done, info = env.step(a)
                agent.remember(state, a, r, next_state, done, logp, val,
                               env.action_mask())
                last_loss = agent.train_step()
            ep_reward += r
            state = next_state
            mask = env.action_mask()

        ev = env.evaluate()
        score = combined_score(ev['Z1'], ev['Z2'], Z1_0, Z2_0)
        if score < best['score']:
            best = {'score': score, 'snapshot': env.scn.snapshot(),
                    'metrics': dict(ev, episode=ep), 'episode': ep}

        hist['episode'].append(ep)
        hist['reward'].append(ep_reward)
        hist['Z1'].append(ev['Z1'])
        hist['Z2'].append(ev['Z2'])
        hist['loss'].append(last_loss if last_loss is not None else np.nan)
        hist['confidence'].append(ev['confidence'])
        hist['edges'].append(ev['edges'])

        if algo == 'DQN':
            agent.decay_epsilon()
        if (ep + 1) % log_every == 0:
            print(f"  [{algo} seed={seed}] ep {ep+1}/{episodes} "
                  f"reward={ep_reward:+.2f} Z1={ev['Z1']:.0f} "
                  f"Z2={ev['Z2']:.0f} score={score:.3f}", flush=True)

    return agent, hist, best


def greedy_rollout(agent, algo, env, scn_start, max_steps=20,
                   log_actions=False):
    """TEST phase: run the trained policy greedily, keep the best topology.

    v2: with log_actions=True the executed action sequence is recorded
    (op, src, dst, tier-pair, executed?) for the policy-behavior analysis.
    """
    state = env.reset(scn=scn_start)
    mask = env.action_mask()
    Z1_0, Z2_0 = env.scn.calculate_objectives()
    best = {'score': np.inf, 'snapshot': None, 'metrics': None}
    action_log = []
    tier_of = {n: d['layer'] for n, d in env.scn.G.nodes(data=True)}

    for step in range(max_steps):
        if algo == 'DQN':
            a = agent.act(state, mask, greedy=True)
        else:
            a, _, _ = agent.act(state, mask, greedy=True)
        op, src, dst = env.actions[a]
        next_state, r, done, info = env.step(a)
        if log_actions:
            action_log.append({'step': step, 'op': op, 'src': src,
                               'dst': dst,
                               'tier_pair': (LAYER_LABEL[tier_of[src]],
                                             LAYER_LABEL[tier_of[dst]]),
                               'executed': info['executed'],
                               'reward': r})
        state = next_state
        mask = env.action_mask()
        ev = env.evaluate()
        score = combined_score(ev['Z1'], ev['Z2'], Z1_0, Z2_0)
        if score < best['score']:
            best = {'score': score, 'snapshot': env.scn.snapshot(),
                    'metrics': dict(ev, step=step)}
    if best['metrics'] is None:
        best['metrics'] = env.evaluate()
        best['snapshot'] = env.scn.snapshot()
    best['actions'] = action_log
    return best


# ---------------------------------------------------------------------------
# v2: multi-seed test evaluation + significance tests
# ---------------------------------------------------------------------------
def test_multiseed(training_out, csv_path, scale='medium'):
    """Greedy rollout of EVERY (algorithm, seed) policy on its own graph.

    Returns (per_seed_df, stats_dict, mean_df). The stats dict contains
    Kruskal-Wallis over the 3 algorithms and pairwise Welch t / Mann-Whitney
    tests for the three key metrics.
    """
    rows = []
    for algo in ALGOS:
        for si, seed in enumerate(training_out[algo]['envs'] and
                                  range(len(training_out[algo]['envs']))):
            env = training_out[algo]['envs'][si]
            agent = training_out[algo]['agents'][si]
            best = greedy_rollout(agent, algo, env, env.initial)
            m = best['metrics']
            Z1_0, Z2_0 = env.initial.calculate_objectives()
            rows.append({'algorithm': algo, 'seed': seed,
                         'Z1': m['Z1'], 'Z2': m['Z2'], 'ratio': m['ratio'],
                         'FOF': m['FOF'], 'edges': m['edges'],
                         'confidence': m['confidence'],
                         'headroom': m.get('headroom', np.nan),
                         'dZ1_%': 100 * (m['Z1'] - Z1_0) / Z1_0,
                         'dZ2_%': 100 * (m['Z2'] - Z2_0) / Z2_0})
    df = pd.DataFrame(rows)

    mean_df = df.groupby('algorithm').agg(['mean', 'std']).round(4)

    st = {'per_metric': {}}
    for metric in ('Z1', 'Z2', 'ratio'):
        groups = [df[df.algorithm == a][metric].values for a in ALGOS]
        try:
            kw = sps.kruskal(*groups)
            st['per_metric'][metric] = {'kruskal_H': float(kw.statistic),
                                        'kruskal_p': float(kw.pvalue)}
        except Exception:
            st['per_metric'][metric] = None
        st['per_metric'].setdefault(metric, {})
        pairs = {}
        for i, a1 in enumerate(ALGOS):
            for a2 in ALGOS[i + 1:]:
                x = df[df.algorithm == a1][metric].values
                y = df[df.algorithm == a2][metric].values
                try:
                    t = sps.ttest_ind(x, y, equal_var=False)
                    u = sps.mannwhitneyu(x, y, alternative='two-sided')
                    pairs[f'{a1}-{a2}'] = {
                        'welch_t': float(t.statistic),
                        'welch_p': float(t.pvalue),
                        'mwu_p': float(u.pvalue)}
                except Exception:
                    pairs[f'{a1}-{a2}'] = None
        st['per_metric'][metric]['pairs'] = pairs
    return df, st, mean_df


# ---------------------------------------------------------------------------
# v2: policy-behavior metrics
# ---------------------------------------------------------------------------
def policy_metrics(agent, algo, env):
    """Behavioral fingerprint of a trained policy on the initial graph.

    Returns action-type distribution of the greedy rollout, per-tier edge
    deltas, and the initial-state action entropy (exploration appetite of
    the learned stochastic policy; DQN reports its epsilon schedule value).
    """
    import torch
    best = greedy_rollout(agent, algo, env, env.initial, log_actions=True)
    log = best['actions']
    n_add = sum(1 for x in log if x['op'] == 'add' and x['executed'])
    n_rm = sum(1 for x in log if x['op'] == 'remove' and x['executed'])
    n_skip = sum(1 for x in log if not x['executed'])

    state = env.reset()
    mask = env.action_mask()
    entropy = np.nan
    if algo != 'DQN':
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            m = torch.BoolTensor(mask).unsqueeze(0)
            logits, _ = agent.model(s)
            p = torch.softmax(logits.masked_fill(~m, -1e9), dim=-1)
            entropy = float(-(p * torch.log(p + 1e-12)).sum().item())
    eps = None
    if algo == 'DQN':
        eps = agent.epsilon

    # per-tier edge delta (optimized vs initial)
    init_edges = {lp: 0 for lp in env.initial.G.edges()}
    opt_edges = list(best['snapshot'].G.edges())
    init_set = set(env.initial.G.edges())
    tier_delta = {}
    for u, v in opt_edges:
        lu = env.initial.G.nodes[u]['layer']
        lv = env.initial.G.nodes[v]['layer']
        tier_delta[f'{lu}->{lv}'] = tier_delta.get(f'{lu}->{lv}', 0) + \
            (0 if (u, v) in init_set else 1)
    for u, v in init_set:
        if not best['snapshot'].G.has_edge(u, v):
            lu = env.initial.G.nodes[u]['layer']
            lv = env.initial.G.nodes[v]['layer']
            tier_delta[f'{lu}->{lv}'] = tier_delta.get(f'{lu}->{lv}', 0) - 1

    return {'n_add': n_add, 'n_remove': n_rm, 'n_skipped': n_skip,
            'entropy': entropy, 'epsilon': eps,
            'n_actions': int(env.action_size),   # v3: for entropy normalisation
            'tier_delta': tier_delta,
            'edges_initial': env.initial.G.number_of_edges(),
            'edges_final': best['snapshot'].G.number_of_edges(),
            'best_metrics': best['metrics']}


# ---------------------------------------------------------------------------
# Disruption experiment (v2: cumulative, fixed affected set)
# ---------------------------------------------------------------------------
def disruption_experiment(env_template, agent, algo, disruption='failure',
                          stages=10, step=0.05, affected_frac=0.3, seed=7,
                          cumulative=True):
    """Progressive stress test with SIoT-RL re-optimization at each stage.

    v2: when cumulative=True the disrupted node set is sampled ONCE with
    `seed` and stays fixed; failure probability then accumulates on the
    SAME nodes across stages (monotonic degradation, directly matching the
    '+0.05 per step' protocol). v1 re-sampled the set each stage.
    """
    rng = np.random.RandomState(seed)
    records = []
    base = env_template.initial.snapshot()
    n_nodes = base.G.number_of_nodes()
    n_affected = max(2, int(n_nodes * affected_frac))

    # v2: fixed affected set for the whole scenario
    nodes = list(base.G.nodes())
    fixed_chosen = list(np.array(nodes)[
        rng.permutation(len(nodes))[:n_affected]])

    for k in range(stages + 1):
        level = step * k
        scn = base.snapshot()
        if disruption == 'failure':
            for node in fixed_chosen:
                p0 = scn.original_failure_probs[node]
                scn.G.nodes[node]['failure_prob'] = min(0.95, p0 + level)
        else:
            for r in scn.layers['retailers']:
                scn.demand[r] = scn.original_demand[r] * (1.0 + level)

        # -- static: disrupted network, no re-optimization
        scn.normalize_costs()
        scn.allocate_flow()
        Z1_s, Z2_s = scn.calculate_objectives()
        FOF_s = scn.fulfillment_rate()

        # -- SIoT-RL re-optimization (adaptation)
        best = greedy_rollout(agent, algo, env_template, scn)
        m = best['metrics']

        records.append({
            'stage': k, 'level': level,
            'Z1_static': Z1_s, 'Z2_static': Z2_s,
            'Z1_rl': m['Z1'], 'Z2_rl': m['Z2'],
            'ratio_static': Z2_s / Z1_s if Z1_s else 0.0,
            'ratio_rl': m['ratio'], 'FOF_rl': m['FOF'],
            'FOF_static': FOF_s,                      # v2
            'edges_rl': m['edges'], 'confidence_rl': m['confidence'],
            'headroom_rl': m.get('headroom', np.nan),  # v2
            'adaptation_gain': (Z1_s - m['Z1']) / max(Z1_s, 1e-9),
        })
    return pd.DataFrame(records)


def run_all_disruptions(training_out, csv_path, stages=10, seed=7):
    """Run both disruption scenarios for every algorithm (seed-0 agents)."""
    all_rows = []
    for algo in ALGOS:
        env = training_out[algo]['envs'][0]
        agent = training_out[algo]['agents'][0]
        for disr, step, label in [('failure', 0.05, 'Node failure prob'),
                                  ('demand', 0.10, 'Demand increase')]:
            print(f"\n=== Disruption: {label} ({algo}) ===", flush=True)
            df = disruption_experiment(
                env, agent, algo, disruption=disr, stages=stages,
                step=step, seed=seed)
            df['algorithm'] = algo
            df['disruption'] = disr
            all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)
