"""Resumable experiment pipeline for SIoT-RL (v2).

Phases (each runnable within a single shell call, < 10 min):
  --train ALGO SEED    train one algorithm on one seed, cache agent+history
  --test2              v2 multi-seed test evaluation + significance tests
  --policy             v2 policy-behavior metrics (action mix, entropy, tiers)
  --disruption2        v2 cumulative progressive disruptions (both scenarios)
  --analysis           v2 full result analysis -> results/analysis.json
  --figures2           regenerate ALL publication figures incl. v2 ones
  --config             export the central CONFIG to results/config.json
  --summary            merge training histories into training_histories.csv
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG, export_json as export_config          # noqa: E402
from environment import SIoTRLEnvironment            # noqa: E402
from experiments import (ALGOS, train_agent, greedy_rollout,          # noqa: E402
                          test_multiseed, policy_metrics,
                          run_all_disruptions)
import analysis as an                                             # noqa: E402
import visualization as viz                                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'dynamic_supply_chain_logistics_dataset.csv')
CACHE = os.path.join(ROOT, 'cache')
MET = os.path.join(ROOT, 'results', 'metrics')
FIG = os.path.join(ROOT, 'results', 'figures')
RES = os.path.join(ROOT, 'results')
for d in (CACHE, MET, FIG):
    os.makedirs(d, exist_ok=True)

EPISODES = CONFIG['training']['episodes']
SEEDS = tuple(CONFIG['training']['seeds'])
DEVICE = CONFIG['runtime']['device']


# ------------------------------------------------------------------ shared
def load_training_out():
    """Rebuild the training_out structure from cache (agents + histories)."""
    out = {a: {'histories': [], 'agents': [], 'bests': [], 'envs': []}
           for a in ALGOS}
    for algo in ALGOS:
        for seed in SEEDS:
            with open(f'{CACHE}/hist_{algo}_{seed}.json') as f:
                hist = json.load(f)
            hist['loss'] = [np.nan if (v is None or (isinstance(v, float)
                             and np.isnan(v))) else v for v in hist['loss']]
            out[algo]['histories'].append(hist)
            env = SIoTRLEnvironment(DATA, scale=CONFIG['network']['scale'],
                                    seed=seed, max_steps=CONFIG['env']['max_steps'])
            out[algo]['envs'].append(env)
            from agents import DQNAgent, PPOAgent, A2CAgent
            if algo == 'DQN':
                agent = DQNAgent(env.state_size, env.action_size, seed=seed)
            elif algo == 'PPO':
                agent = PPOAgent(env.state_size, env.action_size, seed=seed)
            else:
                agent = A2CAgent(env.state_size, env.action_size, seed=seed)
            ckpt = torch.load(f'{CACHE}/agent_{algo}_{seed}.pt',
                              weights_only=True)
            agent.policy.load_state_dict(ckpt['state']) if algo == 'DQN' else \
                agent.model.load_state_dict(ckpt['state'])
            if algo == 'DQN':
                # v3: restore the *trained* exploration level (the checkpoint
                # stores weights only; epsilon follows the analytic schedule)
                c = CONFIG['agents']['DQN']
                agent.epsilon = max(c['epsilon_min'],
                                    c['epsilon_start'] *
                                    c['epsilon_decay'] ** EPISODES)
            out[algo]['agents'].append(agent)
            with open(f'{CACHE}/best_{algo}_{seed}.json') as f:
                out[algo]['bests'].append(json.load(f))
    return out


# ------------------------------------------------------------------ phases
def phase_train(algo, seed):
    t0 = time.time()
    env = SIoTRLEnvironment(DATA, scale=CONFIG['network']['scale'], seed=seed,
                            max_steps=CONFIG['env']['max_steps'])
    agent, hist, best = train_agent(algo, env, episodes=EPISODES,
                                    seed=seed, device=DEVICE)
    net = agent.policy if algo == 'DQN' else agent.model
    torch.save({'state': net.state_dict(), 'algo': algo, 'seed': seed},
               f'{CACHE}/agent_{algo}_{seed}.pt')
    with open(f'{CACHE}/hist_{algo}_{seed}.json', 'w') as f:
        json.dump({k: [float(x) if x is not None else None for x in v]
                   for k, v in hist.items()}, f)
    with open(f'{CACHE}/best_{algo}_{seed}.json', 'w') as f:
        json.dump({'score': float(best['score']),
                   'metrics': {k: float(v) for k, v in
                               best['metrics'].items()},
                   'episode': int(best['episode'])}, f)
    print(f"[{algo} seed={seed}] done in {(time.time()-t0)/60:.1f} min "
          f"(best score {best['score']:.3f} @ ep {best['episode']})")


def phase_test2():
    """v2: evaluate EVERY (algo, seed) policy + significance tests."""
    training_out = load_training_out()
    df, st, mean_df = test_multiseed(training_out, DATA,
                                     scale=CONFIG['network']['scale'])
    df.to_csv(f'{MET}/test_metrics_multiseed.csv', index=False)
    with open(f'{RES}/test_stats.json', 'w') as f:
        json.dump(st, f, indent=2)
    print(df.groupby('algorithm')[['Z1', 'Z2', 'ratio', 'FOF', 'edges']]
          .agg(['mean', 'std']).round(3))
    print('\nKruskal-Wallis p-values:',
          {m: round(st['per_metric'][m]['kruskal_p'], 4)
           for m in ('Z1', 'Z2', 'ratio')})


def phase_policy():
    """v2: policy-behavior fingerprint per algorithm (seed-0 agents)."""
    import math

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(x) for x in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        return o

    training_out = load_training_out()
    mets = {}
    for algo in ALGOS:
        env = training_out[algo]['envs'][0]
        agent = training_out[algo]['agents'][0]
        print(f'policy metrics for {algo} ...', flush=True)
        mets[algo] = policy_metrics(agent, algo, env)
        mets[algo].pop('best_metrics', None)
    with open(f'{RES}/policy_metrics.json', 'w') as f:
        json.dump(_clean(mets), f, indent=2, allow_nan=False)
    for a, m in mets.items():
        print(f"  {a}: add={m['n_add']} remove={m['n_remove']} "
              f"skip={m['n_skipped']} entropy={m['entropy']} "
              f"edges {m['edges_initial']}->{m['edges_final']} "
              f"tier_delta={m['tier_delta']}")


def phase_disruption2(algo_filter=None):
    """v2: cumulative disruptions with fixed affected node set."""
    training_out = load_training_out()
    algos = [a for a in ALGOS if (algo_filter is None or a == algo_filter)]
    dfs = []
    for algo in algos:
        env = training_out[algo]['envs'][0]
        agent = training_out[algo]['agents'][0]
        for disr, step, label in [('failure', 0.05, 'Node failure prob'),
                                  ('demand', 0.10, 'Demand increase')]:
            print(f"\n=== Disruption: {label} ({algo}) ===", flush=True)
            from experiments import disruption_experiment
            df = disruption_experiment(
                env, agent, algo, disruption=disr,
                stages=CONFIG['disruption']['stages'],
                step=step, seed=CONFIG['disruption']['seed'],
                cumulative=CONFIG['disruption']['cumulative'])
            df['algorithm'] = algo
            df['disruption'] = disr
            dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    path = f'{MET}/disruption_results.csv'
    if algo_filter is not None and os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~old.algorithm.isin([algo_filter])]
        out = pd.concat([old, out], ignore_index=True)
    out = out.sort_values(['disruption', 'algorithm', 'stage'])
    out.to_csv(path, index=False)
    print(out.groupby(['disruption', 'algorithm'])
          [['Z1_rl', 'Z2_rl', 'FOF_rl', 'FOF_static']].last().round(2))


def phase_analysis():
    """v2: full result analysis -> results/analysis.json."""
    training_out = load_training_out()
    with open(f'{RES}/test_stats.json') as f:
        st = json.load(f)
    test_df = pd.read_csv(f'{MET}/test_metrics_multiseed.csv')
    disr = pd.read_csv(f'{MET}/disruption_results.csv')
    with open(f'{RES}/policy_metrics.json') as f:
        policy_mets = json.load(f)
    env0 = SIoTRLEnvironment(DATA, scale=CONFIG['network']['scale'],
                             seed=SEEDS[0], max_steps=CONFIG['env']['max_steps'])
    Z1_0, Z2_0 = env0.initial.calculate_objectives()
    A = an.build_analysis(training_out, test_df, st, disr, policy_mets,
                          Z1_0, Z2_0, CONFIG)
    an.export_json(A, f'{RES}/analysis.json')
    for line in an.findings_text(A, test_df, CONFIG):
        print(line)


def phase_figures2():
    """Regenerate all figures (v1 set + 3 new v2 figures)."""
    training_out = load_training_out()
    test_df = pd.read_csv(f'{MET}/test_metrics_multiseed.csv')
    disr = pd.read_csv(f'{MET}/disruption_results.csv')
    with open(f'{RES}/analysis.json') as f:
        A = json.load(f)
    env0 = SIoTRLEnvironment(DATA, scale=CONFIG['network']['scale'],
                             seed=SEEDS[0], max_steps=CONFIG['env']['max_steps'])
    Z1_0, Z2_0 = env0.initial.calculate_objectives()

    # test-phase single-table (seed-0 view, consistent with v1 fig04)
    seed0 = test_df[test_df.seed == SEEDS[0]].set_index('algorithm')
    seed0 = seed0.loc[ALGOS]
    seed0['dZ1_vs_initial_%'] = 100 * (seed0['Z1'] - Z1_0) / Z1_0
    seed0['dZ2_vs_initial_%'] = 100 * (seed0['Z2'] - Z2_0) / Z2_0
    best_algo = str(seed0['ratio'].idxmin())

    best_topology = {}
    for algo in ALGOS:
        agent = training_out[algo]['agents'][0]
        env_t = training_out[algo]['envs'][0]
        best_topology[algo] = greedy_rollout(agent, algo, env_t, env0.initial)

    viz.plot_topology(env0.initial,
                      'Initial SIoT supply chain network (before RL)',
                      f'{FIG}/fig01_topology_initial.png')
    # per-algorithm initial topologies (v3.1)
    for _algo in ALGOS:
        viz.plot_topology(env0.initial,
                          f'Initial SIoT supply chain network ({_algo}, before RL)',
                          f'{FIG}/fig01_topology_initial_{_algo}.png')
    viz.plot_topology(best_topology[best_algo]['snapshot'],
                      f'Optimized SIoT-RL network ({best_algo}, after training)',
                      f'{FIG}/fig01_topology_optimized.png')
    # per-algorithm optimized topologies + side-by-side comparison (v3.1)
    for _algo in ALGOS:
        viz.plot_topology(best_topology[_algo]['snapshot'],
                          f'Optimized SIoT-RL network ({_algo}, after training)',
                          f'{FIG}/fig01_topology_optimized_{_algo}.png')
    viz.plot_topology_comparison(best_topology,
                                 f'{FIG}/fig01_topology_comparison.png')
    viz.plot_training_curves(training_out, f'{FIG}/fig02_training_curves.png')
    viz.plot_objective_convergence(training_out, f'{FIG}/fig03_objectives.png',
                                   Z1_0, Z2_0)
    viz.plot_algorithm_comparison(seed0, f'{FIG}/fig04_algorithm_comparison.png')
    viz.plot_trust_matrix(best_topology[best_algo]['snapshot'],
                          f'{FIG}/fig07_trust_matrix.png')
    viz.plot_pep_scores(best_topology[best_algo]['snapshot'],
                        f'{FIG}/fig08_pep_scores.png')
    viz.plot_confidence_evolution(training_out, f'{FIG}/fig10_confidence.png')
    viz.plot_disruption(disr, 'failure', f'{FIG}/fig05_disruption_failure.png')
    viz.plot_disruption(disr, 'demand', f'{FIG}/fig06_disruption_demand.png')
    viz.plot_adaptation(disr, f'{FIG}/fig09_adaptation_gain.png')
    # ---- v2 figures
    viz.plot_tradeoff(A['pareto'], f'{FIG}/fig11_tradeoff.png')
    with open(f'{RES}/policy_metrics.json') as f:
        policy_mets = json.load(f)
    viz.plot_policy_behavior(policy_mets, f'{FIG}/fig12_policy_behavior.png')
    viz.plot_multiseed_test(test_df, f'{FIG}/fig13_multiseed.png')
    print('all figures regenerated (14 total)')


def phase_summary():
    """Merge training histories into results/metrics/training_histories.csv."""
    rows = []
    for algo in ALGOS:
        for si, seed in enumerate(SEEDS):
            with open(f'{CACHE}/hist_{algo}_{seed}.json') as f:
                h = json.load(f)
            for ep in range(len(h['episode'])):
                rows.append({'algorithm': algo, 'seed': seed,
                             'episode': h['episode'][ep],
                             'reward': h['reward'][ep], 'Z1': h['Z1'][ep],
                             'Z2': h['Z2'][ep],
                             'confidence': h['confidence'][ep],
                             'edges': h['edges'][ep], 'loss': h['loss'][ep]})
    hist_df = pd.DataFrame(rows)
    hist_df.to_csv(f'{MET}/training_histories.csv', index=False)
    print(f"training_histories.csv: {len(hist_df)} rows")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', nargs=2, metavar=('ALGO', 'SEED'))
    ap.add_argument('--test2', action='store_true')
    ap.add_argument('--policy', action='store_true')
    ap.add_argument('--disruption2', nargs='?', const='ALL', default=None,
                    metavar='ALGO')
    ap.add_argument('--analysis', action='store_true')
    ap.add_argument('--figures2', action='store_true')
    ap.add_argument('--config', action='store_true')
    ap.add_argument('--summary', action='store_true')
    args = ap.parse_args()

    if args.config:
        print('config exported to', export_config(f'{RES}/config.json'))
    if args.train:
        phase_train(args.train[0], int(args.train[1]))
    if args.test2:
        phase_test2()
    if args.policy:
        phase_policy()
    if args.disruption2:
        flt = None if args.disruption2 == 'ALL' else args.disruption2
        phase_disruption2(flt)
    if args.analysis:
        phase_analysis()
    if args.figures2:
        phase_figures2()
    if args.summary:
        phase_summary()
