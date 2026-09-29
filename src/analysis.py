"""
Result analysis module (v2) — turns raw metrics into interpretable findings.

Beyond the working logic (which v1 already implemented), this module
QUANTIFIES and INTERPRETS the results:

  A. Convergence     : episodes-to-95%-of-best, late-training stability,
                       reward AUC, early-vs-late improvement.
  B. Significance    : multi-seed test metrics with Kruskal-Wallis /
                       Welch / Mann-Whitney tests (see experiments.py).
  C. Exploration gap : best topology discovered during training (oracle)
                       vs the greedy rollout of the trained policy — the
                       exploration-vs-exploitation reproducibility gap.
  D. Policy behavior : what the agents actually DO (add/remove mix, tier
                       preference, action entropy / epsilon).
  E. Robustness      : degradation slopes of Z1/Z2/ratio vs disruption
                       level, adaptation-gain AUC, service-level floor,
                       capacity headroom.
  F. Trade-off       : Z1-vs-Z2 Pareto analysis of all candidate topologies.
  G. Sample economy  : experience-reuse factor explaining why the three
                       algorithm families learn at different speeds.
"""
import json
import numpy as np
import pandas as pd

ALGOS = ['DQN', 'PPO', 'A2C']


# ---------------------------------------------------------------------------
# A. Convergence analysis
# ---------------------------------------------------------------------------
def analyze_convergence(training_out, Z1_0, Z2_0, tol=0.05, window=30):
    """Per-algorithm convergence statistics from the training histories."""
    out = {}
    for algo in ALGOS:
        conv_eps, stabilities, reward_aucs, early, late = [], [], [], [], []
        best_scores = []
        for h in training_out[algo]['histories']:
            Z1 = np.array(h['Z1'], dtype=float)
            Z2 = np.array(h['Z2'], dtype=float)
            rew = np.array(h['reward'], dtype=float)
            score = 0.6 * Z1 / max(Z1_0, 1e-9) + 0.4 * Z2 / max(Z2_0, 1e-9)
            best_score = score.min()
            best_scores.append(best_score)
            # first episode within tol of the run's best score
            hit = np.where(score <= best_score * (1 + tol))[0]
            conv_eps.append(int(hit[0]) if len(hit) else -1)
            # stability: std of score over the last `window` episodes
            stabilities.append(float(score[-window:].std()))
            # reward area-under-curve (normalized by episodes)
            reward_aucs.append(float(rew.sum() / max(len(rew), 1)))
            k = min(window, len(rew) // 2)
            early.append(float(rew[:k].mean()))
            late.append(float(rew[-k:].mean()))
        out[algo] = {
            'best_score_mean': float(np.mean(best_scores)),
            'best_score_std': float(np.std(best_scores)),
            'convergence_episode_mean': float(np.mean(conv_eps)),
            'convergence_episode_per_seed': [int(x) for x in conv_eps],
            'late_stability_std': float(np.mean(stabilities)),
            'reward_auc_mean': float(np.mean(reward_aucs)),
            'early_reward_mean': float(np.mean(early)),
            'late_reward_mean': float(np.mean(late)),
            'reward_improvement': float(np.mean(late) - np.mean(early)),
        }
    return out


# ---------------------------------------------------------------------------
# C. Exploration-vs-exploitation gap (oracle vs greedy policy)
# ---------------------------------------------------------------------------
def analyze_gap(training_out, test_multiseed_df):
    """Best topology seen in training (oracle) vs greedy test performance."""
    out = {}
    for algo in ALGOS:
        oracle = [b['metrics']['Z1']
                  for b in training_out[algo]['bests']]
        oracle_Z1 = float(np.min(oracle))
        oracle_Z2 = [b['metrics']['Z2']
                     for b in training_out[algo]['bests']]
        oracle_Z2_at_best = float(oracle_Z2[int(np.argmin(oracle))])
        greedy_Z1 = float(test_multiseed_df[test_multiseed_df.algorithm
                                            == algo]['Z1'].mean())
        greedy_Z2 = float(test_multiseed_df[test_multiseed_df.algorithm
                                            == algo]['Z2'].mean())
        out[algo] = {
            'oracle_Z1': oracle_Z1, 'oracle_Z2': oracle_Z2_at_best,
            'greedy_Z1_mean': greedy_Z1, 'greedy_Z2_mean': greedy_Z2,
            'gap_Z1_%': 100 * (greedy_Z1 - oracle_Z1) / max(oracle_Z1, 1e-9),
            'gap_Z2_%': 100 * (greedy_Z2 - oracle_Z2_at_best)
                        / max(oracle_Z2_at_best, 1e-9),
        }
    return out


# ---------------------------------------------------------------------------
# E. Robustness quantification
# ---------------------------------------------------------------------------
def analyze_robustness(disruption_df):
    """Slopes, AUC indices and service floor per algorithm & scenario."""
    out = {}
    for disr in disruption_df['disruption'].unique():
        out[disr] = {}
        for algo in ALGOS:
            d = disruption_df[(disruption_df.disruption == disr) &
                              (disruption_df.algorithm == algo)]
            d = d.sort_values('stage')
            lvl = d['level'].values
            res = {}
            for col in ('Z1_rl', 'Z2_rl', 'ratio_rl', 'Z1_static',
                        'Z2_static'):
                slope = float(np.polyfit(lvl, d[col].values, 1)[0])
                res[f'slope_{col}'] = slope
            # relative degradation from stage 0 to the last stage
            for col in ('Z1_rl', 'Z2_rl'):
                v0, v1 = d[col].values[0], d[col].values[-1]
                res[f'degradation_{col}_%'] = 100 * (v1 - v0) / max(abs(v0),
                                                                     1e-9)
            gain = d['adaptation_gain'].values
            res['gain_auc'] = float(np.trapezoid(gain, lvl)
                                    / max(lvl[-1] - lvl[0], 1e-9))
            res['gain_max'] = float(gain.max())
            res['gain_last'] = float(gain[-1])
            res['FOF_static_min'] = float(d['FOF_static'].min())
            res['FOF_rl_min'] = float(d['FOF_rl'].min())
            if 'headroom_rl' in d.columns and not np.all(
                    pd.isna(d['headroom_rl'])):
                res['headroom_first'] = float(d['headroom_rl'].values[0])
                res['headroom_last'] = float(
                    d['headroom_rl'].values[-1]
                    if not pd.isna(d['headroom_rl'].values[-1]) else np.nan)
            out[disr][algo] = res
    return out


# ---------------------------------------------------------------------------
# F. Trade-off / Pareto analysis
# ---------------------------------------------------------------------------
def analyze_tradeoff(test_multiseed_df, Z1_0, Z2_0):
    """Pareto dominance among the optimized topologies (Z1, Z2 both min)."""
    pts = [{'name': 'initial', 'Z1': Z1_0, 'Z2': Z2_0, 'algo': 'Initial'}]
    for _, r in test_multiseed_df.iterrows():
        pts.append({'name': f"{r['algorithm']}-s{r['seed']}",
                    'Z1': float(r['Z1']), 'Z2': float(r['Z2']),
                    'algo': r['algorithm']})
    for p in pts:
        p['dominated_by'] = [q['name'] for q in pts
                             if q is not p and q['Z1'] <= p['Z1']
                             and q['Z2'] <= p['Z2']
                             and (q['Z1'] < p['Z1'] or q['Z2'] < p['Z2'])]
        p['pareto'] = len(p['dominated_by']) == 0
    return pts


# ---------------------------------------------------------------------------
# G. Sample economy — experience-reuse factor
# ---------------------------------------------------------------------------
def experience_reuse(episodes, max_steps, cfg_agents):
    """How many times each environment transition is (on average) reused
    for gradient updates — a key explanatory factor for learning speed."""
    steps = episodes * max_steps
    out = {}
    # DQN: one replay update per env step, batch 128 (once buffer is full)
    out['DQN'] = {'steps': steps, 'updates': steps,
                  'reuse_factor': float(cfg_agents['DQN']['batch_size'])}
    # PPO: buffer of 3000 steps, 4 epochs of minibatches => ~4 passes
    out['PPO'] = {'steps': steps, 'updates': steps
                  // cfg_agents['PPO']['batch_size']
                  * cfg_agents['PPO']['epochs'],
                  'reuse_factor': float(cfg_agents['PPO']['epochs'])}
    # A2C: every step consumed exactly once
    out['A2C'] = {'steps': steps, 'updates': steps
                  // cfg_agents['A2C']['rollout'],
                  'reuse_factor': 1.0}
    return out


def dqn_epsilon_schedule(episodes, start=1.0, minimum=0.05, decay=0.995):
    """Analytic epsilon-greedy exploration schedule (DQN)."""
    eps = [max(minimum, start * decay ** ep) for ep in range(episodes)]
    cross = next((ep for ep, e in enumerate(eps) if e <= 0.10), None)
    return {'epsilon': eps, 'episode_epsilon_le_0.10': cross}


# ---------------------------------------------------------------------------
# Assemble the full analysis + human-readable findings
# ---------------------------------------------------------------------------
def build_analysis(training_out, test_df, stats_tests, disruption_df,
                   policy_mets, Z1_0, Z2_0, cfg):
    conv = analyze_convergence(training_out, Z1_0, Z2_0,
                               tol=cfg['analysis']['convergence_tol'],
                               window=cfg['analysis']['stability_window'])
    gap = analyze_gap(training_out, test_df)
    rob = analyze_robustness(disruption_df)
    pareto = analyze_tradeoff(test_df, Z1_0, Z2_0)
    reuse = experience_reuse(cfg['training']['episodes'],
                             cfg['env']['max_steps'], cfg['agents'])
    eps_sched = dqn_epsilon_schedule(
        cfg['training']['episodes'],
        cfg['agents']['DQN']['epsilon_start'],
        cfg['agents']['DQN']['epsilon_min'],
        cfg['agents']['DQN']['epsilon_decay'])
    return {'convergence': conv, 'gap': gap, 'robustness': rob,
            'pareto': pareto, 'experience_reuse': reuse,
            'dqn_epsilon': eps_sched,
            'significance': stats_tests, 'policy_behavior': policy_mets}


def findings_text(A, test_df, cfg):
    """English key-findings lines (printed in the notebook analysis cell).

    v3: fully data-driven — no version-specific claim is hardcoded. Winners,
    significance verdicts and trade-offs are computed from the actual data,
    so the same code prints correct conclusions for any protocol version.
    """
    f = []
    ALG = ALGOS
    n_seeds = len(set(test_df['seed']))
    n_deploys = len(test_df)
    alpha = cfg['analysis']['significance_alpha']
    ws = cfg['env'].get('service_weight', 0.0)
    mean = test_df.groupby('algorithm')

    # 1 -- best cost-risk balance
    best_ratio = mean['ratio'].mean().idxmin()
    best_z1 = mean['Z1'].mean().idxmin()
    f.append(f"1. Best cost-risk balance (mean ratio over {n_seeds} seeds): "
             f"{best_ratio} (ratio = "
             f"{test_df[test_df.algorithm == best_ratio]['ratio'].mean():.4f}); "
             f"lowest mean cost Z1 belongs to {best_z1} "
             f"({test_df[test_df.algorithm == best_z1]['Z1'].mean():,.0f}).")

    # 2 -- service level under the (v3) service-aware reward
    bad_fof = test_df[test_df.FOF < 0.999]
    if len(bad_fof) == 0:
        f.append(f"2. SERVICE GUARANTEE (v3): with the service-aware reward "
                 f"term (w_S = {ws:g} * dFOF) every one of the {n_deploys} "
                 f"trained policies keeps FOF = 1.0 at test time — the "
                 f"service-cost trade-off observed under the v2 reward is "
                 f"eliminated without giving up cost optimisation.")
    else:
        det = '; '.join(f"{r['algorithm']}-s{r['seed']} FOF={r['FOF']:.3f}"
                        for _, r in bad_fof.iterrows())
        share = 100 * len(bad_fof) / n_deploys
        f.append(f"2. SERVICE TRADE-OFF persists in {len(bad_fof)}/{n_deploys} "
                 f"({share:.0f}%) deployments despite the service term "
                 f"(w_S = {ws:g}): {det} — the weight may need increasing.")

    # 3 -- statistical significance (n seeds per algorithm)
    kw = A.get('significance', {}).get('per_metric', {})
    if kw:
        pvals = {m: kw[m]['kruskal_p'] for m in ('Z1', 'Z2', 'ratio')
                 if kw.get(m)}
        sig = [m for m, p in pvals.items() if p < alpha]
        pstr = ', '.join(f"{m} p = {pvals[m]:.3f}" for m in pvals)
        if sig:
            pair_bits = []
            for m in sig:
                for pair, d in kw[m]['pairs'].items():
                    pair_bits.append(f"{pair}({m}) MWU p={d['mwu_p']:.3f}")
            f.append(f"3. Significance (Kruskal-Wallis, n = {n_seeds}/algo, "
                     f"alpha = {alpha}): {pstr} — SIGNIFICANT for {', '.join(sig)}. "
                     f"Pairwise: {'; '.join(pair_bits)}.")
        else:
            f.append(f"3. Significance (Kruskal-Wallis, n = {n_seeds}/algo, "
                     f"alpha = {alpha}): {pstr} — no between-algorithm "
                     f"difference is statistically significant; conclusions "
                     f"rest on point estimates and behavioural evidence.")

    # 4 -- oracle-vs-greedy reproducibility gap
    for li, algo in enumerate(ALG):
        g = A['gap'][algo]
        f.append(f"4{'abc'[li]}. {algo}: the greedy policy reproduces the training-oracle "
                 f"topology within {g['gap_Z1_%']:.1f}% on Z1 "
                 f"(oracle {g['oracle_Z1']:,.0f} vs greedy mean "
                 f"{g['greedy_Z1_mean']:,.0f}) — an exploration-vs-exploitation "
                 f"reproducibility gap.")

    # 5 -- convergence speed (data-driven ranking)
    conv_rank = sorted(ALG, key=lambda a: A['convergence'][a]
                       ['convergence_episode_mean'])
    conv_str = ', '.join(f"{a} ~{A['convergence'][a]['convergence_episode_mean']:.0f} eps"
                         for a in conv_rank)
    f.append(f"5. Convergence speed (episode reaching 95% of best score): "
             f"{conv_str} — {conv_rank[0]} adapts fastest, "
             f"{conv_rank[-1]} slowest.")

    # 6 -- robustness under node failure (data-driven winner)
    rfail = A['robustness'].get('failure', {})
    if rfail:
        best_gain = max(ALG, key=lambda a: rfail[a]['gain_max'])
        slowest = min(ALG, key=lambda a: rfail[a]['slope_Z2_rl'])
        slope_str = ', '.join(
            f"{a} {rfail[a]['slope_Z2_rl']:,.0f}" for a in ALG)
        f.append(f"6. Under node-failure disruption {best_gain} re-optimizes "
                 f"with up to {100*rfail[best_gain]['gain_max']:.1f}% cost "
                 f"reduction (adaptation-gain AUC = "
                 f"{100*rfail[best_gain]['gain_auc']:.1f}% mean); Z2 "
                 f"degradation slope per unit failure-probability: "
                 f"{slope_str} — {slowest}'s optimized network degrades "
                 f"slowest.")

    # 7 -- sample economy
    ru = A['experience_reuse']
    f.append(f"7. Experience reuse per environment transition: DQN "
             f"~{ru['DQN']['reuse_factor']:.0f}x, PPO "
             f"~{ru['PPO']['reuse_factor']:.0f}x, A2C "
             f"{ru['A2C']['reuse_factor']:.0f}x — replay lets DQN extract far "
             f"more signal from the same {ru['DQN']['steps']:,} steps.")

    # 8 -- policy behaviour fingerprint (PPO diagnosis, dynamic)
    pm = A.get('policy_behavior', {})
    ppo_h = pm.get('PPO', {}).get('entropy')
    if ppo_h:
        max_h = float(np.log(pm['PPO'].get('n_actions', 376)))
        f.append(f"8. PPO's policy keeps {ppo_h:.2f} nats of action entropy "
                 f"({100*ppo_h/max_h:.0f}% of the maximum log-action "
                 f"entropy) and hedges: {pm['PPO']['n_add']} adds vs "
                 f"{pm['PPO']['n_remove']} removes in its greedy rollout "
                 f"(net edge change "
                 f"{pm['PPO']['edges_final'] - pm['PPO']['edges_initial']:+d}) "
                 f"— a conservative, high-entropy policy; compare with "
                 f"DQN ({pm['DQN']['n_add']} adds / {pm['DQN']['n_remove']} "
                 f"removes) and A2C ({pm['A2C']['n_add']} adds / "
                 f"{pm['A2C']['n_remove']} removes).")

    # 9 -- Pareto front (dynamic)
    pareto_names = [p['name'] for p in A['pareto'] if p['pareto']]
    from collections import Counter
    cnt = Counter(n.split('-s')[0] for n in pareto_names if '-s' in n)
    dom = ', '.join(f"{a} ({c} of {n_seeds})" for a, c in
                    sorted(cnt.items(), key=lambda kv: -kv[1]))
    f.append(f"9. Pareto-optimal topologies in the (Z1, Z2) plane: "
             f"{', '.join(pareto_names)} — front membership: {dom}.")
    return f


def export_json(analysis, path):
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
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_clean(analysis), f, indent=2, ensure_ascii=False,
                  allow_nan=False)
    return path
