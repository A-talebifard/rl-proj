"""
English-language publication figures for the SIoT-RL supply chain project.

Style: low-saturation colorblind-safe palette (Paul Tol), constrained
layout, zero-overlap labels, legends outside plot areas, 300 DPI export.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx

from environment import LAYER_ORDER, LAYER_Y, LAYER_LABEL

COLORS = {'DQN': '#0077BB', 'PPO': '#EE7733', 'A2C': '#009988'}
ALGOS = ['DQN', 'PPO', 'A2C']
GRAY = '#6B7280'
LIGHT = '#D1D5DB'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.15,
    'grid.linewidth': 0.6,
    'legend.frameon': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})


def _smooth(x, k=9):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    return pd.Series(x).rolling(k, min_periods=1, center=True).mean().values


def _band(mean, std):
    return mean - std, mean + std


# ---------------------------------------------------------------------------
# 1. Network topology (initial vs optimized)
# ---------------------------------------------------------------------------
def plot_topology(scn, title, path, highlight=None):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    G = scn.G
    pos = {}
    for layer in LAYER_ORDER:
        nodes = scn.layers[layer]
        y = LAYER_Y[layer]
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (i - (len(nodes) - 1) / 2, y)
    layer_color = {'suppliers': '#0077BB', 'manufacturers': '#009988',
                   'distributors': '#EE7733', 'retailers': '#CC3311'}
    node_colors, sizes = [], []
    for n, d in G.nodes(data=True):
        layer = d['layer']
        node_colors.append(layer_color[layer])
        sizes.append(420 + 2600 * d['failure_prob'])
    flow = [G[u][v]['flow'] for u, v in G.edges()]
    fmax = max(flow) if flow else 1.0
    widths = [0.4 + 3.2 * f / max(fmax, 1e-9) for f in flow]
    edge_colors = ['#94A3B8' if f == 0 else '#334155' for f in flow]

    nx.draw_networkx_edges(G, pos, ax=ax, width=widths,
                           edge_color=edge_colors, alpha=0.55,
                           arrows=True, arrowsize=9,
                           connectionstyle='arc3,rad=0.04')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=sizes, edgecolors='white',
                           linewidths=1.2, alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.5, font_color='white')

    for layer in LAYER_ORDER:
        ax.scatter([], [], c=layer_color[layer], s=90,
                   label=f'{LAYER_LABEL[layer]}s ({len(scn.layers[layer])})')
    ax.set_title(title, pad=12, fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.16), ncol=4,
              fontsize=9.5)
    ax.set_axis_off()
    ax.margins(0.06)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


# ---------------------------------------------------------------------------
# 1b. Topology comparison across all algorithms (side-by-side panels)
# ---------------------------------------------------------------------------
def plot_topology_comparison(optimized_topologies, path, algos=None):
    algos = list(algos or ALGOS)
    fig, axes = plt.subplots(1, len(algos), figsize=(16.5, 6.0))
    axes = np.atleast_1d(axes)
    layer_color = {'suppliers': '#0077BB', 'manufacturers': '#009988',
                   'distributors': '#EE7733', 'retailers': '#CC3311'}
    scn = None
    for ax, algo in zip(axes, algos):
        scn = optimized_topologies[algo]['snapshot']
        G = scn.G
        pos = {}
        for layer in LAYER_ORDER:
            nodes = scn.layers[layer]
            y = LAYER_Y[layer]
            for i, node in enumerate(sorted(nodes)):
                pos[node] = (i - (len(nodes) - 1) / 2, y)
        node_colors = [layer_color[d['layer']] for _, d in G.nodes(data=True)]
        sizes = [420 + 2600 * d['failure_prob']
                 for _, d in G.nodes(data=True)]
        flow = [G[u][v]['flow'] for u, v in G.edges()]
        fmax = max(flow) if flow else 1.0
        widths = [0.4 + 3.2 * f / max(fmax, 1e-9) for f in flow]
        edge_colors = ['#94A3B8' if f == 0 else '#334155' for f in flow]
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths,
                               edge_color=edge_colors, alpha=0.55,
                               arrows=True, arrowsize=7,
                               connectionstyle='arc3,rad=0.04')
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=sizes, edgecolors='white',
                               linewidths=1.0, alpha=0.95)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=6.5,
                                font_color='white')
        ax.set_title(f'{algo}', pad=8, fontweight='bold', fontsize=12)
        ax.set_axis_off()
        ax.margins(0.06)
    handles = [Patch(facecolor=layer_color[l],
                     label=f'{LAYER_LABEL[l]}s ({len(scn.layers[l])})')
               for l in LAYER_ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Optimized SIoT-RL networks — all algorithms (seed-0)',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.10,
                        wspace=0.06)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


# ---------------------------------------------------------------------------
# 2. Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(training_out, path, smooth=9):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                             constrained_layout=True)
    ax = axes[0]
    for algo in ALGOS:
        runs = [h['reward'] for h in training_out[algo]['histories']]
        L = min(len(r) for r in runs)
        arr = np.array([r[:L] for r in runs])
        mean, std = arr.mean(0), arr.std(0)
        x = np.arange(L)
        lo, hi = _band(mean, std)
        ax.plot(x, _smooth(mean, smooth), color=COLORS[algo], lw=2.2,
                label=algo)
        ax.fill_between(x, _smooth(lo, smooth), _smooth(hi, smooth),
                        color=COLORS[algo], alpha=0.15, lw=0)
    ax.axhline(0, color=GRAY, lw=0.8, ls='--', alpha=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode reward (smoothed)')
    ax.set_title('Training reward convergence')

    ax = axes[1]
    for algo in ALGOS:
        runs = [h['loss'] for h in training_out[algo]['histories']]
        arr = np.array([[np.nan if v is None else v for v in r] for r in runs])
        mean = np.nanmean(arr, axis=0)
        ax.plot(np.arange(len(mean)), _smooth(mean, smooth),
                color=COLORS[algo], lw=2.2, label=algo)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss (smoothed)')
    ax.set_yscale('log')
    ax.set_title('Optimization loss')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3,
               fontsize=10.5)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


# ---------------------------------------------------------------------------
# 3. Objective convergence
# ---------------------------------------------------------------------------
def plot_objective_convergence(training_out, path, Z1_0, Z2_0, smooth=9):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                             constrained_layout=True)
    for ax, key, base, name in [
            (axes[0], 'Z1', Z1_0, 'Total cost $Z_1$ (relative)'),
            (axes[1], 'Z2', Z2_0, 'Resiliency risk $Z_2$ (relative)')]:
        for algo in ALGOS:
            runs = [h[key] for h in training_out[algo]['histories']]
            L = min(len(r) for r in runs)
            arr = np.array([r[:L] for r in runs])
            mean, std = arr.mean(0), arr.std(0)
            x = np.arange(L)
            ax.plot(x, _smooth(mean / base, smooth), color=COLORS[algo],
                    lw=2.2, label=algo)
            lo, hi = _band(mean / base, std / base)
            ax.fill_between(x, _smooth(lo, smooth), _smooth(hi, smooth),
                            color=COLORS[algo], alpha=0.15, lw=0)
        ax.axhline(1.0, color=GRAY, lw=0.9, ls='--', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel(f'{name}')
        ax.set_title(f'{name} during training')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3,
               fontsize=10.5)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


# ---------------------------------------------------------------------------
# 4. Final algorithm comparison
# ---------------------------------------------------------------------------
def plot_algorithm_comparison(test_metrics, path):
    """test_metrics: DataFrame[algorithm, Z1, Z2, ratio, FOF, edges]."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0),
                             constrained_layout=True)
    panels = [('Z1', 'Total cost $Z_1$', '{:.2e}'),
              ('Z2', 'Resiliency risk $Z_2$', '{:.2e}'),
              ('ratio', 'Risk-to-cost ratio', '{:.3f}'),
              ('FOF', 'Fulfillment rate', '{:.3f}')]
    for ax, (col, label, fmt) in zip(axes, panels):
        vals = [test_metrics.loc[a, col] for a in ALGOS]
        best = np.argmin(vals) if col != 'FOF' else np.argmax(vals)
        colors = [COLORS[a] if i == best else LIGHT
                  for i, a in enumerate(ALGOS)]
        bars = ax.bar(ALGOS, vals, color=colors, width=0.62,
                      edgecolor='white', linewidth=1.0)
        top = max(vals)
        ax.set_ylim(0, top * 1.18)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                        ha='center', va='bottom', fontsize=9.5,
                        xytext=(0, 3), textcoords='offset points')
        ax.set_title(label, fontsize=11.5)
        ax.grid(axis='x', visible=False)
    axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    fig.suptitle('Test-phase performance of optimized SIoT-RL topologies',
                 fontweight='bold')
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


# ---------------------------------------------------------------------------
# 5-6. Disruption scenarios
# ---------------------------------------------------------------------------
def plot_disruption(disruption_df, disruption='failure', path='fig.png',
                    step=0.05):
    df = disruption_df[disruption_df['disruption'] == disruption]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3),
                             constrained_layout=True)
    if disruption == 'failure':
        xlab = 'Failure probability increase'
        supt = 'Robustness under progressive node-failure disruption'
    else:
        xlab = 'Demand increase'
        supt = 'Robustness under progressive demand disruption'

    for ax, (col, label, fmt) in zip(axes, [
            ('Z1_rl', 'Total cost $Z_1$', None),
            ('Z2_rl', 'Resiliency risk $Z_2$', None),
            ('ratio_rl', 'Risk-to-cost ratio', None)]):
        for algo in ALGOS:
            d = df[df['algorithm'] == algo].sort_values('stage')
            ax.plot(d['level'], d[col], color=COLORS[algo], lw=2.2,
                    marker='o', ms=4.5, label=algo)
        d0 = df[df['algorithm'] == 'DQN'].sort_values('stage')
        static_line, = ax.plot(
            d0['level'], d0['Z1_static' if col == 'Z1_rl' else
                      'Z2_static' if col == 'Z2_rl' else 'ratio_static'],
            color=GRAY, lw=1.6, ls='--', alpha=0.85,
            label='Static (no adaptation)')
        ax.set_xlabel(xlab)
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11.5)
    handles, labels = axes[0].get_legend_handles_labels()
    if 'Static (no adaptation)' not in labels:
        handles.append(static_line)
        labels.append('Static (no adaptation)')
    fig.legend(handles, labels, loc='outside lower center', ncol=4,
               fontsize=10)
    fig.suptitle(supt, fontweight='bold')
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


# ---------------------------------------------------------------------------
# 7. SIoT trust matrix
# ---------------------------------------------------------------------------
def plot_trust_matrix(scn, path):
    T = scn.social_trust.copy()
    n = T.shape[0]
    fig, ax = plt.subplots(figsize=(7.6, 6.4), constrained_layout=True)
    masked = np.ma.masked_where(T == 0, T)
    im = ax.imshow(masked, cmap='Blues', vmin=0, vmax=1,
                   aspect='equal')
    # tier separators
    cum = 0
    ticks, labels = [], []
    for layer in LAYER_ORDER:
        size = len(scn.layers[layer])
        cum += size
        ticks.append(cum - size / 2)
        labels.append(f'{LAYER_LABEL[layer]}s')
        if cum < n:
            ax.axhline(cum - 0.5, color='#374151', lw=1.1, alpha=0.7)
            ax.axvline(cum - 0.5, color='#374151', lw=1.1, alpha=0.7)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9.5, rotation=0)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_title('SIoT social trust matrix $T_{ij}$ (active relations)',
                 pad=12, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
    cbar.set_label('Trust value', fontsize=10)
    ax.grid(visible=False)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


# ---------------------------------------------------------------------------
# 8. PEP path scores & allocation
# ---------------------------------------------------------------------------
def plot_pep_scores(scn, path):
    paths = scn.all_supply_paths()
    scores = np.array([scn.path_score(p) for p in paths])
    order = np.argsort(scores)
    ranks = np.arange(1, len(scores) + 1)
    # normalized rank for readability
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                             constrained_layout=True)
    ax = axes[0]
    ax.hist(scores, bins=28, color='#0077BB', alpha=0.75,
            edgecolor='white')
    med = np.median(scores)
    ax.axvline(med, color='#CC3311', lw=1.8, ls='--',
               label=f'median = {med:.2f}')
    ax.set_xlabel('PEP score $V_{Pa}$')
    ax.set_ylabel('Number of paths')
    ax.set_title('PEP path-score distribution')

    ax = axes[1]
    ax.plot(ranks / len(scores), scores[order], color='#009988', lw=2.2)
    ax.fill_between(ranks / len(scores), 0, scores[order],
                    color='#009988', alpha=0.15)
    ax.set_xlabel('Path priority rank (normalized)')
    ax.set_ylabel('PEP score $V_{Pa}$')
    ax.set_title('Ascending priority ordering (lower = served first)')
    ax.set_xlim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2,
               fontsize=10.5)
    fig.suptitle('Path Evaluation Priority (PEP) over '
                 f'{len(scores)} supply paths', fontweight='bold')
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


# ---------------------------------------------------------------------------
# 9. Adaptation gain
# ---------------------------------------------------------------------------
def plot_adaptation(disruption_df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                             constrained_layout=True)
    for ax, disr, label in [(axes[0], 'failure', 'Node-failure disruption'),
                            (axes[1], 'demand', 'Demand disruption')]:
        for algo in ALGOS:
            d = disruption_df[(disruption_df['disruption'] == disr) &
                              (disruption_df['algorithm'] == algo)]
            d = d.sort_values('stage')
            ax.plot(d['level'], 100 * d['adaptation_gain'],
                    color=COLORS[algo], lw=2.2, marker='o', ms=4.5,
                    label=algo)
        ax.axhline(0, color=GRAY, lw=0.9, ls='--', alpha=0.7)
        ax.set_xlabel('Disruption level')
        ax.set_ylabel('Cost reduction vs static (%)')
        ax.set_title(f'{label}: SIoT-RL adaptation gain')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3,
               fontsize=10.5)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


# ---------------------------------------------------------------------------
# 11. v2 — Z1/Z2 trade-off with Pareto front
# ---------------------------------------------------------------------------
def plot_tradeoff(pareto_pts, path):
    """Scatter of candidate topologies in the (Z1, Z2) plane + Pareto front."""
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    init = [p for p in pareto_pts if p['algo'] == 'Initial'][0]
    ax.scatter(init['Z1'], init['Z2'], marker='s', s=150, c=GRAY,
               zorder=3, label='Initial network')
    for algo in ALGOS:
        pts = [p for p in pareto_pts if p['algo'] == algo]
        x = [p['Z1'] for p in pts]
        y = [p['Z2'] for p in pts]
        ax.scatter(x, y, s=95, color=COLORS[algo], zorder=3, label=algo,
                   edgecolor='white', linewidth=0.8)
        for p in pts:
            if p['pareto']:
                ax.annotate(p['name'], (p['Z1'], p['Z2']),
                            xytext=(6, 5), textcoords='offset points',
                            fontsize=8.5, color=COLORS[algo])
    # Pareto front among ALL points (staircase, both objectives minimized)
    pts = sorted(pareto_pts, key=lambda p: p['Z1'])
    front = []
    best_z2 = np.inf
    for p in pts:
        if p['Z2'] < best_z2:
            front.append(p)
            best_z2 = p['Z2']
    fx = [p['Z1'] for p in front]
    fy = [p['Z2'] for p in front]
    ax.step(fx, fy, where='post', color='#CC3311', lw=1.6, ls='--',
            alpha=0.85, zorder=2, label='Pareto front')
    ax.set_xlabel('Total cost $Z_1$')
    ax.set_ylabel('Resiliency risk $Z_2$')
    ax.set_title('Cost-risk trade-off of candidate topologies '
                 '(3 seeds per algorithm)', fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=9.5)
    ax.grid(alpha=0.2)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


# ---------------------------------------------------------------------------
# 12. v2 — policy behavior fingerprint
# ---------------------------------------------------------------------------
def plot_policy_behavior(policy_mets, path):
    """What the trained agents actually DO: action mix, tier focus,
    exploration appetite (entropy / epsilon)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2),
                             constrained_layout=True)
    # (a) executed action mix
    ax = axes[0]
    adds = [policy_mets[a]['n_add'] for a in ALGOS]
    rms = [policy_mets[a]['n_remove'] for a in ALGOS]
    skip = [policy_mets[a]['n_skipped'] for a in ALGOS]
    x = np.arange(len(ALGOS))
    ax.bar(x, adds, 0.58, label='add edge', color='#009988')
    ax.bar(x, rms, 0.58, bottom=adds, label='remove edge', color='#CC3311')
    ax.bar(x, skip, 0.58, bottom=np.array(adds) + np.array(rms),
           label='skipped / reverted', color='#D1D5DB')
    for i, a in enumerate(ALGOS):
        tot = adds[i] + rms[i] + skip[i]
        ax.annotate(f'{tot} steps', (i, tot), ha='center', va='bottom',
                    fontsize=9, xytext=(0, 2), textcoords='offset points')
    ax.set_xticks(x, ALGOS)
    ax.set_ylabel('Greedy-rollout actions')
    ax.set_title('(a) Executed action mix', fontsize=11.5)
    ax.legend(fontsize=9)

    # (b) per-tier net edge change
    ax = axes[1]
    tier_keys = ['suppliers->manufacturers', 'manufacturers->distributors',
                 'distributors->retailers']
    tier_lbl = ['Sup→Man', 'Man→Dis', 'Dis→Ret']
    w = 0.26
    for i, algo in enumerate(ALGOS):
        vals = [policy_mets[algo]['tier_delta'].get(k, 0)
                for k in tier_keys]
        ax.bar(np.arange(3) + (i - 1) * w, vals, w, color=COLORS[algo],
               label=algo)
    ax.axhline(0, color=GRAY, lw=0.9)
    ax.set_xticks(range(3), tier_lbl)
    ax.set_ylabel('Net edge change (final − initial)')
    ax.set_title('(b) Tier-level topology change', fontsize=11.5)
    ax.legend(fontsize=9)

    # (c) exploration appetite at deployment
    ax = axes[2]
    ent = [policy_mets[a]['entropy'] for a in ALGOS]
    eps = [policy_mets[a]['epsilon'] for a in ALGOS]
    xs = np.arange(len(ALGOS))
    for i, a in enumerate(ALGOS):
        if ent[i] is not None and not (isinstance(ent[i], float)
                                       and np.isnan(ent[i])):
            ax.bar(i, ent[i], 0.5, color=COLORS[a])
            ax.annotate(f'H = {ent[i]:.1f}', (i, ent[i]), ha='center',
                        va='bottom', fontsize=9.5, xytext=(0, 2),
                        textcoords='offset points')
        else:  # DQN deterministic -> show its epsilon instead
            ax.bar(i, eps[i], 0.5, color=COLORS[a], alpha=0.65,
                   hatch='//')
            ax.annotate(f'ε = {eps[i]:.2f}', (i, eps[i]), ha='center',
                        va='bottom', fontsize=9.5, xytext=(0, 2),
                        textcoords='offset points')
    ax.set_xticks(xs, ALGOS)
    ax.set_ylabel('Action entropy H / epsilon')
    ax.set_title('(c) Exploration appetite at deployment', fontsize=11.5)
    fig.suptitle('Policy-behavior fingerprint of the trained agents',
                 fontweight='bold')
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


# ---------------------------------------------------------------------------
# 13. v2 — multi-seed test dispersion + significance
# ---------------------------------------------------------------------------
def plot_multiseed_test(test_df, path):
    """Per-seed test metrics: box + individual seed points per algorithm."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4),
                             constrained_layout=True)
    panels = [('Z1', 'Total cost $Z_1$'), ('Z2', 'Resiliency risk $Z_2$'),
              ('ratio', 'Risk-to-cost ratio')]
    for ax, (col, label) in zip(axes, panels):
        data = [test_df[test_df.algorithm == a][col].values for a in ALGOS]
        bp = ax.boxplot(data, tick_labels=ALGOS, widths=0.5, patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                       markeredgecolor='none', markersize=5))
        for patch, algo in zip(bp['boxes'], ALGOS):
            patch.set_facecolor(COLORS[algo])
            patch.set_alpha(0.35)
        rng = np.random.RandomState(0)
        for i, a in enumerate(ALGOS):
            vals = test_df[test_df.algorithm == a][col].values
            jitter = rng.uniform(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), i + 1) + jitter, vals, s=42,
                       color=COLORS[a], zorder=3, edgecolor='white',
                       linewidth=0.7)
            ax.annotate(f'μ={np.mean(vals):.3g}',
                        (i + 1, np.mean(vals)), ha='center', va='bottom',
                        fontsize=8.5, xytext=(0, 8),
                        textcoords='offset points', color='#374151')
        if col != 'ratio':
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set_title(label, fontsize=11.5)
        ax.set_ylabel(label if col == 'ratio' else '')
    fig.suptitle('Test-phase dispersion across independent training seeds '
                 f'(n = {len(set(test_df["seed"]))} per algorithm)',
                 fontweight='bold')
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)


# ---------------------------------------------------------------------------
# 10. SIoT confidence evolution during training
# ---------------------------------------------------------------------------
def plot_confidence_evolution(training_out, path, smooth=9):
    fig, ax = plt.subplots(figsize=(8.6, 4.4), constrained_layout=True)
    for algo in ALGOS:
        runs = [h['confidence'] for h in training_out[algo]['histories']]
        L = min(len(r) for r in runs)
        arr = np.array([r[:L] for r in runs])
        mean, std = arr.mean(0), arr.std(0)
        x = np.arange(L)
        ax.plot(x, _smooth(mean, smooth), color=COLORS[algo], lw=2.2,
                label=algo)
        lo, hi = _band(mean, std)
        ax.fill_between(x, _smooth(lo, smooth), _smooth(hi, smooth),
                        color=COLORS[algo], alpha=0.15, lw=0)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Network confidence (mean SIoT trust)')
    ax.set_title('SIoT social confidence evolution during training')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3,
               fontsize=10.5)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}", flush=True)
