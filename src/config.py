"""
SIoT-RL Supply Chain — Central Configuration (v3)
==================================================
Single source of truth for EVERY key parameter of the models and the
paper/report. The notebook exposes the same dict in its top "CONFIG" cell;
this module keeps the standalone pipeline in sync.

Any parameter changed here propagates to:
  * environment construction (network scale, reward shaping),
  * agent construction   (DQN / PPO / A2C hyper-parameters),
  * experiment protocol  (training budget, seeds, disruption scenarios),
  * results/config.json  (parameter tables printed in the Persian report
                          and the ISI paper are generated from this file).

v3 changes:
  * training seeds expanded 3 -> 10 (statistically valid comparisons);
  * reward gains an explicit SERVICE term  w_S * dFOF  so policies can no
    longer buy cost cuts with unmet demand (fixes the v2 finding R-tradeoff).
"""
import json
import os
import copy

# ---------------------------------------------------------------------------
# CONFIG — edit this dict to re-run the whole study with new settings
# ---------------------------------------------------------------------------
CONFIG = {
    # ---- network ------------------------------------------------------
    "network": {
        "scale": "medium",                    # small | medium | large
        "tier_sizes": {"suppliers": 6, "manufacturers": 8,
                       "distributors": 10, "retailers": 6},
        "demand_range": [50, 150],            # retailer demand ~ U(50,150)
        "edge_conn_frac": [0.4, 0.6],         # per-layer edge sampling ratio
    },
    # ---- SIoT social model --------------------------------------------
    "siot": {
        "trust_init_clip": [0.30, 0.95],      # initial trust clip range
        "trust_gain": 0.03,                   # success: T += 0.03*(1-T)
        "trust_loss": 0.05,                   # failure: T -= 0.05
        "trust_floor": 0.10,
    },
    # ---- RL environment ------------------------------------------------
    "env": {
        "state_size": 9,                      # [FOF,SOF,E,Os,Im,Om,Id,Od,Ir]
        "max_steps": 20,                      # topology edits per episode
        "reward_weights": [0.6, 0.4],         # [w_Z1, w_Z2]
        "service_weight": 5.0,                # v3: w_S in w_S*dFOF service term
        "disconnect_penalty": -10.0,
        "reward_clip": [-25.0, 25.0],
    },
    # ---- agents (identical conditions: same trunk 128-128, gamma, mask) --
    "agents": {
        "shared": {"hidden": [128, 128], "gamma": 0.95},
        "DQN": {
            "lr": 1e-3, "gamma": 0.95,
            "epsilon_start": 1.0, "epsilon_min": 0.05,
            "epsilon_decay": 0.995,
            "buffer_size": 20000, "batch_size": 128,
            "target_update": 10,
        },
        "PPO": {
            "lr": 3e-4, "gamma": 0.95, "clip": 0.2,
            "epochs": 4, "batch_size": 64, "gae_lambda": 0.95,
            "entropy_coef": 0.01, "value_coef": 0.5,
        },
        "A2C": {
            "lr": 5e-4, "gamma": 0.95, "rollout": 32,
            "entropy_coef": 0.01, "value_coef": 0.5,
        },
    },
    # ---- experiment protocol ---------------------------------------------
    "training": {
        "episodes": 150,
        "seeds": list(range(10)),            # v3: 10 independent runs/algo
    },
    # ---- disruption scenarios --------------------------------------------
    "disruption": {
        "stages": 10,
        "failure_step": 0.05,                 # +0.05 per stage (10 -> +0.50)
        "demand_step": 0.10,                  # +10%  per stage (10 -> +100%)
        "affected_frac": 0.3,                 # fraction of nodes disrupted
        "seed": 7,                            # RNG for choosing affected nodes
        "cumulative": True,                   # v2: fixed node set, cumulative
    },
    # ---- analysis (v2) ------------------------------------------------------
    "analysis": {
        "convergence_tol": 0.05,              # within 5% of best = converged
        "stability_window": 30,               # last-N episodes for stability
        "significance_alpha": 0.05,
    },
    # ---- reproducibility / figure settings ---------------------------------
    "runtime": {
        "device": "cpu",
        "torch_seed": None,                   # per-agent seeds come from seeds
        "figure_dpi": 300,
    },
    "version": "3.0",
}


def get(section, key=None, default=None):
    """Safe dotted accessor: get('agents','DQN') -> CONFIG['agents']['DQN']."""
    node = CONFIG.get(section, {})
    if key is None:
        return node
    return node.get(key, default)


def export_json(path):
    """Write CONFIG to disk (consumed by the report / paper generators)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)
    return path


def as_table_rows():
    """Flat (section, parameter, value) rows for document parameter tables."""
    rows = []
    for section, params in CONFIG.items():
        if isinstance(params, dict):
            for k, v in params.items():
                rows.append((section, k, _fmt(v)))
    return rows


def _fmt(v):
    if isinstance(v, list):
        return ", ".join(_fmt(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)
