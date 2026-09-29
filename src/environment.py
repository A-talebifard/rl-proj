"""
SIoT-RL Supply Chain Environment
================================
Social Internet of Things (SIoT) enabled four-tier supply chain modeled as a
directed graph, optimized with Deep Reinforcement Learning (DQN / PPO / A2C).

Tiers (upstream -> downstream, one-directional edges):
    Suppliers -> Manufacturers -> Distributors -> Retailers

The rule-based topology optimizer from the legacy implementation has been
REMOVED. Topology decisions are made exclusively by RL agents. The PEP
(Path Evaluation Priority) scoring is kept for flow allocation only.

Author: SIoT-RL Project
"""
import os
import copy
import random
import numpy as np
import pandas as pd
import networkx as nx

try:                                    # v2: central CONFIG (optional)
    import config as _cfg
except Exception:                        # standalone / notebook fallback
    _cfg = None

# ---------------------------------------------------------------------------
# Configuration (v2: defaults mirror config.py CONFIG exactly, so previously
# trained checkpoints remain fully compatible)
# ---------------------------------------------------------------------------
TIER_SIZES = {              # 'medium' scale network
    'suppliers': 6,
    'manufacturers': 8,
    'distributors': 10,
    'retailers': 6,
}
LAYER_ORDER = ['suppliers', 'manufacturers', 'distributors', 'retailers']
LAYER_PAIRS = [
    ('suppliers', 'manufacturers'),
    ('manufacturers', 'distributors'),
    ('distributors', 'retailers'),
]
LAYER_Y = {'suppliers': 3, 'manufacturers': 2, 'distributors': 1, 'retailers': 0}
LAYER_LABEL = {
    'suppliers': 'Supplier', 'manufacturers': 'Manufacturer',
    'distributors': 'Distributor', 'retailers': 'Retailer',
}
STATE_SIZE = 9              # [FOF, SOF, E_norm, Os, Im, Om, Id, Od, Ir]


# ---------------------------------------------------------------------------
# Supply chain network with an SIoT (Social IoT) overlay
# ---------------------------------------------------------------------------
class SupplyChainGraph:
    """Four-tier supply chain network with SIoT social overlay.

    Every node is an IoT-enabled smart object. Social relationships between
    objects are described by a *trust matrix* T (N x N) that evolves with the
    success/failure of interactions, and a network-level *confidence*
    parameter (mean trust over active relations).
    """

    def __init__(self, csv_path, scale='medium', seed=0):
        self.rng = np.random.RandomState(seed)
        self.pyrng = random.Random(seed)
        self.scale = scale
        self.G = nx.DiGraph()

        sizes = dict(TIER_SIZES)
        if scale == 'small':
            sizes = {'suppliers': 3, 'manufacturers': 4,
                     'distributors': 4, 'retailers': 3}
        elif scale == 'large':
            sizes = {'suppliers': 10, 'manufacturers': 15,
                     'distributors': 18, 'retailers': 12}

        self.num = sizes
        self.total_nodes = sum(sizes.values())

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
        self.csv_data = pd.read_csv(csv_path)

        self._build_layers()
        self._build_nodes()
        # SIoT overlay must exist before edges are created
        n = self.total_nodes
        self.social_trust = np.zeros((n, n))
        self.social_interaction = np.zeros((n, n))
        self._build_edges()

        self.demand = {r: float(self.rng.uniform(50, 150))
                       for r in self.layers['retailers']}
        self.original_failure_probs = {
            n: self.G.nodes[n]['failure_prob'] for n in self.G.nodes()}
        self.original_demand = dict(self.demand)
        self.normalize_costs()
        self.allocate_flow()

    # -- construction -------------------------------------------------------
    def _build_layers(self):
        self.layers, start = {}, 0
        for layer in LAYER_ORDER:
            self.layers[layer] = list(range(start, start + self.num[layer]))
            start += self.num[layer]

    def _build_nodes(self):
        df = self.csv_data
        sampled = df.sample(n=self.total_nodes, replace=True,
                            random_state=self.rng).reset_index(drop=True)
        idx = 0
        for layer in LAYER_ORDER:
            for node in self.layers[layer]:
                row = sampled.iloc[idx]
                idx += 1
                p_i = float(np.clip(row['disruption_likelihood_score'], 0.01, 0.5))
                rel = float(np.clip(row['supplier_reliability_score'], 0.05, 0.99))
                self.G.add_node(node, layer=layer, failure_prob=p_i,
                                reliability=rel)

    def _build_edges(self):
        costs = self.csv_data['shipping_costs'].dropna().values
        for src_layer, dst_layer in LAYER_PAIRS:
            for src in self.layers[src_layer]:
                n_conn = max(2, int(len(self.layers[dst_layer]) *
                                    self.rng.uniform(0.4, 0.6)))
                targets = self.pyrng.sample(
                    self.layers[dst_layer],
                    min(n_conn, len(self.layers[dst_layer])))
                for dst in targets:
                    c_ij = float(self.rng.choice(costs)) if len(costs) else \
                        float(self.rng.uniform(100, 1000))
                    self.add_edge(src, dst, transport_cost=c_ij)

    def add_edge(self, src, dst, transport_cost=None):
        """Add a directed upstream->downstream edge with SIoT attributes."""
        c_ij = transport_cost if transport_cost is not None \
            else float(self.rng.uniform(50, 500))
        c_q = float(self.rng.uniform(10, 100))          # connection cost
        w_pa = float(self.rng.uniform(100, 500))        # flow capacity
        self.G.add_edge(src, dst, transport_cost=c_ij, connection_cost=c_q,
                        capacity=w_pa, flow=0.0)
        # SIoT: initial social trust from object reliabilities
        base = (self.G.nodes[src].get('reliability', 0.7) +
                self.G.nodes[dst].get('reliability', 0.7)) / 2.0
        self.social_trust[src, dst] = float(np.clip(
            base + self.rng.uniform(-0.1, 0.1), 0.30, 0.95))
        self.social_interaction[src, dst] = 1

    def _init_social_overlay(self):
        n = self.total_nodes
        self.social_trust = np.zeros((n, n))
        self.social_interaction = np.zeros((n, n))
        for u, v in self.G.edges():
            base = (self.G.nodes[u].get('reliability', 0.7) +
                    self.G.nodes[v].get('reliability', 0.7)) / 2.0
            self.social_trust[u, v] = float(np.clip(
                base + self.rng.uniform(-0.1, 0.1), 0.30, 0.95))
            self.social_interaction[u, v] = 1

    # -- SIoT social dynamics ----------------------------------------------
    def update_social_trust(self, u, v, success=True):
        """SIoT trust evolution: successful interactions reinforce trust."""
        self.social_interaction[u, v] += 1
        t = self.social_trust[u, v]
        if success:
            self.social_trust[u, v] = min(1.0, t + 0.03 * (1.0 - t))
        else:
            self.social_trust[u, v] = max(0.10, t - 0.05)

    @property
    def confidence(self):
        """Network-level SIoT confidence = mean trust over active relations."""
        vals = [self.social_trust[u, v] for u, v in self.G.edges()]
        return float(np.mean(vals)) if vals else 0.5

    # -- helpers -------------------------------------------------------------
    def normalize_costs(self):
        t = [d['transport_cost'] for _, _, d in self.G.edges(data=True)]
        q = [d['connection_cost'] for _, _, d in self.G.edges(data=True)]
        if not t:
            return
        t_min, t_max, q_min, q_max = min(t), max(t), min(q), max(q)
        for _, _, d in self.G.edges(data=True):
            d['norm_transport_cost'] = \
                (d['transport_cost'] - t_min) / (t_max - t_min + 1e-9)
            d['norm_connection_cost'] = \
                (d['connection_cost'] - q_min) / (q_max - q_min + 1e-9)

    def flow_in(self, node):
        return sum(self.G[u][node]['flow'] for u in self.G.predecessors(node))

    def flow_out(self, node):
        return sum(self.G[node][v]['flow'] for v in self.G.successors(node))

    # -- path utilities ------------------------------------------------------
    def all_supply_paths(self):
        """All simple supplier->retailer paths (length-4 tier chains)."""
        paths = []
        for s in self.layers['suppliers']:
            for r in self.layers['retailers']:
                try:
                    paths.extend(nx.all_simple_paths(self.G, s, r, cutoff=5))
                except nx.NetworkXNoPath:
                    pass
        return paths

    def path_score(self, path):
        """PEP - Path Evaluation Priority score (ascending = higher priority).

        V_Pa = sum(norm c_ij)                          transport term
              + (1 - prod(1 - p_i) * prod(T_ij)) q_bar  SIoT risk term
              + sum(norm c_q(ij))                      connection term
        """
        G, cost_sum, conn_sum, trust_prod = self.G, 0.0, 0.0, 1.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            d = G[u][v]
            cost_sum += d.get('norm_transport_cost', 0.5)
            conn_sum += d.get('norm_connection_cost', 0.5)
            trust_prod *= self.social_trust[u, v]
        survival = 1.0
        for node in path:
            survival *= (1.0 - G.nodes[node]['failure_prob'])
        q_bar = conn_sum / (len(path) - 1) if len(path) > 1 else 0.0
        risk = 1.0 - survival * trust_prod          # SIoT-aware path risk
        return cost_sum + risk * q_bar + conn_sum

    def path_capacity(self, path):
        caps = []
        for i in range(len(path) - 1):
            d = self.G[path[i]][path[i + 1]]
            caps.append(max(0.0, d['capacity'] - d['flow']))
        return min(caps) if caps else 0.0

    def allocate_flow(self):
        """Allocate retailer demand over paths sorted by ascending PEP score.

        Demands are served greedily: the cheapest-score path is used up to
        its bottleneck capacity (W_Pa), then the next path, and so on.
        Every traversed relation updates the SIoT trust matrix.
        """
        G = self.G
        for u, v in G.edges():
            G[u][v]['flow'] = 0.0

        for retailer in self.layers['retailers']:
            remaining = self.demand[retailer]
            if remaining <= 0:
                continue
            all_paths = []
            for s in self.layers['suppliers']:
                try:
                    all_paths.extend(
                        nx.all_simple_paths(G, s, retailer, cutoff=5))
                except nx.NetworkXNoPath:
                    pass
            scored = sorted((self.path_score(p), i, p)
                            for i, p in enumerate(all_paths))
            for score, _, path in scored:
                if remaining <= 0:
                    break
                amount = min(remaining, self.path_capacity(path))
                if amount > 0:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        G[u][v]['flow'] += amount
                        self.update_social_trust(u, v, success=True)
                    remaining -= amount

    # -- objectives ----------------------------------------------------------
    def calculate_objectives(self):
        """Z1 = total logistics cost, Z2 = SIoT-aware resiliency risk."""
        Z1 = 0.0
        for _, _, d in self.G.edges(data=True):
            Z1 += d['flow'] * d['transport_cost'] + d['connection_cost']

        Z2 = 0.0
        for path in self.all_supply_paths():
            survival, trust_prod = 1.0, 1.0
            for node in path:
                survival *= (1.0 - self.G.nodes[node]['failure_prob'])
            for i in range(len(path) - 1):
                trust_prod *= self.social_trust[path[i], path[i + 1]]
            risk = 1.0 - survival * trust_prod
            max_flow = max(self.G[path[i]][path[i + 1]]['flow']
                           for i in range(len(path) - 1))
            Z2 += risk * max_flow
        return Z1, Z2

    def fulfillment_rate(self):
        """FOF - fraction of total retailer demand actually satisfied."""
        total_demand = sum(self.demand.values())
        if total_demand <= 0:
            return 1.0
        served = sum(self.flow_in(r) for r in self.layers['retailers'])
        return float(min(1.0, served / total_demand))

    def path_coverage(self):
        """SOF - realized supply paths vs. maximum possible pairs."""
        n_pairs = len(self.layers['suppliers']) * len(self.layers['retailers'])
        n_paths = 0
        for s in self.layers['suppliers']:
            for r in self.layers['retailers']:
                try:
                    n_paths += len(list(nx.all_simple_paths(
                        self.G, s, r, cutoff=5)))
                except nx.NetworkXNoPath:
                    pass
        return float(min(1.0, n_paths / max(n_pairs, 1)))

    # -- state ---------------------------------------------------------------
    def get_state(self):
        """9-parameter state: [FOF, SOF, E_norm, Os, Im, Om, Id, Od, Ir]."""
        self.normalize_costs()
        self.allocate_flow()
        FOF = self.fulfillment_rate()
        SOF = self.path_coverage()
        e_norm = self.G.number_of_edges() / 100.0
        os_ = np.mean([self.flow_out(s) for s in self.layers['suppliers']]) / 100.0
        im = np.mean([self.flow_in(m) for m in self.layers['manufacturers']]) / 100.0
        om = np.mean([self.flow_out(m) for m in self.layers['manufacturers']]) / 100.0
        id_ = np.mean([self.flow_in(d) for d in self.layers['distributors']]) / 100.0
        od = np.mean([self.flow_out(d) for d in self.layers['distributors']]) / 100.0
        ir = np.mean([self.flow_in(r) for r in self.layers['retailers']]) / 100.0
        return np.array([FOF, SOF, e_norm, os_, im, om, id_, od, ir],
                        dtype=np.float32)

    # -- disruptions ---------------------------------------------------------
    def apply_failure_disruption(self, level, num_affected=None):
        """Disruption A: raise failure probability of randomly chosen nodes."""
        nodes = list(self.G.nodes())
        if num_affected is None:
            num_affected = max(2, len(nodes) // 6)
        chosen = self.pyrng.sample(nodes, min(num_affected, len(nodes)))
        for node in chosen:
            base = self.original_failure_probs.get(
                node, self.G.nodes[node]['failure_prob'])
            self.G.nodes[node]['failure_prob'] = min(0.95, base + level)

    def apply_demand_disruption(self, level):
        """Disruption B: multiply retailer demand by (1 + level)."""
        for r in self.layers['retailers']:
            self.demand[r] = self.original_demand[r] * (1.0 + level)

    def reset_disruption(self):
        for node in self.G.nodes():
            self.G.nodes[node]['failure_prob'] = \
                self.original_failure_probs[node]
        self.demand = dict(self.original_demand)

    # -- connectivity --------------------------------------------------------
    def is_fully_connected(self):
        """Every retailer must keep at least one supplier->retailer path."""
        for r in self.layers['retailers']:
            if not any(nx.has_path(self.G, s, r)
                       for s in self.layers['suppliers']):
                return False
        return True

    def capacity_headroom(self):
        """v2 analysis metric: spare capacity on ACTIVE (used) edges.

        headroom = 1 - sum(flow)/sum(capacity) over edges carrying flow.
        A high headroom means the network can absorb demand surges
        without opening new (more expensive) routes.
        """
        used_flow = used_cap = 0.0
        for u, v, d in self.G.edges(data=True):
            if d['flow'] > 0:
                used_flow += d['flow']
                used_cap += d['capacity']
        return float(1.0 - used_flow / used_cap) if used_cap > 0 else 1.0

    def total_demand(self):
        return float(sum(self.demand.values()))

    def snapshot(self):
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Gym-style RL environment on top of the SIoT supply chain
# ---------------------------------------------------------------------------
class SIoTRLEnvironment:
    """RL environment: 9-parameter state, {add, remove} x candidate edges.

    Actions: (op, src, dst) over all legal upstream->downstream pairs.
    Reward (v3, service-aware):
        r = w_Z1 * dZ1/Z1 + w_Z2 * dZ2/Z2 + w_S * dFOF + disconnect_penalty
    where dFOF is the change in the order Fulfillment Order Rate caused by
    the topology edit. The service term (v3) stops the agent from buying
    cost reductions with unmet customer demand. Weights come from the
    central CONFIG (single source of truth).
    """

    def __init__(self, csv_path, scale='medium', seed=0,
                 max_steps=20):
        self.seed = seed
        self.max_steps = max_steps
        self.initial = SupplyChainGraph(csv_path, scale=scale, seed=seed)
        self.scn = None
        self.actions, self.candidates = self._build_action_space()
        self.action_size = len(self.actions)
        self.state_size = STATE_SIZE
        # v3: reward shaping parameters from the central CONFIG
        _e = (_cfg.CONFIG['env'] if _cfg else {})
        _w = _e.get('reward_weights', [0.6, 0.4])
        self._w_z1 = float(_w[0])
        self._w_z2 = float(_w[1])
        self._w_service = float(_e.get('service_weight', 0.0))   # v3
        self._disc_pen = float(_e.get('disconnect_penalty', -10.0))
        self._rclip = tuple(_e.get('reward_clip', [-25.0, 25.0]))
        self._Z1 = self._Z2 = 0.0
        self._FOF = 1.0
        self._steps = 0
        self.reset()

    def _build_action_space(self):
        actions, candidates = [], []
        scn = self.initial
        for src_layer, dst_layer in LAYER_PAIRS:
            for src in scn.layers[src_layer]:
                for dst in scn.layers[dst_layer]:
                    candidates.append((src, dst))
                    actions.append(('add', src, dst))
                    actions.append(('remove', src, dst))
        return actions, candidates

    # -- RL API ---------------------------------------------------------------
    def reset(self, scn=None):
        self.scn = scn.snapshot() if scn is not None else self.initial.snapshot()
        self._steps = 0
        self.scn.normalize_costs()
        self.scn.allocate_flow()
        self._Z1, self._Z2 = self.scn.calculate_objectives()
        self._FOF = self.scn.fulfillment_rate()          # v3 service baseline
        return self._observe()

    def _observe(self):
        return self.scn.get_state()

    def action_mask(self):
        """True where the action is executable (edge state matches op)."""
        mask = np.zeros(self.action_size, dtype=bool)
        G = self.scn.G
        for i, (op, src, dst) in enumerate(self.actions):
            if op == 'add':
                mask[i] = not G.has_edge(src, dst)
            else:
                mask[i] = G.has_edge(src, dst)
        return mask

    def step(self, action_idx):
        op, src, dst = self.actions[action_idx]
        G = self.scn.G
        executed, disconnected = False, False

        if op == 'add' and not G.has_edge(src, dst):
            self.scn.add_edge(src, dst)
            executed = True
        elif op == 'remove' and G.has_edge(src, dst):
            data = G[src][dst].copy()
            trust = self.scn.social_trust[src, dst]
            inter = self.scn.social_interaction[src, dst]
            G.remove_edge(src, dst)
            if not self.scn.is_fully_connected():
                # connectivity guard -> revert + penalty
                G.add_edge(src, dst, **data)
                self.scn.social_trust[src, dst] = trust
                self.scn.social_interaction[src, dst] = inter
                disconnected = True
            else:
                executed = True
                # SIoT: dissolved relation loses trust
                self.scn.social_trust[src, dst] = 0.0
                self.scn.social_interaction[src, dst] = 0

        self.scn.normalize_costs()
        self.scn.allocate_flow()
        Z1_new, Z2_new = self.scn.calculate_objectives()
        FOF_new = self.scn.fulfillment_rate()            # v3

        cost_imp = (self._Z1 - Z1_new) / max(abs(self._Z1), 1.0)
        res_imp = (self._Z2 - Z2_new) / max(abs(self._Z2), 1.0)
        service_imp = FOF_new - self._FOF                # v3 (+1 restore/-1 drop)
        reward = (self._w_z1 * cost_imp + self._w_z2 * res_imp
                  + self._w_service * service_imp)        # v3 service-aware
        if disconnected:
            reward += self._disc_pen
        reward = float(np.clip(reward, self._rclip[0], self._rclip[1]))

        self._Z1, self._Z2 = Z1_new, Z2_new
        self._FOF = FOF_new                              # v3
        self._steps += 1
        done = self._steps >= self.max_steps
        return self._observe(), reward, done, {
            'Z1': Z1_new, 'Z2': Z2_new, 'executed': executed,
            'disconnected': disconnected, 'FOF': FOF_new,
            'confidence': self.scn.confidence}

    def evaluate(self):
        """Static evaluation of the current topology (no RL)."""
        self.scn.normalize_costs()
        self.scn.allocate_flow()
        Z1, Z2 = self.scn.calculate_objectives()
        return {'Z1': Z1, 'Z2': Z2,
                'ratio': Z2 / Z1 if Z1 else 0.0,
                'FOF': self.scn.fulfillment_rate(),
                'edges': self.scn.G.number_of_edges(),
                'confidence': self.scn.confidence,
                'headroom': self.scn.capacity_headroom()}  # v2 metric
