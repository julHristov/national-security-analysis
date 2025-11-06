# modules_agents/graph_builder.py
"""
Graph Builder — бърз networkx graph per document
Nodes: actor (normalized), concepts, actions (optionally)
Edges: (actor1 -> actor2) with attributes: weight, confidence, relation_label
"""

import networkx as nx
from pathlib import Path
import json
from config import RESULTS_DIR


def build_graph_from_scenarios(scenarios: list):
    G = nx.DiGraph()
    for s in scenarios:
        a1 = s.get("actor_1")
        for t in s.get("targets", []) or []:
            a2 = t.lower()
            rel = s.get("action") or s.get("action_phrase")
            conf = s.get("verification", {}).get("confidence", 0.5) if isinstance(s.get("verification"), dict) else 0.5
            weight = s.get("weight", 1.0)
            if not G.has_node(a1):
                G.add_node(a1, type="actor")
            if not G.has_node(a2):
                G.add_node(a2, type="entity")
            if G.has_edge(a1, a2):
                G[a1][a2]["weight"] += weight
                G[a1][a2]["confidence"] = max(G[a1][a2].get("confidence", 0.0), conf)
            else:
                G.add_edge(a1, a2, weight=weight, confidence=conf, label=rel)
    return G


def save_graph(G, output_path: Path):
    # simple json export: nodes + edges
    nodes = [{"id": n, **G.nodes[n]} for n in G.nodes()]
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({"source": u, "target": v, **data})
    obj = {"nodes": nodes, "edges": edges}
    output_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
