"""
Random DAG generator for Stage 6A training data.

Generates structurally diverse but valid timing DAGs with:
- Single source, single sink
- No cycles, all nodes reachable
- 1-3 reconvergence points
- Binary splits
- Path length asymmetry
- Random gate loads
- Spatial coordinates reflecting graph topology
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class GeneratedGate:
    name: str
    load_ff: float
    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class GeneratedGraph:
    graph_id: str
    gates: Dict[str, GeneratedGate]
    successors: Dict[str, List[str]]
    coordinates: Dict[str, Tuple[float, float]]
    gate_loads: Dict[str, float]
    reconvergence_points: List[str]
    source: str
    sink: str


def _assign_coordinates(
    graph: Dict[str, List[str]],
    source: str,
    sink: str,
) -> Dict[str, Tuple[float, float]]:
    """Assign 2D coordinates based on topological level and branch position."""
    # Compute topological levels
    in_degree: Dict[str, int] = {g: 0 for g in graph}
    for g, succs in graph.items():
        for s in succs:
            in_degree[s] += 1

    level: Dict[str, int] = {}
    queue = [source]
    level[source] = 0
    visited = {source}
    while queue:
        g = queue.pop(0)
        for s in graph.get(g, []):
            if s not in visited:
                level[s] = level[g] + 1
                visited.add(s)
                queue.append(s)

    max_level = max(level.values()) if level else 0

    # Group nodes by level
    by_level: Dict[int, List[str]] = {}
    for g, lvl in level.items():
        by_level.setdefault(lvl, []).append(g)

    # Assign coordinates
    coords: Dict[str, Tuple[float, float]] = {}
    for lvl, nodes in by_level.items():
        x = lvl * 2.0
        if len(nodes) == 1:
            coords[nodes[0]] = (x, 0.0)
        else:
            y_spacing = 2.0 / (len(nodes) - 1) if len(nodes) > 1 else 0.0
            start_y = -1.0 if len(nodes) > 1 else 0.0
            for i, node in enumerate(nodes):
                coords[node] = (x, start_y + i * y_spacing)

    return coords


def _validate_graph(graph: Dict[str, List[str]], source: str, sink: str) -> bool:
    """Validate that the graph is a valid DAG with single source/sink."""
    # Check all nodes reachable from source
    visited = set()
    stack = [source]
    while stack:
        g = stack.pop()
        if g in visited:
            continue
        visited.add(g)
        for s in graph.get(g, []):
            if s not in visited:
                stack.append(s)

    if len(visited) != len(graph):
        return False

    # Check topological sort (no cycles)
    in_degree: Dict[str, int] = {g: 0 for g in graph}
    for g, succs in graph.items():
        for s in succs:
            in_degree[s] += 1

    queue = [g for g in graph if in_degree[g] == 0]
    count = 0
    while queue:
        g = queue.pop(0)
        count += 1
        for s in graph.get(g, []):
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    return count == len(graph)


def _find_reconvergence_points(graph: Dict[str, List[str]]) -> List[str]:
    """Find nodes with multiple predecessors (reconvergence points)."""
    pred_count: Dict[str, int] = {}
    for g, succs in graph.items():
        for s in succs:
            pred_count[s] = pred_count.get(s, 0) + 1

    return [g for g, count in pred_count.items() if count > 1]


def _build_dag(
    n_gates: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, List[str]], str, str, Set[str]]:
    """Build a random DAG with exactly n_gates nodes."""
    counter = [0]

    def fresh_name() -> str:
        name = f"n{counter[0]}"
        counter[0] += 1
        return name

    source = fresh_name()
    sink = fresh_name()
    graph: Dict[str, List[str]] = {source: [sink]}
    all_nodes = {source, sink}
    next_idx = 2

    attempts = 0
    while len(all_nodes) < n_gates and attempts < 1000:
        attempts += 1
        edges = [(g, s) for g, succs in graph.items() for s in succs]
        if not edges:
            break

        g, s = edges[int(rng.integers(0, len(edges)))]

        if len(all_nodes) >= n_gates:
            break

        if rng.random() < 0.5:
            # Subdivide: g -> new -> s
            if len(all_nodes) >= n_gates:
                break
            new_node = f"g{next_idx}"
            next_idx += 1
            all_nodes.add(new_node)
            graph[g].remove(s)
            graph[g].append(new_node)
            graph[new_node] = [s]
        else:
            # Split-reconverge: g -> a -> ... -> s, g -> b -> ... -> s
            if len(all_nodes) + 2 > n_gates:
                continue
            a = f"g{next_idx}"
            next_idx += 1
            b = f"g{next_idx}"
            next_idx += 1
            all_nodes.update([a, b])

            path_a = [a]
            path_b = [b]
            len_a = int(rng.integers(1, 4))
            len_b = int(rng.integers(1, 4))

            for _ in range(len_a - 1):
                if len(all_nodes) >= n_gates:
                    break
                node = f"g{next_idx}"
                next_idx += 1
                all_nodes.add(node)
                path_a.append(node)
            for _ in range(len_b - 1):
                if len(all_nodes) >= n_gates:
                    break
                node = f"g{next_idx}"
                next_idx += 1
                all_nodes.add(node)
                path_b.append(node)

            graph[g].remove(s)
            graph[g].append(a)
            prev = a
            for node in path_a[1:]:
                graph[prev] = [node]
                prev = node
            graph[prev] = [s]

            graph[g].append(b)
            prev = b
            for node in path_b[1:]:
                graph[prev] = [node]
                prev = node
            graph[prev] = [s]

        for node in all_nodes:
            if node not in graph:
                graph[node] = []

    return graph, source, sink, all_nodes


def generate_random_dag(
    n_gates: int | None = None,
    graph_id: str | None = None,
    rng: np.random.Generator | None = None,
    min_reconvergence: int = 0,
) -> GeneratedGraph:
    """Generate a random valid timing DAG."""
    if rng is None:
        rng = np.random.default_rng()
    if n_gates is None:
        n_gates = int(rng.integers(4, 13))  # Uniform(4, 12)
    if graph_id is None:
        graph_id = f"graph_{random.randint(0, 999999):06d}"

    existing = getattr(generate_random_dag, "_used_ids", set())
    while graph_id in existing:
        graph_id = f"graph_{random.randint(0, 999999):06d}"
    generate_random_dag._used_ids = existing | {graph_id}

    for attempt in range(50):
        graph, source, sink, all_nodes = _build_dag(n_gates, rng)

        if not _validate_graph(graph, source, sink):
            continue

        reconvergence_points = _find_reconvergence_points(graph)
        if len(reconvergence_points) < min_reconvergence:
            continue

        coords = _assign_coordinates(graph, source, sink)
        gate_loads = {name: float(rng.uniform(0.8, 1.6)) for name in all_nodes}
        gates = {
            name: GeneratedGate(name=name, load_ff=gate_loads[name], x=coords[name][0], y=coords[name][1])
            for name in all_nodes
        }

        return GeneratedGraph(
            graph_id=graph_id,
            gates=gates,
            successors=graph,
            coordinates=coords,
            gate_loads=gate_loads,
            reconvergence_points=reconvergence_points,
            source=source,
            sink=sink,
        )

    raise RuntimeError(f"Failed to generate valid graph {graph_id} with min_reconvergence={min_reconvergence} after 50 attempts")


def generate_dataset(
    n_graphs: int,
    n_gates_range: Tuple[int, int] = (4, 12),
    min_reconvergence: int = 0,
    seed: int = 42,
) -> List[GeneratedGraph]:
    """Generate a dataset of random DAGs."""
    rng = np.random.default_rng(seed)
    random.seed(seed)
    generate_random_dag._used_ids = set()

    graphs = []
    for i in range(n_graphs):
        n_gates = int(rng.integers(n_gates_range[0], n_gates_range[1] + 1))
        graph = generate_random_dag(
            n_gates=n_gates,
            graph_id=f"graph_{i:06d}",
            rng=rng,
            min_reconvergence=min_reconvergence,
        )
        graphs.append(graph)

    return graphs


def print_graph_summary(graph: GeneratedGraph) -> None:
    """Print a human-readable summary of a generated graph."""
    print(f"Graph: {graph.graph_id}")
    print(f"  Gates: {len(graph.gates)}")
    print(f"  Source: {graph.source}, Sink: {graph.sink}")
    print(f"  Reconvergence points: {graph.reconvergence_points}")
    print(f"  Edges: {sum(len(v) for v in graph.successors.values())}")
    print("  Adjacency:")
    for g, succs in graph.successors.items():
        print(f"    {g} -> {succs}")
    print()
