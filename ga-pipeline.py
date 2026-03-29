"""
By: JetHayes

Find the minimum spanning tree where Node 1 = Delivery Point and Nodes 2-9 = Well Heads

I'll be using the Prüfer sequences appraoach, which will evaluate the total pipeline length as the fitness function.

"""

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from itertools import combinations
import copy

# === Graph Definition ===

# I am extracting the edges from the problem figure with the format (node_i, node_j, distance)
edges = [
    (1, 2, 5),
    (1, 4, 20),
    (1, 5, 4),
    (1, 8, 14),
    (2, 3, 6),
    (2, 5, 7),
    (3, 4, 15),
    (3, 5, 10),
    (4, 5, 20),
    (4, 6, 7),
    (4, 7, 12),
    (5, 6, 3),
    (5, 7, 5),
    (5, 8, 13),
    (5, 9, 6),
    (7, 8, 7),
    (8, 9, 5),
]
# Next, de-duping edges to ensure we have a unique set of edges
_edge_dict = {}
for u, v, w in edges:
    # Ensure (u, v) and (v, u) are treated the same
    key = (min(u, v), max(u, v))
    if key not in _edge_dict or w < _edge_dict[key]:
        _edge_dict[key] = w
EDGES = [(u, v, w) for (u, v), w in _edge_dict.items()]
NODES = list(range(1, 10))  # Nodes 1-9
N = len(NODES)

# Building ajacency, which is a nested dictionary to instantly lookup any distance between the connected nodes for a quick lookup
ADJ = {}
for u, v, w in EDGES:
    ADJ.setdefault(u, {})[v] = w
    ADJ.setdefault(v, {})[u] = w

# === Spanning Tree Utulity (Prüfer Sequence) ===


def prufer_to_tree(sequence):
    """Convert a Prüfer sequence to a tree represented as an edge list."""
    nodes = list(range(1, N + 1))
    degree = [1] * (N + 1)
    for node in sequence:
        degree[node] += 1

    edges = []
    seq = list(sequence)
    for node in seq:
        for leaf in nodes:
            if degree[leaf] == 1:
                edges.append((leaf, node))
                degree[leaf] -= 1
                degree[node] -= 1
                break
    # last edge = the two remaining nodes with degree 1
    remaining = [n for n in nodes if degree[n] == 1]
    edges.append((remaining[0], remaining[1]))
    return edges


def tree_cost(edges):
    """
    total pipeline cost of a spanning tree.
    Returns infinity if an edge does not exist in the graph.
    """
    total = 0
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if u in ADJ and v in ADJ[u]:
            total += ADJ[u][v]
        else:
            return float('inf')  # Edge does not exist, return infinity
    return total


def random_prufer():
    """Generate a random Prüfer sequence of length N-2."""
    return [random.randint(1, N) for _ in range(N - 2)]


def is_feasible(sequence):
    """Check if a Prüfer sequence corresponds to a valid spanning tree."""
    edges = prufer_to_tree(sequence)
    return tree_cost(edges) < float('inf')

#  === Genetic Algorithm ===

# tournament selection


def tournament_selection(population, fitnesses, k=3):
    """Select an individual from the population using tournament selection."""
    candidates = random.sample(list(zip(population, fitnesses)), k)
    winner = min(candidates, key=lambda x: x[1])  # minimize cost
    return winner[0]
# two-point crossover


def two_point_crossover(parent1, parent2):
    """Perform two-point crossover between two parents."""
    if len(parent1) < 2:
        # No crossover if sequence is too short
        return list(parent1), list(parent2)
    pt1, pt2 = sorted(random.sample(range(len(parent1)), 2))
    child1 = parent1[:pt1] + parent2[pt1:pt2] + parent1[pt2:]
    child2 = parent2[:pt1] + parent1[pt1:pt2] + parent2[pt2:]
    return child1, child2

# random reset mutation


def mutate(individual, mutation_rate=0.15):
    """Randomly mutate an individual with a given mutation rate."""
    return [random.randint(1, N) if random.random() < mutation_rate else g for g in individual]

# repair - replace infeasible edges with alternatives


def repair(individual):
    """Repair an individual to ensure it corresponds to a valid spanning tree."""
    for _ in range(200):
        if is_feasible(individual):
            return individual
        idx = random.randint(0, len(individual) - 1)
        individual[idx] = random.randint(1, N)
    return individual  # Return the best effort after max attempts

# === Main Genetic Algorithm  ===


def genetic_algorithm(
        pop_size=100,
        generations=300,
        crossover_rate=0.85,
        mutation_rate=0.15,
        elite_size=5,
        tournament_k=3,
        seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Initialize population
    population = []
    attempts = 0
    while len(population) < pop_size:
        ind = random_prufer()
        ind = repair(ind)
        if is_feasible(ind):
            population.append(ind)
        attempts += 1
        if attempts > pop_size * 50:  # Avoid infinite loop
            break

    # Evaluate fitness
    def evaluate(ind):
        edges = prufer_to_tree(ind)
        return tree_cost(edges)

    fitnesses = [evaluate(ind) for ind in population]

    best_costs = min(fitnesses) if fitnesses else float('inf')
    best_individual = population[fitnesses.index(
        best_costs)] if fitnesses else None

    history = {
        'best': [],
        'avg': [],
        'worst': []
    }

    # generation loop
    for gen in range(generations):
        # recording stats
        valid_fits = [f for f in fitnesses if f < float('inf')]
        history['best'].append(min(fitnesses) if fitnesses else float('inf'))
        history['avg'].append(np.mean(valid_fits)
                              if valid_fits else float('inf'))
        history['worst'].append(
            max(valid_fits) if valid_fits else float('inf'))

        # elite - unchanged best individuals (sorted by fitness)
        sorted_pairs = sorted(zip(population, fitnesses), key=lambda x: x[1])
        new_population = [list(ind) for ind, _ in sorted_pairs[:elite_size]]

        # fill the rest of the new population
        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses, tournament_k)
            p2 = tournament_selection(population, fitnesses, tournament_k)

            # crossover
            if random.random() < crossover_rate:
                c1, c2 = two_point_crossover(p1, p2)
            else:
                c1, c2 = list(p1), list(p2)

            # mutation
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)

            # repair
            c1 = repair(c1)
            c2 = repair(c2)

            new_population.extend([c1, c2])

        # Ensure population size is maintained
        population = new_population[:pop_size]
        fitnesses = [evaluate(ind) for ind in population]

        # update global best
        gen_best = min(fitnesses) if fitnesses else float('inf')
        if gen_best < best_costs:
            best_costs = gen_best
            best_individual = population[fitnesses.index(best_costs)]

    best_edges = prufer_to_tree(
        best_individual) if best_individual is not None else []
    return best_individual, best_costs, best_edges, history

# ===THE PLOT ===

# approximate node positions


NODE_POS = {
    1: (5.0,  5.5),   # delivery point (top centre)
    2: (3.5,  4.5),
    3: (2.0,  3.5),
    4: (3.5,  3.2),
    5: (5.0,  3.2),
    6: (3.3,  2.0),
    7: (5.0,  1.8),
    8: (6.8,  3.0),
    9: (6.8,  4.2),
}


def plot_graph(edges, best_edges, history=None, filename="convergence.png"):
    """Plot the network edges and (optionally) convergence history to the provided filename."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if history and 'best' in history:
        gens = range(len(history['best']))
        ax.plot(gens, history['best'], label='Best fitness',
                color='steelblue', linewidth=2)
        ax.plot(gens, history['avg'], label='Avg  fitness',
                color='orange', linewidth=1.5, linestyle='--')
        ax.set_xlabel('Generation', fontsize=13)
        ax.set_ylabel('Total pipeline length', fontsize=13)
        ax.set_title('GA Convergence – Minimum Pipeline Network',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        if history.get('avg'):
            try:
                ax.set_ylim([0, max(history['avg'][:5]) *
                            1.2 if len(history['avg']) >= 5 else max(history['avg']) * 1.2])
            except Exception:
                ax.set_ylim([0, 200])
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No convergence history available",
                ha='center', va='center')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Convergence plot saved → {filename}")


def plot_convergence(history, best_cost, filename="convergence.png"):
    """Compatibility wrapper used in main to save the convergence plot given history and filename."""
    # Use the generic plot_graph routine to render convergence only.
    plot_graph([], [], history=history, filename=filename)


def plot_network(ga_edges, filename="network.png"):
    """Plot the GA best solution network."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw all feasible edges
    for u, v, w in EDGES:
        x = [NODE_POS[u][0], NODE_POS[v][0]]
        y = [NODE_POS[u][1], NODE_POS[v][1]]
        ax.plot(x, y, color='lightgrey', linewidth=1, zorder=1)
        mx, my = (x[0]+x[1])/2, (y[0]+y[1])/2
        ax.text(mx, my, str(w), fontsize=7, color='grey',
                ha='center', va='center', zorder=2)

    # Draw tree edges
    total = 0
    for u, v in ga_edges:
        w = ADJ[u][v]
        total += w
        x = [NODE_POS[u][0], NODE_POS[v][0]]
        y = [NODE_POS[u][1], NODE_POS[v][1]]
        ax.plot(x, y, color='steelblue', linewidth=3, zorder=3)

    # Draw nodes
    for node, (px, py) in NODE_POS.items():
        color = 'gold' if node == 1 else 'lightblue'
        circle = plt.Circle((px, py), 0.25, color=color,
                            ec='black', lw=1.5, zorder=4)
        ax.add_patch(circle)
        ax.text(px, py, str(node), ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=5)

    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0.5, 6.5)
    ax.set_aspect('equal')
    ax.set_title(
        f'GA – Best Solution\nTotal length = {total}', fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Network plot saved    → {filename}")

# Main thingy


def main():
    print("=" * 60)
    print("  Minimum Pipeline Network – Genetic Algorithm Solution")
    print("=" * 60)

    # ── Run GA ───────────────────────────────────────────────────
    print("\n[GA] Running …")
    best_ind, best_cost, best_edges, history = genetic_algorithm(
        pop_size=120,
        generations=400,
        crossover_rate=0.85,
        mutation_rate=0.15,
        elite_size=5,
        tournament_k=4,
        seed=42
    )

    print(f"\n[GA Result]    Total length = {best_cost}")
    print("  Annotated edges:")
    for u, v in best_edges:
        w = ADJ[u][v] if v in ADJ.get(u, {}) else '???'
        print(f"    {u} ─── {v}  (length {w})")

    # ── Plots ────────────────────────────────────────────────────
    print("\n[Plotting]")
    plot_convergence(history, best_cost,
                     filename="convergence.png")
    plot_network(best_edges,
                 filename="network.png")

    # ── Summary table ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GA PARAMETER SUMMARY & RATIONALE")
    print("=" * 60)
    summary = """
  Encoding        : Prüfer sequence (length N-2 = 7 integers in 1…9)
                    → bijection between sequences and labelled trees
                    → no explicit connectivity constraint needed

  Population size : 120
                    → large enough for diversity on 9-node graph

  Generations     : 400
                    → sufficient for convergence (plateau < 200 gen)

  Selection       : Tournament (k=4)
                    → selective pressure without diversity collapse
                    → no need for fitness scaling

  Crossover       : Two-point on Prüfer sequence  (rate 0.85)
                    → preserves structural building blocks
                    → high rate encourages exploitation

  Mutation        : Random-reset per gene           (rate 0.15)
                    → maintains exploration for short sequences

  Elitism         : Top 5 carried unchanged
                    → guarantees monotone best-fitness improvement

  Constraint      : Infeasible edges repaired by random gene reset
                    (up to 200 attempts) → penalty-free feasibility
"""
    print(summary)


if __name__ == "__main__":
    main()
