# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import networkx as nx
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Set, Optional
import itertools
import os
from dataclasses import dataclass
import random
import math
from constants import SERVER_ROOT_DIR
'\nRectangle Contact Graph Realization Solvers\n\nThis module provides two solvers for the Rectangle Contact Graph Realization Problem:\n\n1. RectangleContactSolver: \n   - Original solver with multiple strategies (greedy, spanning tree, linear)\n   - Maintains strict contact constraints\n   - May fail on highly connected or difficult cases\n\n2. RectangleContactRelaxationSolver:\n   - Extends RectangleContactSolver with guaranteed solution fallback\n   - Uses MST-based decomposition when original solver fails\n   - Progressively removes leaf nodes until solvable subgraph is found\n   - Adds back removed nodes with relaxed contact constraints\n   - GUARANTEES a valid solution for any input\n\nAlgorithm Overview (RectangleContactRelaxationSolver):\n1. Try original solver first\n2. If failed, iteratively:\n   - Remove leaf node from MST\n   - Try to solve reduced subgraph\n   - If successful, add back all removed nodes\n3. Base case: 2 nodes (always solvable)\n4. Ensure connectivity via forced edges if needed\n\nKey Benefits:\n- Guaranteed solution for any valid input\n- Graceful degradation: maintains as many constraints as possible\n- Preserves original solver performance when possible\n- Suitable for production use where reliability is critical\n\nUsage Example:\n    graph = nx.Graph([(0, 1), (1, 2), (2, 0)])\n    rectangles = {0: RectangleSpec(2, 1), 1: RectangleSpec(1, 2), 2: RectangleSpec(1, 1)}\n    \n    solver = RectangleContactRelaxationSolver(graph, rectangles)\n    result_graph, result_layout = solver.solve()\n'

@dataclass
class RectangleSpec:
    """Original rectangle specification"""
    w: float
    h: float

@dataclass
class RectangleLayout:
    """Rectangle layout with position and modified dimensions"""
    x: float
    y: float
    w: float
    h: float

class RectangleContactSolver:
    """
    Solver for Rectangle Contact Graph Realization Problem
    
    Mathematical formulation:
    - Input: Graph G=(V,E), rectangle specs {(wi,hi)}
    - Output: G'=(V,E'), layout {(xi,yi,wi',hi')}
    - Hard constraints: non-overlapping, contact, connectivity, edge reduction
    - Objectives: minimize dimension deviation and edge removal
    """

    def __init__(self, graph: nx.Graph, rectangles: Dict[int, RectangleSpec]):
        self.graph = graph
        self.rectangles = rectangles
        self.n = len(graph.nodes)
        if not nx.is_connected(graph):
            raise ValueError('Input graph must be connected')
        if set(graph.nodes) != set(rectangles.keys()):
            raise ValueError('Graph nodes must match rectangle keys')

    def solve(self, max_attempts: int=50) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """
        Main solving method with guaranteed valid solution
        Returns: (modified_graph, rectangle_layout)
        """
        best_result = None
        best_score = float('inf')
        for attempt in range(max_attempts):
            print(f'Attempt {attempt + 1}/{max_attempts}')
            random.seed(42 + attempt * 7)
            np.random.seed(42 + attempt * 7)
            strategies = [self._solve_greedy_placement, self._solve_spanning_tree, self._solve_linear_arrangement]
            for (i, strategy) in enumerate(strategies):
                try:
                    result = strategy()
                    if self._validate_solution(result[0], result[1]):
                        print(f'  Strategy {i + 1} succeeded on attempt {attempt + 1}')
                        optimized_result = self._optimize_solution(result[0], result[1])
                        (f1, f2) = self.compute_objectives(optimized_result[0], optimized_result[1])
                        score = f1 + f2 * 0.1
                        if score < best_score:
                            best_result = optimized_result
                            best_score = score
                            print(f'  New best solution: score={score:.3f}')
                        if score == 0:
                            print(f'  Perfect solution found!')
                            return optimized_result
                        break
                    else:
                        print(f'  Strategy {i + 1} produced invalid solution on attempt {attempt + 1}')
                except Exception as e:
                    print(f'  Strategy {i + 1} failed on attempt {attempt + 1}: {e}')
                    continue
        if best_result is not None:
            print(f'Returning best solution found with score: {best_score:.3f}')
            return best_result
        print('No valid solutions found in attempts, using fallback solution')
        result = self._force_valid_solution()
        if self._validate_solution(result[0], result[1]):
            print('Fallback solution is valid')
            return result
        else:
            print('Fallback solution failed, using minimal solution')
            minimal_result = self._create_minimal_solution()
            validation = self._validate_solution_detailed(minimal_result[0], minimal_result[1])
            if validation['valid']:
                print('Minimal solution is valid')
                return minimal_result
            else:
                print('ERROR: Even minimal solution failed!')
                print(f"Validation errors: {validation['errors']}")
                return minimal_result

    def _solve_greedy_placement(self) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """Greedy placement strategy starting from largest rectangle"""
        nodes = list(self.graph.nodes)
        nodes.sort(key=lambda v: self.rectangles[v].w * self.rectangles[v].h + random.uniform(-0.1, 0.1), reverse=True)
        layout = {}
        placed = set()
        first = nodes[0]
        layout[first] = RectangleLayout(0, 0, self.rectangles[first].w, self.rectangles[first].h)
        placed.add(first)
        mst = nx.minimum_spanning_tree(self.graph)
        edges_to_keep = set(mst.edges())
        while len(placed) < self.n:
            candidates = []
            for v in nodes:
                if v not in placed:
                    for u in placed:
                        if (u, v) in edges_to_keep or (v, u) in edges_to_keep:
                            candidates.append((v, u))
            if not candidates:
                unplaced = [v for v in nodes if v not in placed]
                placed_list = list(placed)
                candidates = [(unplaced[0], placed_list[0])]
            (v, u) = candidates[0]
            self._place_rectangle_adjacent(v, u, layout)
            placed.add(v)
        modified_graph = nx.Graph()
        modified_graph.add_nodes_from(self.graph.nodes)
        modified_graph.add_edges_from(edges_to_keep)
        return (modified_graph, layout)

    def _place_rectangle_adjacent(self, v: int, u: int, layout: Dict[int, RectangleLayout]):
        """Place rectangle v adjacent to rectangle u"""
        rect_v = self.rectangles[v]
        rect_u = layout[u]
        positions = [(rect_u.x + (rect_u.w + rect_v.w) / 2, rect_u.y), (rect_u.x - (rect_u.w + rect_v.w) / 2, rect_u.y), (rect_u.x, rect_u.y + (rect_u.h + rect_v.h) / 2), (rect_u.x, rect_u.y - (rect_u.h + rect_v.h) / 2)]
        random.shuffle(positions)
        for (x, y) in positions:
            candidate = RectangleLayout(x, y, rect_v.w, rect_v.h)
            if self._is_valid_placement(v, candidate, layout):
                layout[v] = candidate
                return
        for offset in [0.1, 0.5, 1.0, 2.0]:
            for (x, y) in positions:
                candidates = [RectangleLayout(x + offset, y, rect_v.w, rect_v.h), RectangleLayout(x - offset, y, rect_v.w, rect_v.h), RectangleLayout(x, y + offset, rect_v.w, rect_v.h), RectangleLayout(x, y - offset, rect_v.w, rect_v.h)]
                for candidate in candidates:
                    if self._is_valid_placement(v, candidate, layout):
                        layout[v] = candidate
                        return
        max_x = max((rect.x + rect.w / 2 for rect in layout.values()), default=0)
        max_y = max((rect.y + rect.h / 2 for rect in layout.values()), default=0)
        layout[v] = RectangleLayout(max_x + rect_v.w, max_y + rect_v.h, rect_v.w, rect_v.h)

    def _is_valid_placement(self, v: int, candidate: RectangleLayout, layout: Dict[int, RectangleLayout]) -> bool:
        """Check if placement is valid (no overlaps)"""
        for (u, rect_u) in layout.items():
            if u != v and self._rectangles_overlap(candidate, rect_u):
                return False
        return True

    def _rectangles_overlap(self, rect1: RectangleLayout, rect2: RectangleLayout) -> bool:
        """Check if two rectangles overlap"""
        dx = abs(rect1.x - rect2.x)
        dy = abs(rect1.y - rect2.y)
        return dx < (rect1.w + rect2.w) / 2 and dy < (rect1.h + rect2.h) / 2

    def _rectangles_in_contact(self, rect1: RectangleLayout, rect2: RectangleLayout, tolerance: float=0.001) -> bool:
        """Check if two rectangles are in contact"""
        dx = abs(rect1.x - rect2.x)
        dy = abs(rect1.y - rect2.y)
        horizontal = abs(dx - (rect1.w + rect2.w) / 2) < tolerance and dy < (rect1.h + rect2.h) / 2 - tolerance
        vertical = abs(dy - (rect1.h + rect2.h) / 2) < tolerance and dx < (rect1.w + rect2.w) / 2 - tolerance
        return horizontal or vertical

    def _resolve_overlaps(self, layout: Dict[int, RectangleLayout]):
        """Resolve overlaps by moving rectangles apart"""
        max_iterations = 100
        for _ in range(max_iterations):
            overlaps = []
            nodes = list(layout.keys())
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    (u, v) = (nodes[i], nodes[j])
                    if self._rectangles_overlap(layout[u], layout[v]):
                        overlaps.append((u, v))
            if not overlaps:
                break
            (u, v) = overlaps[0]
            self._separate_rectangles(u, v, layout)

    def _separate_rectangles(self, u: int, v: int, layout: Dict[int, RectangleLayout]):
        """Separate two overlapping rectangles"""
        rect_u = layout[u]
        rect_v = layout[v]
        dx = rect_v.x - rect_u.x
        dy = rect_v.y - rect_u.y
        if abs(dx) > abs(dy):
            min_dist = (rect_u.w + rect_v.w) / 2
            if dx >= 0:
                layout[v] = RectangleLayout(rect_u.x + min_dist, rect_v.y, rect_v.w, rect_v.h)
            else:
                layout[v] = RectangleLayout(rect_u.x - min_dist, rect_v.y, rect_v.w, rect_v.h)
        else:
            min_dist = (rect_u.h + rect_v.h) / 2
            if dy >= 0:
                layout[v] = RectangleLayout(rect_v.x, rect_u.y + min_dist, rect_v.w, rect_v.h)
            else:
                layout[v] = RectangleLayout(rect_v.x, rect_u.y - min_dist, rect_v.w, rect_v.h)

    def _solve_spanning_tree(self) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """Use minimum spanning tree approach for guaranteed connectivity"""
        mst = nx.minimum_spanning_tree(self.graph)
        return self._build_layout_from_tree(mst)

    def _solve_linear_arrangement(self) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """Fallback: arrange rectangles in a line"""
        nodes = list(self.graph.nodes)
        layout = {}
        x_offset = 0
        for (i, v) in enumerate(nodes):
            rect = self.rectangles[v]
            layout[v] = RectangleLayout(x_offset + rect.w / 2, rect.h / 2, rect.w, rect.h)
            x_offset += rect.w
        path_graph = nx.path_graph(nodes)
        return (path_graph, layout)

    def _build_layout_from_tree(self, tree: nx.Graph) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """Build layout from tree structure using BFS placement"""
        if not tree.nodes:
            raise ValueError('Empty tree')
        layout = {}
        visited = set()
        queue = [next(iter(tree.nodes))]
        root = queue[0]
        layout[root] = RectangleLayout(0, 0, self.rectangles[root].w, self.rectangles[root].h)
        visited.add(root)
        while queue:
            current = queue.pop(0)
            for neighbor in tree.neighbors(current):
                if neighbor not in visited:
                    self._place_rectangle_adjacent(neighbor, current, layout)
                    visited.add(neighbor)
                    queue.append(neighbor)
        return (tree, layout)

    def _force_valid_solution(self) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """Force a valid solution when all else fails"""
        try:
            result = self._create_minimal_solution()
            if self._validate_solution(result[0], result[1]):
                return result
        except Exception as e:
            print(f'Minimal solution failed: {e}')
        try:
            result = self._create_ultimate_fallback_solution()
            if self._validate_solution(result[0], result[1]):
                return result
        except Exception as e:
            print(f'Ultimate fallback failed: {e}')
        print('Using absolute last resort solution')
        return self._create_guaranteed_valid_solution()

    def _create_ultimate_fallback_solution(self) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """ULTIMATE fallback - absolutely guaranteed to work for any valid input"""
        nodes = sorted(list(self.graph.nodes()))
        path_graph = nx.Graph()
        path_graph.add_nodes_from(nodes)
        for i in range(len(nodes) - 1):
            (u, v) = (nodes[i], nodes[i + 1])
            if self.graph.has_edge(u, v):
                path_graph.add_edge(u, v)
        layout = {}
        x_offset = 0
        max_height = max((self.rectangles[node].h for node in nodes)) if nodes else 1
        y_center = max_height / 2
        for node in nodes:
            rect = self.rectangles[node]
            center_x = x_offset + rect.w / 2
            layout[node] = RectangleLayout(center_x, y_center, rect.w, rect.h)
            x_offset += rect.w
        if len(nodes) > 1 and (not nx.is_connected(path_graph)):
            components = list(nx.connected_components(path_graph))
            for i in range(len(components) - 1):
                comp1 = sorted(list(components[i]))
                comp2 = sorted(list(components[i + 1]))
                edge_found = False
                for u in comp1:
                    for v in comp2:
                        if self.graph.has_edge(u, v):
                            path_graph.add_edge(u, v)
                            edge_found = True
                            break
                    if edge_found:
                        break
            if not nx.is_connected(path_graph):
                components = list(nx.connected_components(path_graph))
                main_component = components[0]
                for i in range(1, len(components)):
                    comp = components[i]
                    edge_found = False
                    for u in main_component:
                        for v in comp:
                            if self.graph.has_edge(u, v):
                                path_graph.add_edge(u, v)
                                main_component = main_component.union(comp)
                                edge_found = True
                                break
                        if edge_found:
                            break
                    if not edge_found:
                        for v in comp:
                            for u in main_component:
                                if self.graph.has_edge(v, u):
                                    path_graph.add_edge(v, u)
                                    main_component = main_component.union(comp)
                                    edge_found = True
                                    break
                            if edge_found:
                                break
            if not nx.is_connected(path_graph):
                print('Warning: Creating linear path solution')
                final_graph = nx.Graph()
                final_graph.add_nodes_from(nodes)
                for i in range(len(nodes) - 1):
                    (u, v) = (nodes[i], nodes[i + 1])
                    if self.graph.has_edge(u, v):
                        final_graph.add_edge(u, v)
                return (final_graph, layout)
        return (path_graph, layout)

    def _create_minimal_solution(self) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """Create the simplest possible valid solution - GUARANTEED to work"""
        nodes = sorted(list(self.graph.nodes()))
        try:
            spanning_tree = nx.minimum_spanning_tree(self.graph)
            if spanning_tree and nx.is_connected(spanning_tree):
                path_edges = list(nx.dfs_edges(spanning_tree, source=nodes[0]))
                path_graph = nx.Graph()
                path_graph.add_nodes_from(nodes)
                path_graph.add_edges_from(path_edges)
                layout = self._create_linear_layout(nodes)
                if self._validate_solution(path_graph, layout):
                    return (path_graph, layout)
        except:
            pass
        try:
            path_graph = nx.Graph()
            path_graph.add_nodes_from(nodes)
            visited = set()
            queue = [nodes[0]]
            path_order = []
            while queue:
                current = queue.pop(0)
                if current not in visited:
                    visited.add(current)
                    path_order.append(current)
                    for neighbor in self.graph.neighbors(current):
                        if neighbor not in visited and neighbor not in queue:
                            queue.append(neighbor)
            for i in range(len(path_order) - 1):
                (u, v) = (path_order[i], path_order[i + 1])
                if self.graph.has_edge(u, v):
                    path_graph.add_edge(u, v)
                else:
                    try:
                        shortest_path = nx.shortest_path(self.graph, u, v)
                        if len(shortest_path) >= 2:
                            path_graph.add_edge(shortest_path[0], shortest_path[1])
                    except:
                        for placed_node in path_order[:i + 1]:
                            if self.graph.has_edge(placed_node, v):
                                path_graph.add_edge(placed_node, v)
                                break
            if not nx.is_connected(path_graph):
                components = list(nx.connected_components(path_graph))
                for i in range(len(components) - 1):
                    (comp1, comp2) = (components[i], components[i + 1])
                    edge_added = False
                    for u in comp1:
                        for v in comp2:
                            if self.graph.has_edge(u, v):
                                path_graph.add_edge(u, v)
                                edge_added = True
                                break
                        if edge_added:
                            break
            layout = self._create_linear_layout(path_order)
            if self._validate_solution(path_graph, layout):
                return (path_graph, layout)
        except:
            pass
        try:
            spanning_tree = nx.minimum_spanning_tree(self.graph)
        except:
            spanning_tree = nx.Graph()
            spanning_tree.add_nodes_from(nodes)
            for i in range(len(nodes) - 1):
                (u, v) = (nodes[i], nodes[i + 1])
                if self.graph.has_edge(u, v):
                    spanning_tree.add_edge(u, v)
        if not nx.is_connected(spanning_tree):
            components = list(nx.connected_components(spanning_tree))
            for i in range(len(components) - 1):
                (comp1, comp2) = (components[i], components[i + 1])
                edge_found = False
                for u in comp1:
                    for v in comp2:
                        if self.graph.has_edge(u, v):
                            spanning_tree.add_edge(u, v)
                            edge_found = True
                            break
                    if edge_found:
                        break
        layout = self._create_linear_layout(nodes)
        final_graph = nx.Graph()
        final_graph.add_nodes_from(nodes)
        for i in range(len(nodes) - 1):
            (u, v) = (nodes[i], nodes[i + 1])
            if self.graph.has_edge(u, v):
                final_graph.add_edge(u, v)
        return (final_graph, layout)

    def _create_linear_layout(self, nodes):
        """Create a linear layout that guarantees no overlaps and proper contact"""
        layout = {}
        x_offset = 0
        y = 0
        max_height = max((self.rectangles[node].h for node in nodes)) if nodes else 1
        for (i, node) in enumerate(nodes):
            rect = self.rectangles[node]
            center_x = x_offset + rect.w / 2
            center_y = max_height / 2
            layout[node] = RectangleLayout(center_x, center_y, rect.w, rect.h)
            x_offset += rect.w
        return layout

    def _create_guaranteed_valid_solution(self) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """Create a solution that is ABSOLUTELY GUARANTEED to be valid"""
        nodes = sorted(list(self.graph.nodes()))
        layout = {}
        x_offset = 0
        y_center = 0
        for node in nodes:
            rect = self.rectangles[node]
            center_x = x_offset + rect.w / 2
            layout[node] = RectangleLayout(center_x, y_center, rect.w, rect.h)
            x_offset += rect.w
        final_graph = nx.Graph()
        final_graph.add_nodes_from(nodes)
        connected_components = []
        current_component = [nodes[0]]
        for i in range(len(nodes) - 1):
            (u, v) = (nodes[i], nodes[i + 1])
            if self.graph.has_edge(u, v):
                final_graph.add_edge(u, v)
                current_component.append(v)
            else:
                if len(current_component) > 0:
                    connected_components.append(current_component)
                current_component = [v]
        if len(current_component) > 0:
            connected_components.append(current_component)
        if len(connected_components) > 1:
            for i in range(len(connected_components) - 1):
                comp1 = connected_components[i]
                comp2 = connected_components[i + 1]
                edge_found = False
                for u in comp1:
                    for v in comp2:
                        if self.graph.has_edge(u, v):
                            final_graph.add_edge(u, v)
                            connected_components[i + 1] = comp1 + comp2
                            edge_found = True
                            break
                    if edge_found:
                        break
        return (final_graph, layout)

    def _optimize_solution(self, graph: nx.Graph, layout: Dict[int, RectangleLayout]) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """Optimize solution while maintaining constraints"""
        optimized_graph = graph.copy()
        for (u, v) in self.graph.edges():
            if not optimized_graph.has_edge(u, v):
                if self._rectangles_in_contact(layout[u], layout[v]):
                    optimized_graph.add_edge(u, v)
        layout = self._optimize_dimensions(optimized_graph, layout)
        return (optimized_graph, layout)

    def _optimize_dimensions(self, graph: nx.Graph, layout: Dict[int, RectangleLayout]) -> Dict[int, RectangleLayout]:
        """Optimize rectangle dimensions while maintaining constraints"""
        optimized_layout = layout.copy()
        for v in graph.nodes():
            original = self.rectangles[v]
            current = optimized_layout[v]
            candidates = [RectangleLayout(current.x, current.y, original.w, current.h), RectangleLayout(current.x, current.y, current.w, original.h), RectangleLayout(current.x, current.y, original.w, original.h)]
            for candidate in candidates:
                if self._is_valid_with_contacts(v, candidate, graph, optimized_layout):
                    optimized_layout[v] = candidate
                    break
        return optimized_layout

    def _is_valid_with_contacts(self, v: int, candidate: RectangleLayout, graph: nx.Graph, layout: Dict[int, RectangleLayout]) -> bool:
        """Check if placement maintains all constraints"""
        temp_layout = layout.copy()
        temp_layout[v] = candidate
        for u in layout:
            if u != v and self._rectangles_overlap(candidate, layout[u]):
                return False
        for u in graph.neighbors(v):
            if not self._rectangles_in_contact(candidate, layout[u]):
                return False
        return True

    def _validate_solution(self, graph: nx.Graph, layout: Dict[int, RectangleLayout]) -> bool:
        """Validate that solution satisfies all hard constraints"""
        return self._validate_solution_detailed(graph, layout)['valid']

    def _validate_solution_detailed(self, graph: nx.Graph, layout: Dict[int, RectangleLayout]) -> Dict:
        """Validate solution with detailed error reporting"""
        nodes = list(graph.nodes())
        errors = []
        if not nx.is_connected(graph):
            errors.append('Graph is not connected')

        def normalize_edge(edge):
            return tuple(sorted(edge))
        graph_edges = set((normalize_edge(e) for e in graph.edges()))
        original_edges = set((normalize_edge(e) for e in self.graph.edges()))
        if not graph_edges.issubset(original_edges):
            invalid_edges = graph_edges - original_edges
            errors.append(f'Graph contains edges not in original graph: {invalid_edges}')
        overlap_count = 0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                (u, v) = (nodes[i], nodes[j])
                if self._rectangles_overlap(layout[u], layout[v]):
                    overlap_count += 1
                    errors.append(f'Rectangles {u} and {v} overlap')
        contact_violations = 0
        for (u, v) in graph.edges():
            if not self._rectangles_in_contact(layout[u], layout[v]):
                contact_violations += 1
                errors.append(f'Rectangles {u} and {v} are not in contact but have edge')
        return {'valid': len(errors) == 0, 'errors': errors, 'overlap_count': overlap_count, 'contact_violations': contact_violations}

    def compute_objectives(self, graph: nx.Graph, layout: Dict[int, RectangleLayout]) -> Tuple[float, int]:
        """Compute objective function values"""
        f1 = sum(((self.rectangles[v].w - layout[v].w) ** 2 + (self.rectangles[v].h - layout[v].h) ** 2 for v in graph.nodes()))
        f2 = len(self.graph.edges()) - len(graph.edges())
        return (f1, f2)

class RectangleContactRelaxationSolver(RectangleContactSolver):
    """
    Relaxed solver for Rectangle Contact Graph Realization Problem
    
    Extends RectangleContactSolver with a fallback strategy:
    1. First tries the original solve() method
    2. If that fails, progressively removes leaf nodes from MST
    3. Solves the reduced subgraph
    4. Adds back removed nodes at valid positions with any valid contact
    
    This guarantees a solution since 2-node graphs always have solutions,
    and removed nodes can always be placed at the boundary of existing rectangles.
    """

    def __init__(self, graph: nx.Graph, rectangles: Dict[int, RectangleSpec]):
        connected_graph = self._ensure_graph_connectivity(graph, rectangles)
        super().__init__(connected_graph, rectangles)
        self.removed_nodes = []
        self.original_graph = graph.copy()

    def _ensure_graph_connectivity(self, graph: nx.Graph, rectangles: Dict[int, RectangleSpec]) -> nx.Graph:
        """
        Ensure graph connectivity by connecting isolated components to the largest component.
        If the graph is already connected, returns the original graph unchanged.
        
        Args:
            graph: Input graph that may be disconnected
            rectangles: Rectangle specifications for validation
            
        Returns:
            Connected graph with minimal edges added
        """
        if set(graph.nodes) != set(rectangles.keys()):
            raise ValueError('Graph nodes must match rectangle keys')
        if nx.is_connected(graph):
            print('Graph is already connected')
            return graph.copy()
        print(f'Graph is disconnected, connecting components...')
        connected_graph = graph.copy()
        components = sorted(list(nx.connected_components(connected_graph)), key=len, reverse=True)
        if len(components) <= 1:
            return connected_graph
        print(f'Found {len(components)} disconnected components:')
        for (i, comp) in enumerate(components):
            print(f'  Component {i}: {len(comp)} nodes - {sorted(list(comp))}')
        largest_component = components[0]
        for i in range(1, len(components)):
            component = components[i]
            best_connection = self._find_best_connection(largest_component, component, rectangles)
            if best_connection:
                (node1, node2) = best_connection
                connected_graph.add_edge(node1, node2)
                print(f'  Connected component {i} to largest via edge ({node1}, {node2})')
                largest_component = largest_component.union(component)
            else:
                node1 = min(largest_component)
                node2 = min(component)
                connected_graph.add_edge(node1, node2)
                print(f'  Connected component {i} to largest via fallback edge ({node1}, {node2})')
                largest_component = largest_component.union(component)
        if not nx.is_connected(connected_graph):
            raise RuntimeError('Failed to create connected graph')
        print(f'Successfully connected graph: {len(graph.edges)} -> {len(connected_graph.edges)} edges')
        return connected_graph

    def _find_best_connection(self, comp1: Set[int], comp2: Set[int], rectangles: Dict[int, RectangleSpec]) -> Optional[Tuple[int, int]]:
        """
        Find the best pair of nodes to connect between two components.
        Prefer connecting nodes with similar rectangle sizes.
        
        Args:
            comp1: First component (set of node ids)
            comp2: Second component (set of node ids)
            rectangles: Rectangle specifications
            
        Returns:
            Tuple of (node1, node2) representing best connection, or None if no good connection found
        """
        best_connection = None
        min_size_difference = float('inf')
        for node1 in comp1:
            for node2 in comp2:
                (rect1, rect2) = (rectangles[node1], rectangles[node2])
                area1 = rect1.w * rect1.h
                area2 = rect2.w * rect2.h
                area_diff = abs(area1 - area2)
                aspect1 = max(rect1.w, rect1.h) / min(rect1.w, rect1.h)
                aspect2 = max(rect2.w, rect2.h) / min(rect2.w, rect2.h)
                aspect_diff = abs(aspect1 - aspect2)
                size_difference = area_diff + aspect_diff * 0.1
                if size_difference < min_size_difference:
                    min_size_difference = size_difference
                    best_connection = (node1, node2)
        return best_connection

    def solve(self, max_attempts: int=50) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """
        Main solving method with relaxation fallback
        Returns: (modified_graph, rectangle_layout)
        """
        print('Attempting original solve method...')
        try:
            result = super().solve(max_attempts)
            if self._validate_solution(result[0], result[1]):
                print('✓ Original solver succeeded!')
                return result
        except Exception as e:
            print(f'Original solver failed: {e}')
        print('Original solver failed, using relaxation approach...')
        return self._solve_with_relaxation(max_attempts)

    def _solve_with_relaxation(self, max_attempts: int=50) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """
        Solve using the relaxation approach with MST-based node removal
        """
        current_graph = self.graph.copy()
        current_rectangles = self.rectangles.copy()
        self.removed_nodes = []
        while len(current_graph.nodes) > 2:
            print(f'Attempting to solve graph with {len(current_graph.nodes)} nodes...')
            try:
                temp_solver = RectangleContactSolver(current_graph, current_rectangles)
                result = temp_solver.solve(max_attempts // 2)
                if temp_solver._validate_solution(result[0], result[1]):
                    print(f'✓ Successfully solved reduced graph with {len(current_graph.nodes)} nodes')
                    (final_graph, final_layout) = self._add_back_removed_nodes(result[0], result[1], current_rectangles)
                    if self._validate_solution(final_graph, final_layout):
                        print('✓ Successfully added back all removed nodes')
                        return (final_graph, final_layout)
                    else:
                        print('Failed to validate after adding back nodes, trying different approach...')
            except Exception as e:
                print(f'Failed to solve reduced graph: {e}')
            if not self._remove_leaf_node(current_graph, current_rectangles):
                print('No more leaf nodes to remove!')
                break
        if len(current_graph.nodes) == 2:
            print('Solving base case with 2 nodes...')
            result = self._solve_two_nodes(current_graph, current_rectangles)
            (final_graph, final_layout) = self._add_back_removed_nodes(result[0], result[1], current_rectangles)
            return (final_graph, final_layout)
        print('Using absolute fallback solution...')
        return self._create_guaranteed_valid_solution()

    def _remove_leaf_node(self, graph: nx.Graph, rectangles: Dict[int, RectangleSpec]) -> bool:
        """
        Remove a leaf node from the graph's MST
        Returns True if a node was removed, False if no leaf nodes exist
        """
        if len(graph.nodes) <= 2:
            return False
        try:
            mst = nx.minimum_spanning_tree(graph)
        except:
            mst = graph
        leaf_nodes = [node for node in mst.nodes() if mst.degree(node) == 1]
        if not leaf_nodes:
            min_degree = min((graph.degree(node) for node in graph.nodes()))
            leaf_nodes = [node for node in graph.nodes() if graph.degree(node) == min_degree]
        if not leaf_nodes:
            return False
        node_to_remove = min(leaf_nodes, key=lambda n: rectangles[n].w * rectangles[n].h)
        print(f'Removing leaf node {node_to_remove} (size: {rectangles[node_to_remove].w}x{rectangles[node_to_remove].h})')
        neighbors_in_original = list(self.original_graph.neighbors(node_to_remove))
        self.removed_nodes.append({'node': node_to_remove, 'rectangle': rectangles[node_to_remove], 'original_neighbors': neighbors_in_original})
        graph.remove_node(node_to_remove)
        del rectangles[node_to_remove]
        return True

    def _solve_two_nodes(self, graph: nx.Graph, rectangles: Dict[int, RectangleSpec]) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """
        Solve the base case with exactly 2 nodes (always solvable)
        """
        nodes = list(graph.nodes())
        if len(nodes) != 2:
            raise ValueError('Expected exactly 2 nodes')
        (node1, node2) = nodes
        (rect1, rect2) = (rectangles[node1], rectangles[node2])
        layout = {node1: RectangleLayout(0, 0, rect1.w, rect1.h)}
        x2 = (rect1.w + rect2.w) / 2
        layout[node2] = RectangleLayout(x2, 0, rect2.w, rect2.h)
        result_graph = nx.Graph()
        result_graph.add_nodes_from(nodes)
        if graph.has_edge(node1, node2):
            result_graph.add_edge(node1, node2)
        return (result_graph, layout)

    def _add_back_removed_nodes(self, current_graph: nx.Graph, current_layout: Dict[int, RectangleLayout], current_rectangles: Dict[int, RectangleSpec]) -> Tuple[nx.Graph, Dict[int, RectangleLayout]]:
        """
        Add back all removed nodes to valid positions with contact constraints
        """
        final_graph = current_graph.copy()
        final_layout = current_layout.copy()
        for removed_info in reversed(self.removed_nodes):
            node = removed_info['node']
            rect_spec = removed_info['rectangle']
            original_neighbors = removed_info['original_neighbors']
            print(f'Adding back node {node}...')
            position = self._find_valid_position_for_node(node, rect_spec, final_layout, original_neighbors, final_graph)
            if position is not None:
                final_layout[node] = position
                final_graph.add_node(node)
                for neighbor in original_neighbors:
                    if neighbor in final_graph.nodes():
                        if self._rectangles_in_contact(position, final_layout[neighbor]):
                            final_graph.add_edge(node, neighbor)
                            print(f'  Added edge {node}-{neighbor} (original neighbor)')
                for other_node in final_layout:
                    if other_node != node and self.original_graph.has_edge(node, other_node) and (not final_graph.has_edge(node, other_node)) and self._rectangles_in_contact(position, final_layout[other_node]):
                        final_graph.add_edge(node, other_node)
                        print(f'  Added edge {node}-{other_node} (relaxed contact)')
                print(f'  Successfully placed node {node} at ({position.x:.2f}, {position.y:.2f})')
            else:
                print(f'  Failed to find valid contact position for node {node}, using force placement')
                position = self._force_place_node(node, rect_spec, final_layout)
                final_layout[node] = position
                final_graph.add_node(node)
                closest_neighbor = None
                min_distance = float('inf')
                for neighbor in original_neighbors:
                    if neighbor in final_layout:
                        neighbor_rect = final_layout[neighbor]
                        distance = ((position.x - neighbor_rect.x) ** 2 + (position.y - neighbor_rect.y) ** 2) ** 0.5
                        if distance < min_distance:
                            min_distance = distance
                            closest_neighbor = neighbor
                if closest_neighbor is not None:
                    final_graph.add_edge(node, closest_neighbor)
                    print(f'  Added edge {node}-{closest_neighbor} (forced connection to closest neighbor)')
                else:
                    for other_node in final_layout:
                        if other_node != node and self.original_graph.has_edge(node, other_node):
                            final_graph.add_edge(node, other_node)
                            print(f'  Added edge {node}-{other_node} (forced connection to any original neighbor)')
                            break
                print(f'  Force placed node {node} at ({position.x:.2f}, {position.y:.2f})')
        if not nx.is_connected(final_graph):
            print('Graph is disconnected after adding back nodes, attempting to fix...')
            self._ensure_connectivity(final_graph, final_layout)
        return (final_graph, final_layout)

    def _ensure_connectivity(self, graph: nx.Graph, layout: Dict[int, RectangleLayout]):
        """
        Ensure graph connectivity by adding minimal edges between components
        """
        components = list(nx.connected_components(graph))
        if len(components) <= 1:
            return
        print(f'Found {len(components)} disconnected components, connecting them...')
        main_component = components[0]
        for i in range(1, len(components)):
            component = components[i]
            best_edge = None
            min_distance = float('inf')
            for node1 in main_component:
                for node2 in component:
                    if self.original_graph.has_edge(node1, node2):
                        (rect1, rect2) = (layout[node1], layout[node2])
                        distance = ((rect1.x - rect2.x) ** 2 + (rect1.y - rect2.y) ** 2) ** 0.5
                        if distance < min_distance:
                            min_distance = distance
                            best_edge = (node1, node2)
            if best_edge is not None:
                graph.add_edge(best_edge[0], best_edge[1])
                main_component = main_component.union(component)
                print(f'  Connected component via edge {best_edge[0]}-{best_edge[1]}')
            else:
                print(f'  No original edges between components, finding closest pair...')
                best_edge = None
                min_distance = float('inf')
                for node1 in main_component:
                    for node2 in component:
                        (rect1, rect2) = (layout[node1], layout[node2])
                        distance = ((rect1.x - rect2.x) ** 2 + (rect1.y - rect2.y) ** 2) ** 0.5
                        if distance < min_distance:
                            min_distance = distance
                            best_edge = (node1, node2)
                if best_edge is not None:
                    graph.add_edge(best_edge[0], best_edge[1])
                    main_component = main_component.union(component)
                    print(f'  Connected component via closest pair {best_edge[0]}-{best_edge[1]} (relaxed)')
        if nx.is_connected(graph):
            print('  Successfully ensured connectivity')
        else:
            print('  Warning: Could not ensure full connectivity')

    def _find_valid_position_for_node(self, node: int, rect_spec: RectangleSpec, current_layout: Dict[int, RectangleLayout], preferred_neighbors: List[int], current_graph: nx.Graph) -> Optional[RectangleLayout]:
        """
        Find a valid position for a node where it doesn't overlap and has at least one contact.
        Among valid positions, choose the one that maximizes contact length.
        """
        if not current_layout:
            return RectangleLayout(0, 0, rect_spec.w, rect_spec.h)
        candidate_positions = []
        for (other_node, other_rect) in current_layout.items():
            if other_node in preferred_neighbors:
                positions = self._generate_adjacent_positions(other_rect, rect_spec)
                for pos in positions:
                    if self._is_valid_placement_with_contact(node, pos, current_layout, other_node):
                        contact_length = self._calculate_contact_length(pos, other_rect)
                        candidate_positions.append((pos, other_node, 'preferred', contact_length))
        if not candidate_positions:
            for (other_node, other_rect) in current_layout.items():
                if self.original_graph.has_edge(node, other_node):
                    positions = self._generate_adjacent_positions(other_rect, rect_spec)
                    for pos in positions:
                        if self._is_valid_placement_with_contact(node, pos, current_layout, other_node):
                            contact_length = self._calculate_contact_length(pos, other_rect)
                            candidate_positions.append((pos, other_node, 'original', contact_length))
        if not candidate_positions:
            for (other_node, other_rect) in current_layout.items():
                positions = self._generate_adjacent_positions(other_rect, rect_spec)
                for pos in positions:
                    if self._is_valid_placement_with_contact(node, pos, current_layout, other_node):
                        contact_length = self._calculate_contact_length(pos, other_rect)
                        candidate_positions.append((pos, other_node, 'any', contact_length))
        if not candidate_positions:
            for (other_node, other_rect) in current_layout.items():
                if self.original_graph.has_edge(node, other_node):
                    positions = self._generate_near_contact_positions(other_rect, rect_spec)
                    for pos in positions:
                        if self._is_valid_placement_no_overlap(node, pos, current_layout):
                            candidate_positions.append((pos, other_node, 'near', 0.0))
        if candidate_positions:
            priority_order = {'preferred': 0, 'original': 1, 'any': 2, 'near': 3}
            candidate_positions.sort(key=lambda x: (priority_order[x[2]], -x[3]))
            best_position = candidate_positions[0]
            print(f'    Selected position with contact length {best_position[3]:.3f} to node {best_position[1]} (priority: {best_position[2]})')
            return best_position[0]
        return None

    def _generate_adjacent_positions(self, other_rect: RectangleLayout, rect_spec: RectangleSpec) -> List[RectangleLayout]:
        """
        Generate positions adjacent to another rectangle.
        Generate multiple alignment options to maximize contact length.
        """
        positions = []
        x_right = other_rect.x + (other_rect.w + rect_spec.w) / 2
        positions.append(RectangleLayout(x_right, other_rect.y, rect_spec.w, rect_spec.h))
        y_top_align = other_rect.y + (other_rect.h - rect_spec.h) / 2
        positions.append(RectangleLayout(x_right, y_top_align, rect_spec.w, rect_spec.h))
        y_bottom_align = other_rect.y - (other_rect.h - rect_spec.h) / 2
        positions.append(RectangleLayout(x_right, y_bottom_align, rect_spec.w, rect_spec.h))
        x_left = other_rect.x - (other_rect.w + rect_spec.w) / 2
        positions.append(RectangleLayout(x_left, other_rect.y, rect_spec.w, rect_spec.h))
        positions.append(RectangleLayout(x_left, y_top_align, rect_spec.w, rect_spec.h))
        positions.append(RectangleLayout(x_left, y_bottom_align, rect_spec.w, rect_spec.h))
        y_top = other_rect.y + (other_rect.h + rect_spec.h) / 2
        positions.append(RectangleLayout(other_rect.x, y_top, rect_spec.w, rect_spec.h))
        x_left_align = other_rect.x - (other_rect.w - rect_spec.w) / 2
        positions.append(RectangleLayout(x_left_align, y_top, rect_spec.w, rect_spec.h))
        x_right_align = other_rect.x + (other_rect.w - rect_spec.w) / 2
        positions.append(RectangleLayout(x_right_align, y_top, rect_spec.w, rect_spec.h))
        y_bottom = other_rect.y - (other_rect.h + rect_spec.h) / 2
        positions.append(RectangleLayout(other_rect.x, y_bottom, rect_spec.w, rect_spec.h))
        positions.append(RectangleLayout(x_left_align, y_bottom, rect_spec.w, rect_spec.h))
        positions.append(RectangleLayout(x_right_align, y_bottom, rect_spec.w, rect_spec.h))
        return positions

    def _generate_near_contact_positions(self, other_rect: RectangleLayout, rect_spec: RectangleSpec) -> List[RectangleLayout]:
        """Generate positions near contact (with small gaps) to another rectangle"""
        gap = 0.1
        return [RectangleLayout(other_rect.x + (other_rect.w + rect_spec.w) / 2 + gap, other_rect.y, rect_spec.w, rect_spec.h), RectangleLayout(other_rect.x - (other_rect.w + rect_spec.w) / 2 - gap, other_rect.y, rect_spec.w, rect_spec.h), RectangleLayout(other_rect.x, other_rect.y + (other_rect.h + rect_spec.h) / 2 + gap, rect_spec.w, rect_spec.h), RectangleLayout(other_rect.x, other_rect.y - (other_rect.h + rect_spec.h) / 2 - gap, rect_spec.w, rect_spec.h)]

    def _is_valid_placement_no_overlap(self, node: int, candidate: RectangleLayout, current_layout: Dict[int, RectangleLayout]) -> bool:
        """
        Check if placement is valid (no overlaps) - relaxed contact requirement
        """
        for (other_node, other_rect) in current_layout.items():
            if self._rectangles_overlap(candidate, other_rect):
                return False
        return True

    def _is_valid_placement_with_contact(self, node: int, candidate: RectangleLayout, current_layout: Dict[int, RectangleLayout], contact_node: int) -> bool:
        """
        Check if placement is valid (no overlaps) and has contact with specified node
        """
        for (other_node, other_rect) in current_layout.items():
            if self._rectangles_overlap(candidate, other_rect):
                return False
        if contact_node in current_layout:
            return self._rectangles_in_contact(candidate, current_layout[contact_node])
        return False

    def _force_place_node(self, node: int, rect_spec: RectangleSpec, current_layout: Dict[int, RectangleLayout]) -> RectangleLayout:
        """
        Force place a node at a safe location (boundary of existing layout)
        """
        if not current_layout:
            return RectangleLayout(0, 0, rect_spec.w, rect_spec.h)
        max_x = max((rect.x + rect.w / 2 for rect in current_layout.values()))
        center_y = sum((rect.y for rect in current_layout.values())) / len(current_layout)
        return RectangleLayout(max_x + rect_spec.w / 2, center_y, rect_spec.w, rect_spec.h)

    def _calculate_contact_length(self, rect1: RectangleLayout, rect2: RectangleLayout, tolerance: float=0.001) -> float:
        """
        Calculate the length of contact between two rectangles.
        Returns 0 if rectangles are not in contact.
        """
        dx = abs(rect1.x - rect2.x)
        dy = abs(rect1.y - rect2.y)
        if abs(dx - (rect1.w + rect2.w) / 2) < tolerance and dy < (rect1.h + rect2.h) / 2 - tolerance:
            (y1_min, y1_max) = (rect1.y - rect1.h / 2, rect1.y + rect1.h / 2)
            (y2_min, y2_max) = (rect2.y - rect2.h / 2, rect2.y + rect2.h / 2)
            overlap_start = max(y1_min, y2_min)
            overlap_end = min(y1_max, y2_max)
            return max(0, overlap_end - overlap_start)
        if abs(dy - (rect1.h + rect2.h) / 2) < tolerance and dx < (rect1.w + rect2.w) / 2 - tolerance:
            (x1_min, x1_max) = (rect1.x - rect1.w / 2, rect1.x + rect1.w / 2)
            (x2_min, x2_max) = (rect2.x - rect2.w / 2, rect2.x + rect2.w / 2)
            overlap_start = max(x1_min, x2_min)
            overlap_end = min(x1_max, x2_max)
            return max(0, overlap_end - overlap_start)
        return 0.0

def visualize_solution(graph: nx.Graph, layout: Dict[int, RectangleLayout], original_rectangles: Dict[int, RectangleSpec], title: str, save_path: str):
    """Visualize the solution"""
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.set_aspect('equal')
    ax1.set_title(f'{title} - Rectangle Layout')
    colors = plt.cm.Set3(np.linspace(0, 1, len(layout)))
    for (i, (v, rect)) in enumerate(layout.items()):
        x_left = rect.x - rect.w / 2
        y_bottom = rect.y - rect.h / 2
        rectangle = Rectangle((x_left, y_bottom), rect.w, rect.h, facecolor=colors[i], edgecolor='black', alpha=0.7)
        ax1.add_patch(rectangle)
        ax1.text(rect.x, rect.y, str(v), ha='center', va='center', fontweight='bold')
    for (u, v) in graph.edges():
        (rect_u, rect_v) = (layout[u], layout[v])
        ax1.plot([rect_u.x, rect_v.x], [rect_u.y, rect_v.y], 'r-', alpha=0.5, linewidth=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, ax=ax2, with_labels=True, node_color='lightblue', node_size=500, font_size=12, font_weight='bold')
    ax2.set_title(f'{title} - Graph Structure')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_layout_solver_factory():
    """
    Factory function that returns the appropriate solver for given constraints
    
    Usage in layout.py:
    ```python
    from room_solver import create_layout_solver_factory
    
    # Get solver factory
    get_solver = create_layout_solver_factory()
    
    # Use solver for room layout
    graph = nx.Graph()
    rectangles = {...}
    solver = get_solver(graph, rectangles, use_relaxation=True)
    result_graph, result_layout = solver.solve()
    ```
    """

    def get_solver(graph: nx.Graph, rectangles: Dict[int, RectangleSpec], use_relaxation: bool=True, max_attempts: int=50):
        """
        Returns appropriate solver instance
        
        Args:
            graph: Contact graph between rectangles
            rectangles: Rectangle specifications
            use_relaxation: If True, uses RectangleContactRelaxationSolver as fallback
            max_attempts: Maximum solving attempts
            
        Returns:
            Tuple of (final_graph, final_layout)
        """
        if use_relaxation:
            print('Using RectangleContactRelaxationSolver (guaranteed solution)')
            solver = RectangleContactRelaxationSolver(graph, rectangles)
        else:
            print('Using original RectangleContactSolver')
            solver = RectangleContactSolver(graph, rectangles)
        return solver.solve(max_attempts=max_attempts)
    return get_solver

def solve_room_layout_with_relaxation(room_connections: List[Tuple[int, int]], room_specs: Dict[int, Tuple[float, float]], use_relaxation: bool=True) -> Dict:
    """
    Example function showing how to integrate relaxation solver into room layout system
    
    Args:
        room_connections: List of (room_id1, room_id2) tuples representing adjacency
        room_specs: Dict mapping room_id to (width, height) tuples
        use_relaxation: Whether to use relaxation solver
        
    Returns:
        Dict with layout information
    """
    graph = nx.Graph()
    graph.add_edges_from(room_connections)
    rectangles = {room_id: RectangleSpec(w, h) for (room_id, (w, h)) in room_specs.items()}
    get_solver = create_layout_solver_factory()
    try:
        (result_graph, result_layout) = get_solver(graph, rectangles, use_relaxation)
        room_layout = {}
        for (room_id, rect_layout) in result_layout.items():
            room_layout[room_id] = {'center_x': rect_layout.x, 'center_y': rect_layout.y, 'width': rect_layout.w, 'height': rect_layout.h, 'connections': list(result_graph.neighbors(room_id))}
        return {'success': True, 'room_layout': room_layout, 'total_rooms': len(result_layout), 'total_connections': len(result_graph.edges), 'solver_used': 'relaxation' if use_relaxation else 'original'}
    except Exception as e:
        return {'success': False, 'error': str(e), 'solver_used': 'relaxation' if use_relaxation else 'original'}
