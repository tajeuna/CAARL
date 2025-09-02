#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 18:25:08 2025

@author: tajeet01
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAARL Step 2: Temporal Dependency Graph Construction
Implementation of Section 3.1.2 from the paper
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import json
from CAARL_incontext_scenario2 import CAARLStep1 
from CAARL_serialization import run_caarl_step3
from CAARL_prediction_step import create_ai_config, run_caarl_step4_ai 


import seaborn as sns
import warnings
from CAARL_DataHandler import DataHandler
warnings.filterwarnings('ignore')

@dataclass
class GraphNode:
    """
    Represents a node in the temporal dependency graph
    """
    model_id: int
    interval: int
    series_vector: np.ndarray  # Binary vector indicating which series use this model
    model_params: Dict  # Store model coefficients and intercept
    
    def __hash__(self):
        return hash((self.model_id, self.interval))
    
    def __eq__(self, other):
        return self.model_id == other.model_id and self.interval == other.interval

@dataclass
class GraphEdge:
    """
    Represents an edge in the temporal dependency graph
    """
    source_node: GraphNode
    target_node: GraphNode
    series_name: str
    transition_type: str  # 'maintain', 'switch'
    
    def __hash__(self):
        return hash((self.source_node, self.target_node, self.series_name))

class CAARLStep2:
    """
    CAARL Step 2: Construction of temporal dependency graph
    Based on Section 3.1.2 of the paper
    """
    
    def __init__(self, step1_results: Dict):
        """
        Initialize Step 2 with results from Step 1
        
        Args:
            step1_results: Output from CAARLStep1.fit()
        """
        self.step1_results = step1_results
        self.temporal_graphs = {}  # {(interval_i, interval_j): nx.DiGraph}
        self.global_nodes = {}     # {(model_id, interval): GraphNode}
        self.all_series_names = self._extract_series_names()
        self.n_series = len(self.all_series_names)
        
    def _extract_series_names(self) -> List[str]:
        """Extract all unique series names from Step 1 results"""
        all_series = set()
        for result in self.step1_results['intervals']:
            all_series.update(result['series_to_cluster'].keys())
        return sorted(list(all_series))
    
    def build_temporal_dependency_graph(self) -> Dict:
        """
        Main method to build the temporal dependency graph as described in Section 3.1.2
        
        Returns:
            graph_info: Dictionary containing graph structure and metadata
        """
        print("=" * 60)
        print("CAARL STEP 2: TEMPORAL DEPENDENCY GRAPH CONSTRUCTION")
        print("=" * 60)
        
        # Step 1: Create nodes for each (model, interval) pair
        print("Step 1: Creating graph nodes...")
        self._create_graph_nodes()
        
        # Step 2: Build bipartite graphs between consecutive intervals
        print("Step 2: Building bipartite graphs between consecutive intervals...")
        self._build_consecutive_interval_graphs()
        
        # Step 3: Analyze dependencies and transitions
        print("Step 3: Analyzing model dependencies and transitions...")
        dependencies = self._analyze_dependencies()
        
        # Step 4: Create comprehensive graph representation
        print("Step 4: Creating comprehensive graph representation...")
        comprehensive_graph = self._create_comprehensive_graph()
        
        print(f"✅ TEMPORAL DEPENDENCY GRAPH COMPLETED")
        print(f"  - Total nodes: {len(self.global_nodes)}")
        print(f"  - Consecutive interval graphs: {len(self.temporal_graphs)}")
        print(f"  - Total dependencies found: {len(dependencies)}")
        
        return {
            'nodes': self.global_nodes,
            'temporal_graphs': self.temporal_graphs,
            'dependencies': dependencies,
            'comprehensive_graph': comprehensive_graph,
            'series_names': self.all_series_names,
            'summary': self._generate_graph_summary()
        }
    
    def _create_graph_nodes(self):
        """
        Create GraphNode objects for each (model_id, interval) pair
        Following the paper's notation: nodes represent models at specific intervals
        """
        for interval_idx, result in enumerate(self.step1_results['intervals']):
            interval_range = result['interval']
            series_to_cluster = result['series_to_cluster']
            cluster_models = result['cluster_models']
            
            # Create nodes for each unique model in this interval
            for cluster_id, model in cluster_models.items():
                # Create binary vector indicating which series use this model
                series_vector = np.zeros(self.n_series, dtype=int)
                
                for series_name, assigned_cluster in series_to_cluster.items():
                    if assigned_cluster == cluster_id:
                        series_idx = self.all_series_names.index(series_name)
                        series_vector[series_idx] = 1
                
                # Store model parameters
                model_params = {
                    'coefficients': model.coefficients.tolist() if model.coefficients is not None else [],
                    'intercept': float(model.intercept) if model.intercept is not None else 0.0,
                    'lag': model.lag,
                    'fitted': model.fitted
                }
                
                # Create node
                node = GraphNode(
                    model_id=cluster_id,
                    interval=interval_idx,
                    series_vector=series_vector,
                    model_params=model_params
                )
                
                self.global_nodes[(cluster_id, interval_idx)] = node
    
    def _build_consecutive_interval_graphs(self):
        """
        Build bipartite directed graphs between consecutive intervals
        Following Equation 3.5: E_{j,(j+1)} edges between intervals T_j and T_{j+1}
        """
        intervals = self.step1_results['intervals']
        
        for i in range(len(intervals) - 1):
            current_interval = i
            next_interval = i + 1
            
            # Create bipartite graph between intervals i and i+1
            graph_key = (current_interval, next_interval)
            G = nx.DiGraph()
            
            # Get series transitions between these intervals
            current_result = intervals[current_interval]
            next_result = intervals[next_interval]
            
            current_series_to_cluster = current_result['series_to_cluster']
            next_series_to_cluster = next_result['series_to_cluster']
            
            # Find all series that exist in both intervals
            common_series = set(current_series_to_cluster.keys()) & set(next_series_to_cluster.keys())
            
            # Create edges based on series transitions
            edges_created = set()
            
            for series_name in common_series:
                current_cluster = current_series_to_cluster[series_name]
                next_cluster = next_series_to_cluster[series_name]
                
                # Create edge from current model to next model for this series
                source_key = (current_cluster, current_interval)
                target_key = (next_cluster, next_interval)
                
                if source_key in self.global_nodes and target_key in self.global_nodes:
                    source_node = self.global_nodes[source_key]
                    target_node = self.global_nodes[target_key]
                    
                    # Add nodes to graph if not already present
                    if not G.has_node(source_key):
                        G.add_node(source_key, node_obj=source_node)
                    if not G.has_node(target_key):
                        G.add_node(target_key, node_obj=target_node)
                    
                    # Create edge
                    edge_key = (source_key, target_key)
                    if edge_key not in edges_created:
                        transition_type = 'maintain' if current_cluster == next_cluster else 'switch'
                        
                        # Add edge with metadata
                        G.add_edge(source_key, target_key, 
                                 series=[series_name],
                                 transition_type=transition_type,
                                 weight=1)
                        edges_created.add(edge_key)
                    else:
                        # Add series to existing edge
                        existing_series = G[source_key][target_key]['series']
                        if series_name not in existing_series:
                            existing_series.append(series_name)
                            G[source_key][target_key]['weight'] += 1
            
            self.temporal_graphs[graph_key] = G
    
    def _analyze_dependencies(self) -> List[Dict]:
        """
        Analyze dependencies based on model sharing patterns
        Implementation of the dependency analysis described in Section 3.1.2
        """
        dependencies = []
        
        # Analyze each consecutive interval pair
        for (interval_i, interval_j), graph in self.temporal_graphs.items():
            
            for edge in graph.edges(data=True):
                source_key, target_key, edge_data = edge
                source_node = self.global_nodes[source_key]
                target_node = self.global_nodes[target_key]
                
                # Extract dependency information
                dependency = {
                    'source_interval': interval_i,
                    'target_interval': interval_j,
                    'source_model': source_node.model_id,
                    'target_model': target_node.model_id,
                    'series_involved': edge_data['series'],
                    'transition_type': edge_data['transition_type'],
                    'strength': len(edge_data['series']),  # Number of series making this transition
                    'source_series_vector': source_node.series_vector.tolist(),
                    'target_series_vector': target_node.series_vector.tolist()
                }
                
                dependencies.append(dependency)
        
        return dependencies
    
    def _create_comprehensive_graph(self) -> nx.DiGraph:
        """
        Create a single comprehensive graph combining all temporal relationships
        """
        comprehensive_graph = nx.DiGraph()
        
        # Add all nodes
        for node_key, node_obj in self.global_nodes.items():
            comprehensive_graph.add_node(
                node_key,
                model_id=node_obj.model_id,
                interval=node_obj.interval,
                series_vector=node_obj.series_vector.tolist(),
                model_params=node_obj.model_params,
                series_count=np.sum(node_obj.series_vector)
            )
        
        # Add all edges from temporal graphs
        for (interval_i, interval_j), graph in self.temporal_graphs.items():
            for source_key, target_key, edge_data in graph.edges(data=True):
                comprehensive_graph.add_edge(
                    source_key, target_key,
                    interval_transition=(interval_i, interval_j),
                    series=edge_data['series'],
                    transition_type=edge_data['transition_type'],
                    weight=edge_data['weight']
                )
        
        return comprehensive_graph
    
    def _generate_graph_summary(self) -> Dict:
        """
        Generate summary statistics about the temporal dependency graph
        """
        total_nodes = len(self.global_nodes)
        total_edges = sum(len(g.edges()) for g in self.temporal_graphs.values())
        
        # Count transition types
        maintain_transitions = 0
        switch_transitions = 0
        
        for graph in self.temporal_graphs.values():
            for _, _, edge_data in graph.edges(data=True):
                if edge_data['transition_type'] == 'maintain':
                    maintain_transitions += 1
                else:
                    switch_transitions += 1
        
        # Analyze model persistence
        model_intervals = {}  # {model_id: [intervals where it appears]}
        for (model_id, interval) in self.global_nodes.keys():
            if model_id not in model_intervals:
                model_intervals[model_id] = []
            model_intervals[model_id].append(interval)
        
        # Calculate model persistence metrics
        model_persistence = {}
        for model_id, intervals in model_intervals.items():
            model_persistence[model_id] = {
                'total_intervals': len(intervals),
                'span': max(intervals) - min(intervals) + 1 if intervals else 0,
                'first_appearance': min(intervals) if intervals else None,
                'last_appearance': max(intervals) if intervals else None
            }
        
        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'maintain_transitions': maintain_transitions,
            'switch_transitions': switch_transitions,
            'unique_models': len(model_intervals),
            'model_persistence': model_persistence,
            'avg_transitions_per_interval': total_edges / max(1, len(self.temporal_graphs))
        }
    
    def visualize_temporal_bipartite_graph(self, save_plot: bool = False, figsize: Tuple[int, int] = (20, 8)):
        """
        Visualize the temporal dependency graph as consecutive bipartite graphs
        Shows first 3 and last 3 transitions with dots in between, like the paper figure
        
        Args:
            save_plot: Whether to save the plot
            figsize: Figure size
        """
        if not self.temporal_graphs:
            print("No temporal graphs to visualize")
            return
            
        # Get all transitions sorted by interval
        all_transitions = sorted(list(self.temporal_graphs.keys()))
        n_total_transitions = len(all_transitions)
        
        if n_total_transitions <= 6:
            # Show all transitions if 6 or fewer
            transitions_to_show = all_transitions
            show_dots = False
        else:
            # Show first 3, last 3, and dots in between
            transitions_to_show = all_transitions[:3] + all_transitions[-3:]
            show_dots = True
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Get all unique model IDs across all intervals
        all_models = set()
        for node_key in self.global_nodes.keys():
            all_models.add(node_key[0])  # model_id
        all_models = sorted(list(all_models))
        
        # Create positions for each interval column
        interval_positions = {}
        x_spacing = 3.0  # Space between intervals
        y_spacing = 0.5  # Space between models
        
        # Get all intervals that will be displayed
        intervals_to_display = set()
        for (interval_i, interval_j) in transitions_to_show:
            intervals_to_display.add(interval_i)
            intervals_to_display.add(interval_j)
        intervals_to_display = sorted(list(intervals_to_display))
        
        # Calculate x positions - handle the gap for dots
        x_positions = {}
        if show_dots and n_total_transitions > 6:
            # First 3 transitions use first 4 x positions (0, 1, 2, 3)
            for i, interval in enumerate(intervals_to_display[:4]):
                x_positions[interval] = i * x_spacing
            
            # Add gap for dots
            dots_x = 4 * x_spacing
            
            # Last 3 transitions use positions after the gap
            for i, interval in enumerate(intervals_to_display[4:]):
                x_positions[interval] = (5 + i) * x_spacing
        else:
            # No gaps, regular spacing
            for i, interval in enumerate(intervals_to_display):
                x_positions[interval] = i * x_spacing
        
        # Position nodes for each interval
        for interval_idx in intervals_to_display:
            x_pos = x_positions[interval_idx]
            
            # Get active models in this interval
            active_models = []
            for (model_id, interval) in self.global_nodes.keys():
                if interval == interval_idx:
                    active_models.append(model_id)
            
            active_models = sorted(active_models)
            
            # Position models vertically, centered around the middle
            n_models = len(active_models)
            if n_models > 0:
                # Center the models vertically
                total_height = (len(all_models) - 1) * y_spacing
                start_y = total_height / 2 - (n_models - 1) * y_spacing / 2
                
                for i, model_id in enumerate(active_models):
                    y_pos = start_y - i * y_spacing
                    interval_positions[(model_id, interval_idx)] = (x_pos, y_pos)
        
        # Draw nodes
        for (model_id, interval), pos in interval_positions.items():
            node_obj = self.global_nodes[(model_id, interval)]
            series_count = np.sum(node_obj.series_vector)
            
            # Color based on model ID with consistent colors
            colors = plt.cm.Set3(np.linspace(0, 1, max(len(all_models), 12)))
            color = colors[all_models.index(model_id)]
            
            circle = plt.Circle(pos, 0.12, color=color, alpha=0.8)
            ax.add_patch(circle)
            
            # Add model label
            ax.text(pos[0], pos[1], f'{model_id}', 
                   ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Draw edges for displayed transitions
        for transition in transitions_to_show:
            if transition in self.temporal_graphs:
                graph = self.temporal_graphs[transition]
                for source_key, target_key, edge_data in graph.edges(data=True):
                    if source_key in interval_positions and target_key in interval_positions:
                        source_pos = interval_positions[source_key]
                        target_pos = interval_positions[target_key]
                        
                        # Color edge based on transition type
                        edge_color = 'blue' if edge_data['transition_type'] == 'maintain' else 'red'
                        line_width = min(edge_data['weight'] * 0.3 + 0.8, 2.5)
                        
                        # Draw arrow
                        ax.annotate('', xy=target_pos, xytext=source_pos,
                                   arrowprops=dict(arrowstyle='->', color=edge_color, 
                                                 lw=line_width, alpha=0.7))
        
        # Add transition labels at the top
        transition_labels = []
        label_positions = []
        
        if show_dots and n_total_transitions > 6:
            # First 3 transitions
            for i, (interval_i, interval_j) in enumerate(transitions_to_show[:3]):
                x_mid = (x_positions[interval_i] + x_positions[interval_j]) / 2
                transition_labels.append(f'T_{interval_i + 1} - T_{interval_j + 1}')
                label_positions.append(x_mid)
            
            # Dots
            dots_x_pos = dots_x
            transition_labels.append('...')
            label_positions.append(dots_x_pos)
            
            # Last 3 transitions
            for i, (interval_i, interval_j) in enumerate(transitions_to_show[3:]):
                x_mid = (x_positions[interval_i] + x_positions[interval_j]) / 2
                transition_labels.append(f'T_{interval_i + 1} - T_{interval_j + 1}')
                label_positions.append(x_mid)
        else:
            # All transitions
            for (interval_i, interval_j) in transitions_to_show:
                x_mid = (x_positions[interval_i] + x_positions[interval_j]) / 2
                transition_labels.append(f'T_{interval_i + 1} - T_{interval_j + 1}')
                label_positions.append(x_mid)
        
        # Draw transition labels
        y_top = max(pos[1] for pos in interval_positions.values()) + 0.8
        for label, x_pos in zip(transition_labels, label_positions):
            ax.text(x_pos, y_top, label, ha='center', va='bottom', 
                   fontsize=12, fontweight='bold')
        
        # Add model numbers on the left side
        left_x = min(x_positions.values()) - 0.8
        for model_id in all_models:
            # Find any position for this model to get y coordinate
            model_positions = [pos for (mid, _), pos in interval_positions.items() if mid == model_id]
            if model_positions:
                # Use the average y position if model appears multiple times
                avg_y = np.mean([pos[1] for pos in model_positions])
                ax.text(left_x, avg_y, str(model_id), ha='right', va='center', 
                       fontsize=11, fontweight='bold')
        
        # Add dots in the middle if needed
        if show_dots and n_total_transitions > 6:
            # Draw three dots vertically centered
            dot_y_positions = [y_top - 1, y_top - 1.5, y_top - 2]
            for dot_y in dot_y_positions:
                ax.plot(dots_x, dot_y, 'ko', markersize=8)
        
        # Set axis properties
        max_x = max(x_positions.values()) if x_positions else 0
        ax.set_xlim(left_x - 0.5, max_x + 0.5)
        ax.set_ylim(min(pos[1] for pos in interval_positions.values()) - 0.5, y_top + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('CAARL Temporal Dependency Graph: Model Transitions Across Time', 
                    fontsize=16, pad=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', lw=2, label='Model Maintained'),
            plt.Line2D([0], [0], color='red', lw=2, label='Model Switch'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('caarl_temporal_bipartite_graph.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_temporal_graph(self, interval_range: Optional[Tuple[int, int]] = None, 
                               save_plot: bool = False, figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize the temporal dependency graph (original 4-panel version)
        
        Args:
            interval_range: Optional tuple (start, end) to visualize specific interval range
            save_plot: Whether to save the plot
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('CAARL Step 2: Temporal Dependency Graph Analysis', fontsize=16)
        
        # Plot 1: Model transitions over time
        self._plot_model_transitions(axes[0, 0])
        
        # Plot 2: Series transition patterns
        self._plot_series_transitions(axes[0, 1])
        
        # Plot 3: Model persistence
        self._plot_model_persistence(axes[1, 0])
        
        # Plot 4: Network graph of a specific interval transition
        self._plot_network_graph(axes[1, 1], interval_range)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('caarl_step2_temporal_graph_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_model_transitions(self, ax):
        """Plot model transitions over time"""
        intervals = list(range(len(self.step1_results['intervals'])))
        
        # Count active models per interval
        models_per_interval = []
        for i, result in enumerate(self.step1_results['intervals']):
            unique_models = set(result['series_to_cluster'].values())
            models_per_interval.append(len(unique_models))
        
        ax.plot(intervals, models_per_interval, 'bo-', linewidth=2, markersize=6)
        ax.set_title('Active Models per Interval')
        ax.set_xlabel('Interval Index')
        ax.set_ylabel('Number of Active Models')
        ax.grid(True, alpha=0.3)
    
    def _plot_series_transitions(self, ax):
        """Plot series transition patterns"""
        maintain_count = []
        switch_count = []
        
        for (interval_i, interval_j), graph in self.temporal_graphs.items():
            maintain = sum(1 for _, _, data in graph.edges(data=True) 
                          if data['transition_type'] == 'maintain')
            switch = sum(1 for _, _, data in graph.edges(data=True) 
                        if data['transition_type'] == 'switch')
            maintain_count.append(maintain)
            switch_count.append(switch)
        
        transitions = list(range(len(self.temporal_graphs)))
        
        ax.bar([t - 0.2 for t in transitions], maintain_count, 0.4, 
               label='Maintain Model', alpha=0.7, color='green')
        ax.bar([t + 0.2 for t in transitions], switch_count, 0.4, 
               label='Switch Model', alpha=0.7, color='red')
        
        ax.set_title('Series Transition Patterns')
        ax.set_xlabel('Interval Transition')
        ax.set_ylabel('Number of Transitions')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_model_persistence(self, ax):
        """Plot model persistence across intervals"""
        summary = self._generate_graph_summary()
        model_persistence = summary['model_persistence']
        
        model_ids = list(model_persistence.keys())
        total_intervals = [model_persistence[mid]['total_intervals'] for mid in model_ids]
        
        ax.bar(range(len(model_ids)), total_intervals, alpha=0.7, color='blue')
        ax.set_title('Model Persistence (Intervals Active)')
        ax.set_xlabel('Model ID')
        ax.set_ylabel('Number of Intervals')
        ax.set_xticks(range(len(model_ids)))
        ax.set_xticklabels([f'Θ{mid}' for mid in model_ids])
        ax.grid(True, alpha=0.3)
    
    def _plot_network_graph(self, ax, interval_range: Optional[Tuple[int, int]] = None):
        """Plot network graph of model transitions"""
        if not self.temporal_graphs:
            ax.text(0.5, 0.5, 'No temporal graphs to display', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Use first temporal graph if no range specified
        if interval_range is None:
            graph_key = list(self.temporal_graphs.keys())[0]
        else:
            graph_key = interval_range
            
        if graph_key not in self.temporal_graphs:
            ax.text(0.5, 0.5, f'No graph found for intervals {graph_key}', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        graph = self.temporal_graphs[graph_key]
        
        # Create layout
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Draw nodes
        node_colors = ['lightblue' if key[1] == graph_key[0] else 'lightgreen' 
                      for key in graph.nodes()]
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=500, ax=ax)
        
        # Draw edges
        edge_colors = ['green' if data['transition_type'] == 'maintain' else 'red' 
                      for _, _, data in graph.edges(data=True)]
        nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, 
                              arrows=True, ax=ax)
        
        # Draw labels
        labels = {key: f'Θ{key[0]}' for key in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(f'Transition Graph: T{graph_key[0]} → T{graph_key[1]}')
        ax.set_aspect('equal')
    
    def _convert_to_json_serializable(self, obj):
        """
        Convert numpy types to JSON serializable Python types
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def export_graph_data(self, filepath: str):
        """
        Export graph data to JSON file
        
        Args:
            filepath: Path to save the JSON file
        """
        # Prepare data for JSON serialization
        export_data = {
            'nodes': {},
            'edges': [],
            'summary': self._convert_to_json_serializable(self._generate_graph_summary()),
            'series_names': self.all_series_names
        }
        
        # Convert nodes
        for node_key, node_obj in self.global_nodes.items():
            export_data['nodes'][f"{int(node_key[0])}_{int(node_key[1])}"] = {
                'model_id': int(node_obj.model_id),
                'interval': int(node_obj.interval),
                'series_vector': [int(x) for x in node_obj.series_vector],
                'model_params': self._convert_to_json_serializable(node_obj.model_params)
            }
        
        # Convert edges
        for (interval_i, interval_j), graph in self.temporal_graphs.items():
            for source_key, target_key, edge_data in graph.edges(data=True):
                export_data['edges'].append({
                    'source': f"{int(source_key[0])}_{int(source_key[1])}",
                    'target': f"{int(target_key[0])}_{int(target_key[1])}",
                    'interval_transition': [int(interval_i), int(interval_j)],
                    'series': edge_data['series'],
                    'transition_type': edge_data['transition_type'],
                    'weight': int(edge_data['weight'])
                })
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Graph data exported to {filepath}")
    
    def get_dependencies_dataframe(self) -> pd.DataFrame:
        """
        Get dependencies as a pandas DataFrame for analysis
        
        Returns:
            df: DataFrame with dependency information
        """
        dependencies = self._analyze_dependencies()
        return pd.DataFrame(dependencies)
    
    def print_graph_analysis(self):
        """
        Print comprehensive analysis of the temporal dependency graph
        """
        summary = self._generate_graph_summary()
        
        print("\n" + "="*60)
        print("TEMPORAL DEPENDENCY GRAPH ANALYSIS")
        print("="*60)
        
        print(f"Graph Structure:")
        print(f"  - Total nodes: {summary['total_nodes']}")
        print(f"  - Total edges: {summary['total_edges']}")
        print(f"  - Unique models: {summary['unique_models']}")
        print(f"  - Avg transitions per interval: {summary['avg_transitions_per_interval']:.2f}")
        
        print(f"\nTransition Analysis:")
        print(f"  - Model maintained: {summary['maintain_transitions']}")
        print(f"  - Model switches: {summary['switch_transitions']}")
        total_transitions = summary['maintain_transitions'] + summary['switch_transitions']
        if total_transitions > 0:
            maintain_pct = summary['maintain_transitions'] / total_transitions * 100
            switch_pct = summary['switch_transitions'] / total_transitions * 100
            print(f"  - Maintain ratio: {maintain_pct:.1f}%")
            print(f"  - Switch ratio: {switch_pct:.1f}%")
        
        print(f"\nModel Persistence Analysis:")
        for model_id, persistence in summary['model_persistence'].items():
            print(f"  - Model Θ{model_id}: active in {persistence['total_intervals']} intervals "
                  f"(span: {persistence['span']}, T{persistence['first_appearance']}-T{persistence['last_appearance']})")


# Example usage function
def run_caarl_step2(step1_results: Dict) -> Dict:
    """
    Run CAARL Step 2 on the results from Step 1
    
    Args:
        step1_results: Results from CAARLStep1.fit()
        
    Returns:
        step2_results: Results from temporal graph construction
    """
    # Initialize Step 2
    step2 = CAARLStep2(step1_results)
    
    # Build temporal dependency graph
    graph_results = step2.build_temporal_dependency_graph()
    
    # Print analysis
    step2.print_graph_analysis()
    
    # Visualize as bipartite graph (like your figure)
    step2.visualize_temporal_bipartite_graph(save_plot=True)
    
    # Also create the 4-panel analysis
    step2.visualize_temporal_graph(save_plot=True)
    
    # Export data
    step2.export_graph_data('caarl_temporal_graph.json')
    
    return graph_results


dataname = 'rock_data' #'SyD-50', 'ETTh1', 'FXs_interpolated', info_tech_sector, passengers
path_to_data = '~/Documents/Projets/Causal-Inference-Graph-Modeling-in-CoEvolving-Time-Sequences 2/dataset/'+dataname+'.csv'
df = DataHandler(path_to_data, size=None, stride=None)._load_data(path_to_data, normalize=True)

# Example configuration for OpenAI
openai_config = create_ai_config(
    provider="openai",
    model="gpt-4",
    api_key="your-openai-api-key"  # or set OPENAI_API_KEY environment variable
)

# Example configuration for Anthropic
anthropic_config = create_ai_config(
    provider="anthropic", 
    model="claude-3-sonnet-20240229",
    api_key="your-anthropic-api-key"  # or set ANTHROPIC_API_KEY environment variable
)


if __name__ == '__main__':
    caarl_auto = CAARLStep1(auto_params=True, change_detection_priority="medium")
    results_auto = caarl_auto.fit(df)
    print(results_auto)
   
    # Visualize results for automatic selection
    print("\n📊 Visualizing Automatic Parameter Selection Results...")
    caarl_auto.visualize_results(df, results_auto, save_plots=True)
    
    # Print model summary for automatic selection
    print("\n📈 MODEL SUMMARY (Automatic Parameters):")
    print("="*80)
    summary_df = caarl_auto.get_model_summary()
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    else:
        print("No models discovered.")
    
    # Print detailed clustering information for first few intervals
    print("\n🔍 DETAILED CLUSTERING (First 3 intervals - Automatic):")
    print("="*80)
    for i in range(min(3, len(results_auto['intervals']))):
        result = results_auto['intervals'][i]
        print(f"\nInterval {i+1}: {result['interval']}")
        print(f"  Series clustering: {result['series_to_cluster']}")
        print(f"  Performance: {dict(list(result['performance'].items())[:5])}...")  # Show first 5 performance values
        
        
    graph_results = run_caarl_step2(results_auto)
    
    result_narrative = run_caarl_step3(results_auto, graph_results, target_series='HUFL') 
    result_narrative2 = run_caarl_step3(results_auto, graph_results, target_series=None) 
    
    results = run_caarl_step4_ai(results_auto, graph_results, result_narrative2, openai_config, "0")
