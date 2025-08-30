#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 22:34:43 2025

@author: tajeet01

CAARL Step 3: Graph Serialization and Narrative Generation
Implementation of Section 3.2 from the paper - converting temporal graphs to natural language
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import json

@dataclass
class SeriesTransition:
    """
    Represents a single transition for a time series
    """
    series_name: str
    from_interval: int
    to_interval: int
    from_model: int
    to_model: int
    transition_type: str  # 'maintain' or 'switch'
    
@dataclass  
class SeriesNarrative:
    """
    Complete narrative for a time series across all intervals
    """
    series_name: str
    transitions: List[SeriesTransition]
    pattern_summary: str
    prediction_context: str

class CAARLStep3:
    """
    CAARL Step 3: Graph serialization into natural language narratives
    Based on Section 3.2 of the paper
    """
    
    def __init__(self, step1_results: Dict, step2_results: Dict):
        """
        Initialize Step 3 with results from Steps 1 and 2
        
        Args:
            step1_results: Output from CAARLStep1.fit()
            step2_results: Output from CAARLStep2.build_temporal_dependency_graph()
        """
        self.step1_results = step1_results
        self.step2_results = step2_results
        self.series_names = step2_results['series_names']
        self.temporal_graphs = step2_results['temporal_graphs']
        self.global_nodes = step2_results['nodes']
        
        # Build comprehensive mapping of series trajectories
        self.series_trajectories = self._build_series_trajectories()
        
    def _build_series_trajectories(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Build complete trajectories for each series: (interval, model_id) pairs
        
        Returns:
            Dict mapping series names to their complete trajectories
        """
        trajectories = {}
        
        # Initialize trajectories for all series
        for series_name in self.series_names:
            trajectories[series_name] = []
            
        # Build trajectories from step1 results
        for interval_idx, result in enumerate(self.step1_results['intervals']):
            series_to_cluster = result['series_to_cluster']
            
            for series_name, cluster_id in series_to_cluster.items():
                if series_name in trajectories:
                    trajectories[series_name].append((interval_idx, cluster_id))
        
        # Sort trajectories by interval
        for series_name in trajectories:
            trajectories[series_name].sort(key=lambda x: x[0])
            
        return trajectories
    
    def create_series_narratives(self, q: int = 5) -> Dict[str, SeriesNarrative]:
        """
        Create natural language narratives for each time series
        
        Args:
            q: Number of previous states to consider for context (as in paper)
            
        Returns:
            Dict mapping series names to their narratives
        """
        print("=" * 60)
        print("CAARL STEP 3: GRAPH SERIALIZATION AND NARRATIVE GENERATION")
        print("=" * 60)
        
        narratives = {}
        
        for series_name in self.series_names:
            if series_name in self.series_trajectories:
                narrative = self._create_single_series_narrative(series_name, q)
                narratives[series_name] = narrative
                
        print(f"Generated narratives for {len(narratives)} time series")
        return narratives
    
    def _create_single_series_narrative(self, series_name: str, q: int) -> SeriesNarrative:
        """
        Create narrative for a single time series
        
        Args:
            series_name: Name of the series
            q: Context window size
            
        Returns:
            SeriesNarrative object
        """
        trajectory = self.series_trajectories[series_name]
        transitions = []
        
        # Create transitions from trajectory
        for i in range(len(trajectory) - 1):
            current_interval, current_model = trajectory[i]
            next_interval, next_model = trajectory[i + 1]
            
            transition_type = 'maintain' if current_model == next_model else 'switch'
            
            transition = SeriesTransition(
                series_name=series_name,
                from_interval=current_interval,
                to_interval=next_interval,
                from_model=current_model,
                to_model=next_model,
                transition_type=transition_type
            )
            transitions.append(transition)
        
        # Generate pattern summary
        pattern_summary = self._generate_pattern_summary(series_name, transitions)
        
        # Generate prediction context
        prediction_context = self._generate_prediction_context(series_name, transitions, q)
        
        return SeriesNarrative(
            series_name=series_name,
            transitions=transitions,
            pattern_summary=pattern_summary,
            prediction_context=prediction_context
        )
    
    def _generate_pattern_summary(self, series_name: str, transitions: List[SeriesTransition]) -> str:
        """
        Generate a summary of the series' transition patterns
        
        Args:
            series_name: Name of the series
            transitions: List of transitions
            
        Returns:
            Natural language summary
        """
        if not transitions:
            return f"Series {series_name} has insufficient data for pattern analysis."
        
        # Count transition types
        maintain_count = sum(1 for t in transitions if t.transition_type == 'maintain')
        switch_count = sum(1 for t in transitions if t.transition_type == 'switch')
        
        # Identify most frequently used models
        model_usage = {}
        for transition in transitions:
            model_usage[transition.from_model] = model_usage.get(transition.from_model, 0) + 1
            model_usage[transition.to_model] = model_usage.get(transition.to_model, 0) + 1
        
        most_used_models = sorted(model_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Generate summary text
        summary_parts = []
        summary_parts.append(f"Series {series_name} exhibits {len(transitions)} model transitions across time intervals.")
        
        if maintain_count > switch_count:
            summary_parts.append(f"The series shows stability with {maintain_count} maintained models versus {switch_count} switches.")
        else:
            summary_parts.append(f"The series shows high variability with {switch_count} model switches versus {maintain_count} maintained models.")
        
        if most_used_models:
            primary_models = [f"Θ{model_id}" for model_id, _ in most_used_models]
            summary_parts.append(f"Primary models used: {', '.join(primary_models)}.")
        
        return " ".join(summary_parts)
    
    def _generate_prediction_context(self, series_name: str, transitions: List[SeriesTransition], q: int) -> str:
        """
        Generate prediction context based on recent q transitions
        
        Args:
            series_name: Name of the series
            transitions: List of transitions
            q: Number of recent transitions to consider
            
        Returns:
            Natural language prediction context
        """
        if len(transitions) < q:
            recent_transitions = transitions
        else:
            recent_transitions = transitions[-q:]
        
        if not recent_transitions:
            return f"No recent transition history available for {series_name}."
        
        context_parts = []
        context_parts.append(f"Recent transition history for {series_name}:")
        
        for i, transition in enumerate(recent_transitions):
            interval_desc = f"T{transition.from_interval + 1}→T{transition.to_interval + 1}"
            
            if transition.transition_type == 'maintain':
                context_parts.append(f"  - {interval_desc}: maintained model Θ{transition.from_model}")
            else:
                context_parts.append(f"  - {interval_desc}: switched from Θ{transition.from_model} to Θ{transition.to_model}")
        
        # Add prediction reasoning
        if recent_transitions:
            last_transition = recent_transitions[-1]
            current_model = last_transition.to_model
            
            # Look for patterns in recent transitions
            recent_models = [t.to_model for t in recent_transitions]
            model_stability = len(set(recent_models[-min(3, len(recent_models)):]))  # Unique models in last 3 transitions
            
            if model_stability == 1:
                context_parts.append(f"Current pattern: {series_name} has been stable with model Θ{current_model}.")
            else:
                context_parts.append(f"Current pattern: {series_name} shows variability, currently using model Θ{current_model}.")
        
        return "\n".join(context_parts)
    
    def serialize_graph_for_llm(self, target_series: Optional[str] = None, q: int = 5) -> str:
        """
        Serialize the temporal graph into natural language for LLM consumption
        Following the serialization function described in Section 3.2
        
        Args:
            target_series: Specific series to focus on (if None, includes all)
            q: Context window size
            
        Returns:
            Natural language serialization of the graph
        """
        narratives = self.create_series_narratives(q)
        
        if target_series and target_series in narratives:
            # Focus on specific series
            narrative = narratives[target_series]
            return self._serialize_single_series(narrative, q)
        else:
            # Serialize all series relationships
            return self._serialize_all_series(narratives, q)
    
    def _serialize_single_series(self, narrative: SeriesNarrative, q: int) -> str:
        """
        Serialize a single series narrative for LLM
        
        Args:
            narrative: SeriesNarrative object
            q: Context window size
            
        Returns:
            Serialized narrative
        """
        serialization_parts = []
        
        # Add header
        serialization_parts.append(f"=== TEMPORAL ANALYSIS FOR {narrative.series_name} ===")
        serialization_parts.append("")
        
        # Add pattern summary
        serialization_parts.append("PATTERN SUMMARY:")
        serialization_parts.append(narrative.pattern_summary)
        serialization_parts.append("")
        
        # Add recent context
        serialization_parts.append("RECENT CONTEXT:")
        serialization_parts.append(narrative.prediction_context)
        serialization_parts.append("")
        
        # Add dependency analysis
        dependencies = self._find_series_dependencies(narrative.series_name)
        if dependencies:
            serialization_parts.append("DEPENDENCY RELATIONSHIPS:")
            for dep in dependencies:
                serialization_parts.append(dep)
            serialization_parts.append("")
        
        # Add prediction prompt
        serialization_parts.append("PREDICTION TASK:")
        current_model = narrative.transitions[-1].to_model if narrative.transitions else "unknown"
        next_interval = narrative.transitions[-1].to_interval + 1 if narrative.transitions else "next"
        
        serialization_parts.append(f"Given the above context, predict whether {narrative.series_name} will:")
        serialization_parts.append(f"  A) Continue using model Θ{current_model} at interval T{next_interval}")
        serialization_parts.append(f"  B) Switch to a different model at interval T{next_interval}")
        serialization_parts.append("")
        serialization_parts.append("If switching, specify which model and explain the reasoning based on the patterns observed.")
        
        return "\n".join(serialization_parts)
    
    def _serialize_all_series(self, narratives: Dict[str, SeriesNarrative], q: int) -> str:
        """
        Serialize all series relationships for comprehensive analysis
        
        Args:
            narratives: Dict of all series narratives
            q: Context window size
            
        Returns:
            Comprehensive serialization
        """
        serialization_parts = []
        
        # Add header
        serialization_parts.append("=== COMPREHENSIVE TEMPORAL DEPENDENCY ANALYSIS ===")
        serialization_parts.append("")
        
        # Add global summary
        total_series = len(narratives)
        total_transitions = sum(len(n.transitions) for n in narratives.values())
        
        serialization_parts.append("GLOBAL SUMMARY:")
        serialization_parts.append(f"Analyzing {total_series} co-evolving time series with {total_transitions} total transitions.")
        serialization_parts.append("")
        
        # Group series by behavior patterns
        stable_series = []
        variable_series = []
        
        for series_name, narrative in narratives.items():
            if narrative.transitions:
                maintain_count = sum(1 for t in narrative.transitions if t.transition_type == 'maintain')
                switch_count = sum(1 for t in narrative.transitions if t.transition_type == 'switch')
                
                if maintain_count > switch_count:
                    stable_series.append(series_name)
                else:
                    variable_series.append(series_name)
        
        serialization_parts.append("BEHAVIORAL CLASSIFICATION:")
        serialization_parts.append(f"Stable series ({len(stable_series)}): {', '.join(stable_series[:10])}")
        if len(stable_series) > 10:
            serialization_parts.append(f"... and {len(stable_series) - 10} others")
        
        serialization_parts.append(f"Variable series ({len(variable_series)}): {', '.join(variable_series[:10])}")
        if len(variable_series) > 10:
            serialization_parts.append(f"... and {len(variable_series) - 10} others")
        serialization_parts.append("")
        
        # Add cross-series dependencies
        serialization_parts.append("CROSS-SERIES DEPENDENCIES:")
        cross_dependencies = self._analyze_cross_dependencies()
        for dep in cross_dependencies[:20]:  # Limit to top 20 dependencies
            serialization_parts.append(dep)
        serialization_parts.append("")
        
        # Add model usage patterns
        serialization_parts.append("MODEL USAGE PATTERNS:")
        model_usage = self._analyze_global_model_usage()
        serialization_parts.append(model_usage)
        
        return "\n".join(serialization_parts)
    
    def _find_series_dependencies(self, target_series: str) -> List[str]:
        """
        Find dependencies involving the target series
        
        Args:
            target_series: Series to analyze dependencies for
            
        Returns:
            List of dependency descriptions
        """
        dependencies = []
        
        # Look through temporal graphs for shared model patterns
        for (interval_i, interval_j), graph in self.temporal_graphs.items():
            # Find what model the target series uses in each interval
            target_model_i = None
            target_model_j = None
            
            # Check step1 results for target series model assignments
            if interval_i < len(self.step1_results['intervals']):
                series_to_cluster_i = self.step1_results['intervals'][interval_i]['series_to_cluster']
                target_model_i = series_to_cluster_i.get(target_series)
            
            if interval_j < len(self.step1_results['intervals']):
                series_to_cluster_j = self.step1_results['intervals'][interval_j]['series_to_cluster']  
                target_model_j = series_to_cluster_j.get(target_series)
            
            if target_model_i is not None and target_model_j is not None:
                # Find other series using the same models
                for other_series in self.series_names:
                    if other_series != target_series:
                        other_model_i = series_to_cluster_i.get(other_series)
                        other_model_j = series_to_cluster_j.get(other_series)
                        
                        # Check for synchronized transitions
                        if (target_model_i == other_model_i and target_model_j == other_model_j):
                            dependencies.append(
                                f"  - {target_series} and {other_series} show synchronized behavior: "
                                f"both use Θ{target_model_i}→Θ{target_model_j} during T{interval_i+1}→T{interval_j+1}"
                            )
                        elif (target_model_i != target_model_j and other_model_i != other_model_j):
                            # Both series switch models at the same time
                            dependencies.append(
                                f"  - {target_series} and {other_series} both change models during T{interval_i+1}→T{interval_j+1}: "
                                f"potential co-evolutionary pattern"
                            )
        
        return dependencies[:5]  # Limit to top 5 dependencies
    
    def _analyze_cross_dependencies(self) -> List[str]:
        """
        Analyze dependencies across all series
        
        Returns:
            List of cross-dependency descriptions
        """
        dependencies = []
        
        # Find model co-usage patterns
        model_cooccurrence = {}
        
        for interval_idx, result in enumerate(self.step1_results['intervals']):
            series_to_cluster = result['series_to_cluster']
            
            # Group series by model
            model_to_series = {}
            for series_name, cluster_id in series_to_cluster.items():
                if cluster_id not in model_to_series:
                    model_to_series[cluster_id] = []
                model_to_series[cluster_id].append(series_name)
            
            # Record co-occurrences
            for cluster_id, series_list in model_to_series.items():
                if len(series_list) > 1:
                    series_key = tuple(sorted(series_list))
                    if series_key not in model_cooccurrence:
                        model_cooccurrence[series_key] = []
                    model_cooccurrence[series_key].append((interval_idx, cluster_id))
        
        # Generate dependency descriptions
        for series_tuple, occurrences in model_cooccurrence.items():
            if len(occurrences) >= 2 and len(series_tuple) <= 4:  # Focus on strong, interpretable patterns
                series_names = ", ".join(series_tuple)
                models_used = [f"Θ{cluster_id}" for _, cluster_id in occurrences]
                intervals = [f"T{interval_idx + 1}" for interval_idx, _ in occurrences]
                
                dependencies.append(
                    f"  - Strong co-evolution: {series_names} share models "
                    f"{', '.join(set(models_used))} across intervals {', '.join(intervals[:3])}"
                )
        
        return sorted(dependencies, key=len, reverse=True)  # Sort by description length (complexity)
    
    def _analyze_global_model_usage(self) -> str:
        """
        Analyze global model usage patterns
        
        Returns:
            Model usage analysis string
        """
        model_usage = {}
        model_intervals = {}
        
        # Count model usage across all intervals and series
        for interval_idx, result in enumerate(self.step1_results['intervals']):
            series_to_cluster = result['series_to_cluster']
            
            for series_name, cluster_id in series_to_cluster.items():
                if cluster_id not in model_usage:
                    model_usage[cluster_id] = 0
                    model_intervals[cluster_id] = set()
                
                model_usage[cluster_id] += 1
                model_intervals[cluster_id].add(interval_idx)
        
        # Sort models by usage
        sorted_models = sorted(model_usage.items(), key=lambda x: x[1], reverse=True)
        
        usage_parts = []
        usage_parts.append(f"Total unique models discovered: {len(sorted_models)}")
        
        if sorted_models:
            # Most used models
            top_models = sorted_models[:5]
            usage_parts.append("Most frequently used models:")
            for model_id, count in top_models:
                intervals_active = len(model_intervals[model_id])
                usage_parts.append(f"  - Θ{model_id}: used by {count} series across {intervals_active} intervals")
        
        return "\n".join(usage_parts)
    
    def generate_prediction_scenarios(self, target_series: str, q: int = 5) -> Dict[str, str]:
        """
        Generate specific prediction scenarios for a target series
        
        Args:
            target_series: Series to generate predictions for
            q: Context window size
            
        Returns:
            Dict with different prediction scenarios
        """
        if target_series not in self.series_trajectories:
            return {"error": f"Series {target_series} not found in trajectories"}
        
        narrative = self._create_single_series_narrative(target_series, q)
        
        scenarios = {}
        
        # Scenario 1: Model maintenance prediction
        scenarios["maintain_model"] = self._generate_maintenance_scenario(narrative)
        
        # Scenario 2: Model switch prediction  
        scenarios["switch_model"] = self._generate_switch_scenario(narrative)
        
        # Scenario 3: Dependency-based prediction
        scenarios["dependency_based"] = self._generate_dependency_scenario(narrative)
        
        return scenarios
    
    def _generate_maintenance_scenario(self, narrative: SeriesNarrative) -> str:
        """Generate scenario where model is maintained"""
        if not narrative.transitions:
            return "Insufficient data for maintenance scenario"
        
        current_model = narrative.transitions[-1].to_model
        next_interval = narrative.transitions[-1].to_interval + 1
        
        return (f"MAINTENANCE SCENARIO:\n"
                f"If {narrative.series_name} continues with model Θ{current_model} at T{next_interval}, "
                f"this would indicate stability in the underlying dynamics. "
                f"This is likely if recent transitions show consistent model usage.")
    
    def _generate_switch_scenario(self, narrative: SeriesNarrative) -> str:
        """Generate scenario where model switches"""
        if not narrative.transitions:
            return "Insufficient data for switch scenario"
        
        current_model = narrative.transitions[-1].to_model
        next_interval = narrative.transitions[-1].to_interval + 1
        
        # Find alternative models this series has used
        used_models = set(t.to_model for t in narrative.transitions)
        used_models.update(t.from_model for t in narrative.transitions)
        alternative_models = [m for m in used_models if m != current_model]
        
        if alternative_models:
            alt_model = alternative_models[0]  # Just pick the first alternative
            return (f"SWITCH SCENARIO:\n"
                    f"If {narrative.series_name} switches from Θ{current_model} to Θ{alt_model} at T{next_interval}, "
                    f"this would indicate a regime change or adaptation to new conditions. "
                    f"This is likely if the series shows patterns of periodic model switching.")
        else:
            return (f"SWITCH SCENARIO:\n"
                    f"If {narrative.series_name} switches to a new model at T{next_interval}, "
                    f"this would represent exploration of new behavioral patterns.")
    
    def _generate_dependency_scenario(self, narrative: SeriesNarrative) -> str:
        """Generate scenario based on dependencies with other series"""
        dependencies = self._find_series_dependencies(narrative.series_name)
        
        if dependencies:
            return (f"DEPENDENCY SCENARIO:\n"
                    f"Prediction for {narrative.series_name} should consider co-evolutionary patterns:\n" +
                    "\n".join(dependencies))
        else:
            return (f"DEPENDENCY SCENARIO:\n"
                    f"No strong dependencies detected for {narrative.series_name}. "
                    f"Prediction can focus on individual series patterns.")
    
    def export_narratives(self, filepath: str, narratives: Dict[str, SeriesNarrative]):
        """
        Export narratives to JSON file
        
        Args:
            filepath: Path to save file
            narratives: Dictionary of narratives to export
        """
        export_data = {}
        
        for series_name, narrative in narratives.items():
            export_data[series_name] = {
                "pattern_summary": narrative.pattern_summary,
                "prediction_context": narrative.prediction_context,
                "transitions": [
                    {
                        "from_interval": t.from_interval,
                        "to_interval": t.to_interval,
                        "from_model": t.from_model,
                        "to_model": t.to_model,
                        "transition_type": t.transition_type
                    }
                    for t in narrative.transitions
                ]
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Narratives exported to {filepath}")


# Example usage function
def run_caarl_step3(step1_results: Dict, step2_results: Dict, target_series: str = None) -> Dict:
    """
    Run CAARL Step 3 on the results from Steps 1 and 2
    
    Args:
        step1_results: Results from CAARLStep1.fit()
        step2_results: Results from CAARLStep2.build_temporal_dependency_graph()
        target_series: Optional specific series to focus on
        
    Returns:
        step3_results: Results from serialization and narrative generation
    """
    # Initialize Step 3
    step3 = CAARLStep3(step1_results, step2_results)
    
    # Create narratives for all series
    print("Creating series narratives...")
    narratives = step3.create_series_narratives(q=5)
    
    # Generate LLM-ready serialization
    print("Generating LLM serialization...")
    if target_series:
        llm_serialization = step3.serialize_graph_for_llm(target_series=target_series, q=5)
        prediction_scenarios = step3.generate_prediction_scenarios(target_series, q=5)
    else:
        llm_serialization = step3.serialize_graph_for_llm(q=5)
        prediction_scenarios = {}
    
    # Export narratives
    step3.export_narratives('caarl_series_narratives.json', narratives)
    
    # Print example narrative
    if target_series and target_series in narratives:
        print(f"\nExample narrative for {target_series}:")
        print("=" * 60)
        print(llm_serialization)
    elif narratives:
        example_series = list(narratives.keys())[0]
        example_serialization = step3.serialize_graph_for_llm(target_series=example_series, q=5)
        print(f"\nExample narrative for {example_series}:")
        print("=" * 60) 
        print(example_serialization)
    
    return {
        'narratives': narratives,
        'llm_serialization': llm_serialization,
        'prediction_scenarios': prediction_scenarios,
        'step3_processor': step3
    }