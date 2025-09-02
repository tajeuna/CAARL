#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 14:24:44 2025

@author: tajeet01

CAARL Step 4: AI-Powered Model Transition Prediction Engine
Implementation using OpenAI/Anthropic APIs for prediction and explanation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import json
import os
import time
from datetime import datetime

# AI API imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: Anthropic library not installed. Install with: pip install anthropic")

@dataclass
class AIAPIConfig:
    """Configuration for AI API services"""
    provider: str  # 'openai' or 'anthropic'
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000


@dataclass
class PredictionResult:
    """
    Result of AI-powered model transition prediction
    """
    series_name: str
    current_model: int
    current_interval: int
    predicted_action: str  # 'maintain' or 'switch'
    predicted_model: Optional[int]
    confidence: float
    ai_reasoning: str
    raw_ai_response: str
    alternative_predictions: List[Dict]
    prediction_timestamp: str
    model_used: str

class CAARLStep4AIPredictor:
    """
    CAARL Step 4: AI-powered model transition prediction using LLM APIs
    """
    
    def __init__(self, step1_results: Dict, step2_results: Dict, step3_results: Dict, 
                 ai_config: AIAPIConfig):
        """
        Initialize Step 4 with AI configuration
        
        Args:
            step1_results: Output from CAARLStep1.fit()
            step2_results: Output from CAARLStep2.build_temporal_dependency_graph()
            step3_results: Output from CAARLStep3 narratives
            ai_config: AI API configuration
        """
        self.step1_results = step1_results
        self.step2_results = step2_results
        self.step3_results = step3_results
        self.narratives = step3_results['narratives']
        self.ai_config = ai_config
        
        # Initialize AI client
        self._initialize_ai_client()
        
        # Build context for better predictions
        self._build_prediction_context()
    
    def _initialize_ai_client(self):
        """Initialize the appropriate AI client"""
        if self.ai_config.provider == 'openai' and OPENAI_AVAILABLE:
            self.ai_client = openai.OpenAI(api_key=self.ai_config.api_key)
        elif self.ai_config.provider == 'anthropic' and ANTHROPIC_AVAILABLE:
            self.ai_client = anthropic.Anthropic(api_key=self.ai_config.api_key)
        else:
            raise ValueError(f"AI provider '{self.ai_config.provider}' not available or not installed")
    
    def _build_prediction_context(self):
        """Build context information for better AI predictions"""
        # Global statistics
        total_series = len(self.narratives)
        all_transitions = []
        model_usage = defaultdict(int)
        
        for narrative in self.narratives.values():
            all_transitions.extend(narrative.transitions)
            for transition in narrative.transitions:
                model_usage[transition.from_model] += 1
                model_usage[transition.to_model] += 1
        
        maintain_count = sum(1 for t in all_transitions if t.transition_type == 'maintain')
        switch_count = sum(1 for t in all_transitions if t.transition_type == 'switch')
        
        self.global_context = {
            'total_series': total_series,
            'total_transitions': len(all_transitions),
            'global_maintain_rate': maintain_count / len(all_transitions) if all_transitions else 0.5,
            'global_switch_rate': switch_count / len(all_transitions) if all_transitions else 0.5,
            'most_common_models': sorted(model_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def _create_prediction_prompt(self, target_series: str) -> str:
        """
        Create a comprehensive prompt for the AI model
        
        Args:
            target_series: Series to predict
            
        Returns:
            Formatted prompt string
        """
        if target_series not in self.narratives:
            raise ValueError(f"Series {target_series} not found in narratives")
        
        narrative = self.narratives[target_series]
        
        # Get current state
        if narrative.transitions:
            current_model = narrative.transitions[-1].to_model
            current_interval = narrative.transitions[-1].to_interval
            next_interval = current_interval + 1
        else:
            return f"Error: No transition data for series {target_series}"
        
        # Build comprehensive prompt
        prompt_parts = []
        
        prompt_parts.append("# CAARL Time Series Model Prediction Task")
        prompt_parts.append("")
        prompt_parts.append("You are an expert in co-evolving time series analysis using the CAARL framework.")
        prompt_parts.append("Your task is to predict whether a time series will maintain its current model or switch to a different model at the next time interval.")
        prompt_parts.append("")
        
        # Add global context
        prompt_parts.append("## Global Dataset Context")
        prompt_parts.append(f"- Total co-evolving series: {self.global_context['total_series']}")
        prompt_parts.append(f"- Total observed transitions: {self.global_context['total_transitions']}")
        prompt_parts.append(f"- Global maintain rate: {self.global_context['global_maintain_rate']:.1%}")
        prompt_parts.append(f"- Global switch rate: {self.global_context['global_switch_rate']:.1%}")
        
        most_common = self.global_context['most_common_models'][:5]
        prompt_parts.append(f"- Most common models: {', '.join([f'Θ{model}' for model, _ in most_common])}")
        prompt_parts.append("")
        
        # Add series-specific narrative
        prompt_parts.append(f"## Series-Specific Analysis for '{target_series}'")
        prompt_parts.append("")
        prompt_parts.append("### Pattern Summary")
        prompt_parts.append(narrative.pattern_summary)
        prompt_parts.append("")
        
        prompt_parts.append("### Recent Context")
        prompt_parts.append(narrative.prediction_context)
        prompt_parts.append("")
        
        # Add dependency information
        dependencies = self._get_dependency_info(target_series)
        if dependencies:
            prompt_parts.append("### Dependency Relationships")
            prompt_parts.extend(dependencies)
            prompt_parts.append("")
        
        # Add statistical patterns
        stats = self._get_series_statistics(target_series)
        prompt_parts.append("### Statistical Patterns")
        prompt_parts.extend(stats)
        prompt_parts.append("")
        
        # Add the prediction task
        prompt_parts.append("## Prediction Task")
        prompt_parts.append(f"Current State: Series '{target_series}' is using model Θ{current_model} at interval T{current_interval}")
        prompt_parts.append(f"Question: What will happen at interval T{next_interval}?")
        prompt_parts.append("")
        prompt_parts.append("Options:")
        prompt_parts.append(f"A) MAINTAIN: Continue using model Θ{current_model}")
        prompt_parts.append(f"B) SWITCH: Change to a different model")
        prompt_parts.append("")
        
        # Add output format requirements
        prompt_parts.append("## Required Output Format")
        prompt_parts.append("Please provide your prediction in the following JSON format:")
        prompt_parts.append("")
        prompt_parts.append("```json")
        prompt_parts.append("{")
        prompt_parts.append('  "prediction": "MAINTAIN" or "SWITCH",')
        prompt_parts.append(f'  "current_model": {current_model},')
        prompt_parts.append('  "predicted_model": <model_id_if_switching_else_current>,')
        prompt_parts.append('  "confidence": <0.0_to_1.0>,')
        prompt_parts.append('  "reasoning": "<detailed_explanation>",')
        prompt_parts.append('  "key_factors": [')
        prompt_parts.append('    "<factor_1>",')
        prompt_parts.append('    "<factor_2>",')
        prompt_parts.append('    "<factor_3>"')
        prompt_parts.append('  ],')
        prompt_parts.append('  "alternative_scenario": {')
        prompt_parts.append('    "action": "<alternative_action>",')
        prompt_parts.append('    "model": <alternative_model>,')
        prompt_parts.append('    "probability": <0.0_to_1.0>')
        prompt_parts.append('  }')
        prompt_parts.append("}")
        prompt_parts.append("```")
        prompt_parts.append("")
        prompt_parts.append("Focus on:")
        prompt_parts.append("1. Recent behavioral patterns and trends")
        prompt_parts.append("2. Historical model usage preferences")
        prompt_parts.append("3. Co-evolutionary dependencies with other series")
        prompt_parts.append("4. Statistical indicators of regime changes")
        prompt_parts.append("5. Model performance and stability patterns")
        
        return "\n".join(prompt_parts)
    
    def _get_dependency_info(self, target_series: str) -> List[str]:
        """Get dependency information for the target series"""
        dependencies = []
        
        # Look through step2 results for dependencies
        if 'dependencies' in self.step2_results:
            for dep in self.step2_results['dependencies']:
                if target_series in dep['series_involved']:
                    other_series = [s for s in dep['series_involved'] if s != target_series]
                    if other_series:
                        dependencies.append(
                            f"- Co-evolves with {', '.join(other_series[:3])}: "
                            f"{dep['transition_type']} pattern (strength: {dep['strength']})"
                        )
        
        return dependencies[:5]  # Limit to top 5
    
    def _get_series_statistics(self, target_series: str) -> List[str]:
        """Get statistical patterns for the series"""
        narrative = self.narratives[target_series]
        stats = []
        
        if narrative.transitions:
            transitions = narrative.transitions
            
            # Calculate statistics
            maintain_count = sum(1 for t in transitions if t.transition_type == 'maintain')
            switch_count = sum(1 for t in transitions if t.transition_type == 'switch')
            total = len(transitions)
            
            # Model usage frequency
            model_usage = Counter()
            for t in transitions:
                model_usage[t.from_model] += 1
                model_usage[t.to_model] += 1
            top_models = model_usage.most_common(3)
            
            # Recent behavior analysis
            recent_transitions = transitions[-5:]
            recent_maintains = sum(1 for t in recent_transitions if t.transition_type == 'maintain')
            
            stats.append(f"- Historical maintain rate: {maintain_count/total:.1%} ({maintain_count}/{total})")
            stats.append(f"- Historical switch rate: {switch_count/total:.1%} ({switch_count}/{total})")
            stats.append(f"- Recent maintain rate (last 5): {recent_maintains/len(recent_transitions):.1%}")
            stats.append(f"- Most used models: {', '.join([f'Θ{m}({c})' for m, c in top_models])}")
            
            # Consecutive pattern analysis
            if len(transitions) >= 2:
                consecutive_maintains = 0
                for t in reversed(transitions):
                    if t.transition_type == 'maintain':
                        consecutive_maintains += 1
                    else:
                        break
                stats.append(f"- Current stability streak: {consecutive_maintains} consecutive maintains")
        
        return stats
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            response = self.ai_client.chat.completions.create(
                model=self.ai_config.model,
                messages=[
                    {"role": "system", "content": "You are an expert time series analyst specializing in co-evolving sequences and the CAARL framework. Provide detailed, data-driven predictions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.ai_config.temperature,
                max_tokens=self.ai_config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")
    
    def _call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API"""
        try:
            response = self.ai_client.messages.create(
                model=self.ai_config.model,
                max_tokens=self.ai_config.max_tokens,
                temperature=self.ai_config.temperature,
                system="You are an expert time series analyst specializing in co-evolving sequences and the CAARL framework. Provide detailed, data-driven predictions.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {str(e)}")
    
    def _call_ai_api(self, prompt: str) -> str:
        """Call the configured AI API"""
        if self.ai_config.provider == 'openai':
            return self._call_openai_api(prompt)
        elif self.ai_config.provider == 'anthropic':
            return self._call_anthropic_api(prompt)
        else:
            raise ValueError(f"Unsupported AI provider: {self.ai_config.provider}")
    
    def _parse_ai_response(self, response: str, target_series: str) -> PredictionResult:
        """Parse AI response and create PredictionResult"""
        narrative = self.narratives[target_series]
        current_model = narrative.transitions[-1].to_model if narrative.transitions else 0
        current_interval = narrative.transitions[-1].to_interval if narrative.transitions else 0
        
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                parsed = json.loads(json_str)
                
                # Extract prediction details
                predicted_action = parsed.get('prediction', 'MAINTAIN').lower()
                predicted_model = parsed.get('predicted_model', current_model)
                confidence = float(parsed.get('confidence', 0.5))
                reasoning = parsed.get('reasoning', 'No reasoning provided')
                key_factors = parsed.get('key_factors', [])
                alternative = parsed.get('alternative_scenario', {})
                
                # Format reasoning with key factors
                formatted_reasoning = f"PREDICTION: {predicted_action.upper()}"
                if predicted_action == 'switch':
                    formatted_reasoning += f" to model Θ{predicted_model}"
                formatted_reasoning += f" (confidence: {confidence:.1%})\n\n"
                formatted_reasoning += f"REASONING:\n{reasoning}\n\n"
                
                if key_factors:
                    formatted_reasoning += "KEY FACTORS:\n"
                    for factor in key_factors:
                        formatted_reasoning += f"• {factor}\n"
                
                alternatives = []
                if alternative:
                    alt_action = alternative.get('action', 'maintain').lower()
                    alt_model = alternative.get('model', current_model)
                    alt_prob = alternative.get('probability', 1.0 - confidence)
                    alternatives.append({
                        'action': alt_action,
                        'model': alt_model,
                        'confidence': alt_prob
                    })
                
                return PredictionResult(
                    series_name=target_series,
                    current_model=current_model,
                    current_interval=current_interval,
                    predicted_action=predicted_action,
                    predicted_model=predicted_model,
                    confidence=confidence,
                    ai_reasoning=formatted_reasoning,
                    raw_ai_response=response,
                    alternative_predictions=alternatives,
                    prediction_timestamp=datetime.now().isoformat(),
                    model_used=f"{self.ai_config.provider}:{self.ai_config.model}"
                )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Could not parse AI response JSON: {e}")
        
        # Fallback parsing
        predicted_action = 'maintain' if 'MAINTAIN' in response.upper() else 'switch'
        confidence = 0.6  # Default confidence
        
        return PredictionResult(
            series_name=target_series,
            current_model=current_model,
            current_interval=current_interval,
            predicted_action=predicted_action,
            predicted_model=current_model,
            confidence=confidence,
            ai_reasoning=f"AI PREDICTION (fallback parsing):\n{response}",
            raw_ai_response=response,
            alternative_predictions=[],
            prediction_timestamp=datetime.now().isoformat(),
            model_used=f"{self.ai_config.provider}:{self.ai_config.model}"
        )
    
    def predict_next_transition(self, target_series: str, retries: int = 3) -> PredictionResult:
        """
        Predict the next transition for a target series using AI
        
        Args:
            target_series: Name of the series to predict
            retries: Number of retries if API call fails
            
        Returns:
            PredictionResult object with AI prediction
        """
        print(f"Generating AI prediction for series: {target_series}")
        
        # Create prompt
        prompt = self._create_prediction_prompt(target_series)
        
        # Call AI API with retries
        for attempt in range(retries):
            try:
                ai_response = self._call_ai_api(prompt)
                break
            except Exception as e:
                if attempt < retries - 1:
                    print(f"API call failed (attempt {attempt + 1}/{retries}): {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise RuntimeError(f"All API attempts failed: {e}")
        
        # Parse response
        prediction = self._parse_ai_response(ai_response, target_series)
        
        print(f"Prediction completed: {prediction.predicted_action} (confidence: {prediction.confidence:.1%})")
        
        return prediction
    
    def generate_prediction_narrative(self, target_series: str) -> str:
        """
        Generate complete prediction narrative including AI analysis
        
        Args:
            target_series: Series to predict
            
        Returns:
            Complete narrative string
        """
        if target_series not in self.narratives:
            return f"Error: No narrative available for series {target_series}"
        
        narrative = self.narratives[target_series]
        
        # Generate AI prediction
        prediction = self.predict_next_transition(target_series)
        
        # Build complete narrative
        narrative_parts = []
        
        narrative_parts.append(f"=== TEMPORAL ANALYSIS FOR {target_series} ===")
        narrative_parts.append("")
        
        narrative_parts.append("PATTERN SUMMARY:")
        narrative_parts.append(narrative.pattern_summary)
        narrative_parts.append("")
        
        narrative_parts.append("RECENT CONTEXT:")
        narrative_parts.append(narrative.prediction_context)
        narrative_parts.append("")
        
        # Add dependency relationships
        dependencies = self._get_dependency_info(target_series)
        if dependencies:
            narrative_parts.append("DEPENDENCY RELATIONSHIPS:")
            narrative_parts.extend(dependencies)
            narrative_parts.append("")
        
        # Add prediction task
        next_interval = prediction.current_interval + 1
        narrative_parts.append("PREDICTION TASK:")
        narrative_parts.append(f"Given the above context, predict whether {target_series} will:")
        narrative_parts.append(f"  A) Continue using model Θ{prediction.current_model} at interval T{next_interval}")
        narrative_parts.append(f"  B) Switch to a different model at interval T{next_interval}")
        narrative_parts.append("")
        narrative_parts.append("If switching, specify which model and explain the reasoning based on the patterns observed.")
        narrative_parts.append("")
        
        # Add AI prediction
        narrative_parts.append("=== AI PREDICTION RESULT ===")
        narrative_parts.append(f"Model: {prediction.model_used}")
        narrative_parts.append(f"Timestamp: {prediction.prediction_timestamp}")
        narrative_parts.append("")
        narrative_parts.append(prediction.ai_reasoning)
        
        if prediction.alternative_predictions:
            narrative_parts.append("")
            narrative_parts.append("ALTERNATIVE SCENARIOS:")
            for alt in prediction.alternative_predictions:
                alt_action = alt['action'].upper()
                alt_model = alt['model']
                alt_conf = alt['confidence']
                narrative_parts.append(f"  - {alt_action} Θ{alt_model}: {alt_conf:.1%} confidence")
        
        return "\n".join(narrative_parts)
    
    def batch_predict(self, series_list: Optional[List[str]] = None, delay: float = 1.0) -> Dict[str, PredictionResult]:
        """
        Generate AI predictions for multiple series
        
        Args:
            series_list: List of series to predict. If None, predict all series.
            delay: Delay between API calls to respect rate limits
            
        Returns:
            Dictionary mapping series names to prediction results
        """
        if series_list is None:
            series_list = list(self.narratives.keys())
        
        predictions = {}
        total = len(series_list)
        
        for i, series_name in enumerate(series_list, 1):
            print(f"Predicting {i}/{total}: {series_name}")
            try:
                predictions[series_name] = self.predict_next_transition(series_name)
                
                # Rate limiting
                if i < total:  # Don't delay after the last prediction
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Failed to predict for {series_name}: {e}")
                # Create error prediction
                predictions[series_name] = PredictionResult(
                    series_name=series_name,
                    current_model=None,
                    current_interval=None,
                    predicted_action='error',
                    predicted_model=None,
                    confidence=0.0,
                    ai_reasoning=f"Prediction failed: {str(e)}",
                    raw_ai_response="",
                    alternative_predictions=[],
                    prediction_timestamp=datetime.now().isoformat(),
                    model_used=f"{self.ai_config.provider}:{self.ai_config.model}"
                )
        
        return predictions
    
    def export_predictions(self, predictions: Dict[str, PredictionResult], filepath: str):
        """
        Export AI predictions to JSON file
        
        Args:
            predictions: Dictionary of prediction results
            filepath: File path to save predictions
        """
        export_data = {
            'metadata': {
                'ai_provider': self.ai_config.provider,
                'ai_model': self.ai_config.model,
                'prediction_timestamp': datetime.now().isoformat(),
                'total_predictions': len(predictions)
            },
            'predictions': {}
        }
        
        for series_name, prediction in predictions.items():
            export_data['predictions'][series_name] = {
                'current_model': prediction.current_model,
                'current_interval': prediction.current_interval,
                'predicted_action': prediction.predicted_action,
                'predicted_model': prediction.predicted_model,
                'confidence': float(prediction.confidence) if prediction.confidence else None,
                'ai_reasoning': prediction.ai_reasoning,
                'raw_ai_response': prediction.raw_ai_response,
                'alternatives': prediction.alternative_predictions,
                'prediction_timestamp': prediction.prediction_timestamp,
                'model_used': prediction.model_used
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"AI predictions exported to {filepath}")


# Factory function to create AI configuration
def create_ai_config(provider: str = "openai", model: str = None, api_key: str = None) -> AIAPIConfig:
    """
    Create AI configuration with sensible defaults
    
    Args:
        provider: 'openai' or 'anthropic'
        model: Model name (uses defaults if None)
        api_key: API key (tries environment variables if None)
        
    Returns:
        AIAPIConfig object
    """
    # Set default models
    if model is None:
        if provider == "openai":
            model = "gpt-4"  # or "gpt-3.5-turbo" for faster/cheaper predictions
        elif provider == "anthropic":
            model = "claude-3-sonnet-20240229"  # or "claude-3-haiku-20240307" for faster predictions
    
    # Get API key from environment if not provided
    if api_key is None:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        raise ValueError(f"API key for {provider} not provided and not found in environment variables")
    
    return AIAPIConfig(
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=0.7,
        max_tokens=1500
    )


# Example usage function
def run_caarl_step4_ai(step1_results: Dict, step2_results: Dict, step3_results: Dict, 
                      ai_config: AIAPIConfig, target_series: str = None) -> Dict:
    """
    Run CAARL Step 4 with AI-powered predictions
    
    Args:
        step1_results: Results from CAARLStep1.fit()
        step2_results: Results from CAARLStep2.build_temporal_dependency_graph()
        step3_results: Results from CAARLStep3 narratives
        ai_config: AI API configuration
        target_series: Optional specific series to predict
        
    Returns:
        step4_results: Results from AI prediction engine
    """
    print("=" * 60)
    print("CAARL STEP 4: AI-POWERED MODEL TRANSITION PREDICTION")
    print("=" * 60)
    print(f"Using AI Provider: {ai_config.provider}")
    print(f"Using Model: {ai_config.model}")
    print("=" * 60)
    
    # Initialize AI predictor
    step4 = CAARLStep4AIPredictor(step1_results, step2_results, step3_results, ai_config)
    
    if target_series:
        # Generate prediction for specific series
        prediction = step4.predict_next_transition(target_series)
        narrative = step4.generate_prediction_narrative(target_series)
        
        print(f"\nPrediction generated for series: {target_series}")
        print("\nCOMPLETE AI PREDICTION NARRATIVE:")
        print("=" * 60)
        print(narrative)
        
        predictions = {target_series: prediction}
        
    else:
        # Generate predictions for all series (with rate limiting)
        print("Generating AI predictions for all series...")
        predictions = step4.batch_predict(delay=1.0)  # 1 second delay between calls
        
        # Show summary
        successful_predictions = {k: v for k, v in predictions.items() if v.predicted_action != 'error'}
        maintain_count = sum(1 for p in successful_predictions.values() if p.predicted_action == 'maintain')
        switch_count = sum(1 for p in successful_predictions.values() if p.predicted_action == 'switch')
        avg_confidence = np.mean([p.confidence for p in successful_predictions.values() if p.confidence])
        
        print(f"\nAI PREDICTION SUMMARY:")
        print(f"  - Total series predicted: {len(successful_predictions)}")
        print(f"  - Failed predictions: {len(predictions) - len(successful_predictions)}")
        print(f"  - Maintain predictions: {maintain_count}")
        print(f"  - Switch predictions: {switch_count}")
        print(f"  - Average confidence: {avg_confidence:.1%}")
        
        # Show example narrative for first successful series
        if successful_predictions:
            example_series = list(successful_predictions.keys())[0]
            example_narrative = step4.generate_prediction_narrative(example_series)
            print(f"\nEXAMPLE AI PREDICTION NARRATIVE FOR {example_series}:")
            print("=" * 60)
            print(example_narrative)
    
    # Export predictions
    step4.export_predictions(predictions, f'caarl_ai_predictions_{ai_config.provider}.json')
    
    return {
        'predictions': predictions,
        'step4_processor': step4,
        'ai_config': ai_config,
        'prediction_narratives': {
            series: step4.generate_prediction_narrative(series) 
            for series in predictions.keys()
            if predictions[series].predicted_action != 'error'
        }
    }


# Example usage:
if __name__ == "__main__":
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
    
    # Use in the pipeline:
    # results = run_caarl_step4_ai(step1_results, step2_results, step3_results, openai_config, "series_0")