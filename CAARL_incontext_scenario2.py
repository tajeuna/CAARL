#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 07:42:57 2025

@author: tajeet01
"""


    


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, pacf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from CAARL_DataHandler import DataHandler
warnings.filterwarnings('ignore')

class AutoRegressiveModel:
    """
    Autoregressive model class for time series approximation
    """
    def __init__(self, lag: int = 2):
        self.lag = lag
        self.coefficients = None
        self.intercept = None
        self.fitted = False
        self.model_id = None
        
    def fit(self, series: np.ndarray) -> float:
        """
        Fit AR model to time series segment
        
        Args:
            series: Time series values for the segment
            
        Returns:
            mse: Mean squared error of the fit
        """
        if len(series) <= self.lag:
            return float('inf')
            
        # Create lagged features
        X = []
        y = []
        
        for i in range(self.lag, len(series)):
            X.append(series[i-self.lag:i])
            y.append(series[i])
            
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
            return float('inf')
            
        # Fit linear regression (AR model)
        model = LinearRegression()
        model.fit(X, y)
        
        self.coefficients = model.coef_
        self.intercept = model.intercept_
        self.fitted = True
        
        # Calculate MSE
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        
        return mse
        
    def predict(self, history: np.ndarray) -> float:
        """
        Predict next value given history
        
        Args:
            history: Array of lag previous values
            
        Returns:
            prediction: Predicted next value
        """
        if not self.fitted or len(history) != self.lag:
            return 0.0
            
        return np.dot(self.coefficients, history) + self.intercept
        
    def calculate_similarity(self, other_model: 'AutoRegressiveModel', tolerance: float = 0.1) -> float:
        """
        Calculate similarity between two AR models based on coefficients
        
        Args:
            other_model: Another AR model to compare with
            tolerance: Tolerance for coefficient similarity
            
        Returns:
            similarity: Similarity score (0 to 1, higher is more similar)
        """
        if not (self.fitted and other_model.fitted):
            return 0.0
            
        if len(self.coefficients) != len(other_model.coefficients):
            return 0.0
            
        # Calculate coefficient similarity
        coeff_diff = np.abs(self.coefficients - other_model.coefficients)
        intercept_diff = abs(self.intercept - other_model.intercept)
        
        # Normalize differences
        max_coeff_diff = np.max([np.abs(self.coefficients), np.abs(other_model.coefficients)])
        max_intercept_diff = max(abs(self.intercept), abs(other_model.intercept))
        
        if max_coeff_diff == 0:
            coeff_similarity = 1.0
        else:
            coeff_similarity = 1.0 - np.mean(coeff_diff) / max_coeff_diff
            
        if max_intercept_diff == 0:
            intercept_similarity = 1.0
        else:
            intercept_similarity = 1.0 - intercept_diff / max_intercept_diff
            
        overall_similarity = 0.8 * coeff_similarity + 0.2 * intercept_similarity
        
        return max(0.0, overall_similarity)

class CAARLParameterSelector:
    """
    Statistical framework for selecting optimal CAARL parameters
    """
    
    def __init__(self):
        self.results = {}
    
    def analyze_data_characteristics(self, df: pd.DataFrame) -> Dict:
        """
        Analyze time series data to inform parameter selection
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            characteristics: Dictionary with data analysis results
        """
        n_series, n_timesteps = df.shape[1], df.shape[0]
        
        characteristics = {
            'n_series': n_series,
            'n_timesteps': n_timesteps,
            'frequency_hint': self.infer_frequency(df),
            'stationarity_results': {},
            'autocorrelation_analysis': {},
            'volatility_analysis': {}
        }
        
        # Analyze each series
        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 50:
                continue
                
            # Stationarity test
            try:
                adf_result = adfuller(series)
                is_stationary = adf_result[1] < 0.05  # p-value < 0.05
            except:
                is_stationary = False
                adf_result = (0, 0.5, None, None, None, None)
                
            # Autocorrelation analysis
            try:
                pacf_values = pacf(series, nlags=min(20, len(series)//4), method='ols')
                significant_lags = self.find_significant_pacf_lags(pacf_values)
            except:
                significant_lags = [1, 2]
            
            # Volatility analysis
            try:
                volatility = np.std(series.diff().dropna())
            except:
                volatility = np.std(series)
            
            characteristics['stationarity_results'][col] = {
                'is_stationary': is_stationary,
                'adf_pvalue': adf_result[1],
                'adf_statistic': adf_result[0]
            }
            
            characteristics['autocorrelation_analysis'][col] = {
                'suggested_lags': significant_lags,
                'max_significant_lag': max(significant_lags) if significant_lags else 2
            }
            
            characteristics['volatility_analysis'][col] = {
                'volatility': volatility,
                'volatility_percentile': self.calculate_volatility_percentile(volatility, df)
            }
        
        return characteristics
    
    def infer_frequency(self, df: pd.DataFrame) -> str:
        """Infer the likely frequency of the time series"""
        n_points = len(df)
        
        if n_points > 8000:  # Very high frequency
            return "minute_or_second"
        elif n_points > 2000:  # High frequency
            return "hourly"
        elif n_points > 500:   # Medium frequency
            return "daily"
        elif n_points > 100:   # Low frequency
            return "weekly"
        else:                  # Very low frequency
            return "monthly"
    
    def find_significant_pacf_lags(self, pacf_values: np.ndarray, threshold: float = 0.1) -> List[int]:
        """Find statistically significant lags in PACF"""
        significant_lags = []
        
        for i, value in enumerate(pacf_values[1:], 1):  # Skip lag 0
            if abs(value) > threshold:
                significant_lags.append(i)
                
        return significant_lags if significant_lags else [1, 2]  # Default fallback
    
    def calculate_volatility_percentile(self, volatility: float, df: pd.DataFrame) -> float:
        """Calculate volatility percentile compared to all series"""
        all_volatilities = []
        for col in df.columns:
            series = df[col].dropna()
            if len(series) > 10:
                try:
                    vol = np.std(series.diff().dropna())
                    if not np.isnan(vol):
                        all_volatilities.append(vol)
                except:
                    continue
        
        if all_volatilities and len(all_volatilities) > 0:
            return (sum(v < volatility for v in all_volatilities) / len(all_volatilities)) * 100
        return 50.0
    
    def suggest_interval_size(self, characteristics: Dict) -> Tuple[int, str]:
        """
        Suggest optimal interval size based on data characteristics
        
        Args:
            characteristics: Results from analyze_data_characteristics
            
        Returns:
            suggested_size: Recommended interval size
            reasoning: Explanation of the choice
        """
        n_timesteps = characteristics['n_timesteps']
        frequency = characteristics['frequency_hint']
        
        # Base recommendations by frequency
        size_map = {
            "minute_or_second": min(max(480, n_timesteps // 20), 1440),  # 8-24 hours
            "hourly": min(max(168, n_timesteps // 15), 720),             # 1-4 weeks  
            "daily": min(max(60, n_timesteps // 20), 180),               # 2-6 months
            "weekly": min(max(12, n_timesteps // 10), 52),               # 3-12 months
            "monthly": min(max(12, n_timesteps // 8), 60)                # 1-5 years
        }
        
        base_size = size_map.get(frequency, max(50, n_timesteps // 15))
        
        # Adjust based on stationarity
        if characteristics['stationarity_results']:
            stationary_ratio = sum(1 for result in characteristics['stationarity_results'].values() 
                                  if result['is_stationary']) / max(1, len(characteristics['stationarity_results']))
            
            if stationary_ratio < 0.3:  # Mostly non-stationary
                base_size = int(base_size * 0.7)  # Smaller intervals
                reasoning = f"Reduced interval size due to low stationarity ({stationary_ratio:.1%})"
            elif stationary_ratio > 0.8:  # Mostly stationary
                base_size = int(base_size * 1.2)  # Larger intervals possible
                reasoning = f"Increased interval size due to high stationarity ({stationary_ratio:.1%})"
            else:
                reasoning = f"Standard interval size for {frequency} data with {stationary_ratio:.1%} stationarity"
        else:
            reasoning = f"Standard interval size for {frequency} data (no stationarity analysis available)"
        
        return max(50, base_size), reasoning
    
    def suggest_lag(self, characteristics: Dict) -> Tuple[int, str]:
        """
        Suggest optimal lag based on PACF analysis
        
        Args:
            characteristics: Results from analyze_data_characteristics
            
        Returns:
            suggested_lag: Recommended lag
            reasoning: Explanation of the choice
        """
        if not characteristics['autocorrelation_analysis']:
            return 2, "Default lag due to insufficient autocorrelation analysis"
        
        # Collect all suggested lags
        all_suggested_lags = []
        for analysis in characteristics['autocorrelation_analysis'].values():
            all_suggested_lags.extend(analysis['suggested_lags'])
        
        if not all_suggested_lags:
            return 2, "Default lag due to no significant autocorrelations found"
        
        # Use median of significant lags, capped at reasonable values
        median_lag = int(np.median(all_suggested_lags))
        suggested_lag = np.clip(median_lag, 1, 10)  # Cap between 1 and 10
        
        reasoning = f"Based on median significant lag ({median_lag}) from PACF analysis across {len(characteristics['autocorrelation_analysis'])} series"
        
        return suggested_lag, reasoning
    
    def suggest_stride(self, interval_size: int, characteristics: Dict, 
                      change_detection_priority: str = "medium") -> Tuple[int, str]:
        """
        Suggest optimal stride based on interval size and change detection needs
        
        Args:
            interval_size: Chosen interval size
            characteristics: Data characteristics
            change_detection_priority: "high", "medium", or "low"
            
        Returns:
            suggested_stride: Recommended stride
            reasoning: Explanation of the choice
        """
        # Base stride based on priority
        if change_detection_priority == "high":
            base_stride = interval_size // 4  # 75% overlap
            overlap_desc = "high overlap (75%)"
        elif change_detection_priority == "medium":
            base_stride = interval_size // 2  # 50% overlap
            overlap_desc = "medium overlap (50%)"
        else:  # "low"
            base_stride = int(interval_size * 0.75)  # 25% overlap
            overlap_desc = "low overlap (25%)"
        
        # Adjust based on volatility
        if characteristics['volatility_analysis']:
            avg_volatility_percentile = np.mean([
                analysis['volatility_percentile'] 
                for analysis in characteristics['volatility_analysis'].values()
                if not np.isnan(analysis['volatility_percentile'])
            ])
            
            if avg_volatility_percentile > 75:  # High volatility data
                base_stride = max(base_stride // 2, 1)  # Increase overlap
                reasoning = f"Reduced stride for high volatility data ({overlap_desc} -> higher overlap)"
            elif avg_volatility_percentile < 25:  # Low volatility data
                base_stride = min(int(base_stride * 1.5), interval_size - 1)  # Decrease overlap
                reasoning = f"Increased stride for low volatility data ({overlap_desc} -> lower overlap)"
            else:
                reasoning = f"Standard stride with {overlap_desc} for {change_detection_priority} change detection priority"
        else:
            reasoning = f"Standard stride with {overlap_desc}"
        
        return max(1, base_stride), reasoning
    
    def recommend_parameters(self, df: pd.DataFrame, 
                           change_detection_priority: str = "medium") -> Dict:
        """
        Complete parameter recommendation pipeline
        
        Args:
            df: Time series DataFrame
            change_detection_priority: Priority for detecting regime changes
            
        Returns:
            recommendations: Complete parameter recommendations with reasoning
        """
        # Step 1: Analyze data
        characteristics = self.analyze_data_characteristics(df)
        
        # Step 2: Get recommendations
        interval_size, interval_reasoning = self.suggest_interval_size(characteristics)
        lag, lag_reasoning = self.suggest_lag(characteristics)
        stride, stride_reasoning = self.suggest_stride(interval_size, characteristics, 
                                                     change_detection_priority)
        
        # Validate constraints
        if lag >= interval_size // 20:
            lag = max(1, interval_size // 20)
            lag_reasoning += f" (adjusted to maintain interval_size/lag ratio > 20)"
        
        if stride >= interval_size:
            stride = interval_size // 2
            stride_reasoning += f" (adjusted to be less than interval size)"
        
        recommendations = {
            'interval_size': interval_size,
            'lag': lag,
            'stride': stride,
            'reasoning': {
                'interval_size': interval_reasoning,
                'lag': lag_reasoning,
                'stride': stride_reasoning
            },
            'data_characteristics': characteristics,
            'validation': {
                'min_points_per_interval': interval_size,
                'min_points_for_ar_fitting': lag * 20,
                'ratio_check': interval_size / lag,
                'total_intervals_expected': max(1, (len(df) - interval_size) // stride + 1)
            }
        }
        
        return recommendations

class CAARLStep1:
    """
    CAARL Step 1: Model Identification and Clustering Implementation
    Includes integrated parameter selection based on statistical analysis
    """
    def __init__(self, lag: int = None, stride: int = None, regularization_tau: float = 0.1, 
                 auto_params: bool = True, change_detection_priority: str = "medium"):
        """
        Initialize CAARL Step 1
        
        Args:
            lag: Number of previous time steps to use for AR modeling (auto-selected if None)
            stride: Stride for moving through time intervals (auto-selected if None)
            regularization_tau: Regularization parameter for clustering
            auto_params: Whether to automatically select parameters based on data
            change_detection_priority: "high", "medium", or "low" - affects stride selection
        """
        self.lag = lag
        self.stride = stride
        self.tau = regularization_tau
        self.auto_params = auto_params
        self.change_detection_priority = change_detection_priority
        
        # Parameter selector
        self.param_selector = CAARLParameterSelector()
        self.parameter_recommendations = None
        
        # Results storage
        self.models_per_interval = {}  # {interval: {series_name: AR_model}}
        self.clustered_models = {}     # {interval: {cluster_id: representative_model}}
        self.series_to_cluster = {}    # {interval: {series_name: cluster_id}}
        self.model_performance = {}    # {interval: {series_name: mse}}
        
    def auto_select_parameters(self, df: pd.DataFrame, interval_length: int = None) -> Dict:
        """
        Automatically select optimal parameters based on statistical analysis
        
        Args:
            df: DataFrame with time series data
            interval_length: Override for interval length (if None, will be auto-selected)
            
        Returns:
            parameter_info: Dictionary with selected parameters and reasoning
        """
        print("🔍 AUTOMATIC PARAMETER SELECTION")
        print("="*60)
        
        # Get recommendations
        recommendations = self.param_selector.recommend_parameters(df, self.change_detection_priority)
        
        # Use provided interval_length if specified, otherwise use recommendation
        if interval_length is not None:
            recommendations['interval_size'] = interval_length
            recommendations['reasoning']['interval_size'] = "User-specified interval size"
        
        # Update instance parameters
        if self.lag is None:
            self.lag = recommendations['lag']
        if self.stride is None:
            self.stride = recommendations['stride']
            
        self.parameter_recommendations = recommendations
        
        # Print selected parameters
        print(f"\n📊 Data Summary:")
        print(f"  - Shape: {df.shape}")
        print(f"  - Inferred frequency: {recommendations['data_characteristics']['frequency_hint']}")
        if recommendations['data_characteristics']['stationarity_results']:
            stationary_count = sum(1 for r in recommendations['data_characteristics']['stationarity_results'].values() if r['is_stationary'])
            total_count = len(recommendations['data_characteristics']['stationarity_results'])
            print(f"  - Stationary series: {stationary_count}/{total_count}")
        
        print(f"\n⚙️ Selected Parameters:")
        print(f"  - Interval Size: {recommendations['interval_size']}")
        print(f"    💡 {recommendations['reasoning']['interval_size']}")
        print(f"  - Lag: {self.lag}")
        print(f"    💡 {recommendations['reasoning']['lag']}")
        print(f"  - Stride: {self.stride}")
        print(f"    💡 {recommendations['reasoning']['stride']}")
        
        print(f"\n✅ Validation Checks:")
        print(f"  - Ratio (interval/lag): {recommendations['validation']['ratio_check']:.1f} (should be > 10)")
        print(f"  - Expected intervals: {recommendations['validation']['total_intervals_expected']}")
        print(f"  - Points per AR fit: {recommendations['interval_size'] - self.lag} (should be > {self.lag * 10})")
        
        return recommendations
        
    def create_time_intervals(self, df: pd.DataFrame, interval_length: int) -> List[Tuple[int, int]]:
        """
        Create time intervals based on stride
        
        Args:
            df: DataFrame with time series data
            interval_length: Length of each time interval
            
        Returns:
            intervals: List of (start, end) tuples for each interval
        """
        intervals = []
        total_length = len(df)
        
        start = 0
        while start + interval_length <= total_length:
            end = start + interval_length
            intervals.append((start, end))
            start += self.stride
            
        return intervals
    
    def fit_ar_models_for_interval(self, df: pd.DataFrame, interval: Tuple[int, int]) -> Dict[str, Tuple[AutoRegressiveModel, float]]:
        """
        Fit AR models for all series in a given time interval
        
        Args:
            df: DataFrame with time series data
            interval: (start, end) tuple defining the interval
            
        Returns:
            models_and_performance: Dict mapping series names to (model, mse) tuples
        """
        start, end = interval
        interval_data = df.iloc[start:end]
        
        models_and_performance = {}
        
        for series_name in df.columns:
            series_values = interval_data[series_name].values
            
            # Skip if series has insufficient data or too many NaNs
            if len(series_values) <= self.lag or np.isnan(series_values).sum() > len(series_values) * 0.3:
                continue
                
            # Handle NaN values by forward fill and backward fill
            series_values = pd.Series(series_values).fillna(method='ffill').fillna(method='bfill').values
            
            # Create and fit AR model
            ar_model = AutoRegressiveModel(lag=self.lag)
            mse = ar_model.fit(series_values)
            
            if mse != float('inf'):
                models_and_performance[series_name] = (ar_model, mse)
                
        return models_and_performance
    
    def cluster_models(self, models_and_performance: Dict[str, Tuple[AutoRegressiveModel, float]], 
                      df: pd.DataFrame, interval: Tuple[int, int]) -> Tuple[Dict[int, AutoRegressiveModel], Dict[str, int]]:
        """
        Cluster AR models based on predictive equivalence (not parameter similarity)
        Two models belong to the same cluster if they can successfully predict the same time series
        
        Args:
            models_and_performance: Dict mapping series names to (model, mse) tuples
            df: Original dataframe to test cross-prediction
            interval: Current time interval for data extraction
            
        Returns:
            clustered_models: Dict mapping cluster IDs to representative models
            series_to_cluster: Dict mapping series names to cluster IDs
        """
        if len(models_and_performance) <= 1:
            if len(models_and_performance) == 1:
                series_name = list(models_and_performance.keys())[0]
                model = models_and_performance[series_name][0]
                return {0: model}, {series_name: 0}
            else:
                return {}, {}
        
        series_names = list(models_and_performance.keys())
        models = [models_and_performance[name][0] for name in series_names]
        
        # Extract interval data
        start, end = interval
        interval_data = df.iloc[start:end]
        
        # Create cross-prediction matrix: can model i predict series j?
        n_series = len(series_names)
        cross_prediction_matrix = np.zeros((n_series, n_series))
        prediction_mse_matrix = np.full((n_series, n_series), float('inf'))
        
        for i, model in enumerate(models):
            for j, series_name in enumerate(series_names):
                series_values = interval_data[series_name].fillna(method='ffill').fillna(method='bfill').values
                
                if len(series_values) > self.lag:
                    # Test if model i can predict series j
                    mse = self.test_model_on_series(model, series_values)
                    prediction_mse_matrix[i, j] = mse
                    
                    # Original model's performance on its own series
                    original_mse = models_and_performance[series_names[i]][1]
                    
                    # Models are considered equivalent if cross-prediction MSE is close to original
                    # Use a tolerance based on the original model's performance
                    tolerance_factor = 1.5  # Allow 50% increase in MSE
                    if mse <= original_mse * tolerance_factor:
                        cross_prediction_matrix[i, j] = 1
        
        # Find clusters based on mutual predictive equivalence
        clusters = []
        assigned = set()
        
        for i in range(n_series):
            if i in assigned:
                continue
                
            # Start a new cluster with series i
            current_cluster = {i}
            assigned.add(i)
            
            # Find all series that can be predicted by model i and vice versa
            for j in range(i + 1, n_series):
                if j in assigned:
                    continue
                    
                # Check mutual predictive equivalence
                if (cross_prediction_matrix[i, j] == 1 and 
                    cross_prediction_matrix[j, i] == 1):
                    current_cluster.add(j)
                    assigned.add(j)
            
            clusters.append(current_cluster)
        
        # Optimize clusters using the loss function from Equation 3.3
        optimized_clusters = self.optimize_clusters_with_loss_function(
            clusters, models_and_performance, series_names, prediction_mse_matrix
        )
        
        # Create cluster representatives and assignments
        clustered_models = {}
        series_to_cluster = {}
        
        for cluster_id, cluster_indices in enumerate(optimized_clusters):
            if cluster_indices:
                # Select the representative model: the one that best predicts all series in the cluster
                best_rep_idx = self.select_representative_model(cluster_indices, prediction_mse_matrix)
                
                if best_rep_idx is not None:
                    clustered_models[cluster_id] = models[best_rep_idx]
                    clustered_models[cluster_id].model_id = cluster_id
                    
                    # Assign all series in cluster to this cluster ID
                    for idx in cluster_indices:
                        series_to_cluster[series_names[idx]] = cluster_id
        
        return clustered_models, series_to_cluster
    
    def select_representative_model(self, cluster_indices: set, prediction_mse_matrix: np.ndarray) -> int:
        """
        Select the representative model from a cluster.
        The representative is the model that best predicts all series in the cluster.
        
        Args:
            cluster_indices: Set of indices of models in the cluster
            prediction_mse_matrix: Matrix of MSE values for cross-predictions
            
        Returns:
            best_rep_idx: Index of the best representative model
        """
        best_rep_idx = None
        best_total_mse = float('inf')
        
        cluster_list = list(cluster_indices)
        
        for candidate_idx in cluster_list:
            # Calculate total MSE when this model predicts all series in the cluster
            total_mse = sum(prediction_mse_matrix[candidate_idx, series_idx] 
                           for series_idx in cluster_list)
            
            # Also consider average MSE for better comparison
            avg_mse = total_mse / len(cluster_list)
            
            if total_mse < best_total_mse:
                best_total_mse = total_mse
                best_rep_idx = candidate_idx
        
        return best_rep_idx
    
    def test_model_on_series(self, model: AutoRegressiveModel, series_values: np.ndarray) -> float:
        """
        Test how well a model can predict a given time series
        
        Args:
            model: AR model to test
            series_values: Time series values to predict
            
        Returns:
            mse: Mean squared error of predictions
        """
        if not model.fitted or len(series_values) <= self.lag:
            return float('inf')
        
        predictions = []
        actuals = []
        
        for i in range(self.lag, len(series_values)):
            history = series_values[i-self.lag:i]
            prediction = model.predict(history)
            actual = series_values[i]
            
            predictions.append(prediction)
            actuals.append(actual)
        
        if len(predictions) == 0:
            return float('inf')
        
        return mean_squared_error(actuals, predictions)
    
    def optimize_clusters_with_loss_function(self, initial_clusters: List[set], 
                                           models_and_performance: Dict[str, Tuple[AutoRegressiveModel, float]],
                                           series_names: List[str],
                                           prediction_mse_matrix: np.ndarray) -> List[set]:
        """
        Optimize clusters using the loss function from Equation 3.3
        L(Γ) = Σ_κ Σ_l∈C_κ MSE(S_l, f(T|Θ_κ, S_l)) + τ Σ_κ Var(C_κ)
        
        Args:
            initial_clusters: Initial clustering based on predictive equivalence
            models_and_performance: Model performance data
            series_names: List of series names
            prediction_mse_matrix: Cross-prediction MSE matrix
            
        Returns:
            optimized_clusters: Optimized clusters
        """
        def calculate_cluster_loss(clusters):
            total_loss = 0.0
            
            for cluster_indices in clusters:
                if not cluster_indices:
                    continue
                    
                cluster_list = list(cluster_indices)
                
                # Find best representative model for this cluster
                best_rep_idx = None
                best_rep_loss = float('inf')
                
                for rep_idx in cluster_list:
                    # MSE component: how well does this representative predict all series in cluster
                    mse_sum = sum(prediction_mse_matrix[rep_idx, j] for j in cluster_list)
                    
                    if mse_sum < best_rep_loss:
                        best_rep_loss = mse_sum
                        best_rep_idx = rep_idx
                
                if best_rep_idx is not None:
                    # MSE component
                    cluster_mses = [prediction_mse_matrix[best_rep_idx, j] for j in cluster_list]
                    total_loss += sum(cluster_mses)
                    
                    # Variance component (regularization)
                    if len(cluster_mses) > 1:
                        cluster_variance = np.var(cluster_mses)
                        total_loss += self.tau * cluster_variance
            
            return total_loss
        
        # Try different clustering configurations to minimize loss
        best_clusters = initial_clusters
        best_loss = calculate_cluster_loss(initial_clusters)
        
        # Simple optimization: try merging clusters if it reduces loss
        improved = True
        while improved:
            improved = False
            current_clusters = [cluster.copy() for cluster in best_clusters]
            
            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    if not current_clusters[i] or not current_clusters[j]:
                        continue
                        
                    # Try merging clusters i and j
                    test_clusters = [cluster.copy() for cluster in current_clusters]
                    test_clusters[i] = current_clusters[i].union(current_clusters[j])
                    test_clusters[j] = set()  # Empty cluster
                    
                    # Remove empty clusters
                    test_clusters = [cluster for cluster in test_clusters if cluster]
                    
                    test_loss = calculate_cluster_loss(test_clusters)
                    
                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_clusters = test_clusters
                        improved = True
                        break
                
                if improved:
                    break
        
        return [cluster for cluster in best_clusters if cluster]
    
    def calculate_clustering_loss(self, clusters: List[set], models_and_performance: Dict[str, Tuple[AutoRegressiveModel, float]], 
                                 series_names: List[str], prediction_mse_matrix: np.ndarray) -> float:
        """
        Calculate clustering loss according to Equation 3.3 in the paper
        L(Γ) = Σ_κ Σ_l∈C_κ MSE(S_l, f(T|Θ_κ, S_l)) + τ Σ_κ Var(C_κ)
        
        Args:
            clusters: List of clusters (sets of indices)
            models_and_performance: Dict mapping series names to (model, mse) tuples
            series_names: List of series names
            prediction_mse_matrix: Cross-prediction MSE matrix
            
        Returns:
            total_loss: Total clustering loss
        """
        total_loss = 0.0
        
        for cluster_indices in clusters:
            if not cluster_indices:
                continue
                
            cluster_list = list(cluster_indices)
            
            # Find best representative model for this cluster
            best_rep_idx = None
            best_rep_loss = float('inf')
            
            for rep_idx in cluster_list:
                # Calculate total MSE when this model represents the cluster
                mse_sum = sum(prediction_mse_matrix[rep_idx, j] for j in cluster_list)
                if mse_sum < best_rep_loss:
                    best_rep_loss = mse_sum
                    best_rep_idx = rep_idx
            
            if best_rep_idx is not None:
                # MSE component: sum of MSEs for all series in cluster using representative model
                cluster_mses = [prediction_mse_matrix[best_rep_idx, j] for j in cluster_list]
                total_loss += sum(cluster_mses)
                
                # Variance component (regularization term)
                if len(cluster_mses) > 1:
                    cluster_variance = np.var(cluster_mses)
                    total_loss += self.tau * cluster_variance
        
        return total_loss
    
    def process_time_interval(self, df: pd.DataFrame, interval: Tuple[int, int], 
                             interval_idx: int) -> Dict[str, any]:
        """
        Process a single time interval: fit models (clustering is done globally later)
        
        Args:
            df: DataFrame with time series data
            interval: (start, end) tuple defining the interval
            interval_idx: Index of the interval
            
        Returns:
            interval_results: Dictionary containing results for this interval
        """
        # Only fit AR models - no clustering at interval level
        models_and_performance = self.fit_ar_models_for_interval(df, interval)
        
        if not models_and_performance:
            return {}
        
        # Store models and performance (clustering will be done globally)
        self.models_per_interval[interval_idx] = {name: model for name, (model, mse) in models_and_performance.items()}
        self.model_performance[interval_idx] = {name: mse for name, (model, mse) in models_and_performance.items()}
        
        return {
            'interval': interval,
            'n_series': len(models_and_performance),
            'n_models_fitted': len(models_and_performance),
            'performance': {name: mse for name, (model, mse) in models_and_performance.items()}
        }
    
    def fit(self, df: pd.DataFrame, interval_length: int = None) -> Dict[str, any]:
        """
        Main method to perform Step 1 of CAARL: Model identification across all time intervals
        
        Args:
            df: DataFrame with time series data (columns = series, rows = timesteps)
            interval_length: Length of each time interval (auto-selected if None)
            
        Returns:
            results: Dictionary containing complete results
        """
        print("="*60)
        print("🚀 CAARL STEP 1: MODEL IDENTIFICATION AND CLUSTERING")
        print("="*60)
        
        # PHASE 1: PARAMETER SELECTION
        print("📊 PHASE 1: AUTOMATIC PARAMETER SELECTION")
        print("-" * 40)
        
        if self.auto_params or interval_length is None or self.lag is None or self.stride is None:
            param_info = self.auto_select_parameters(df, interval_length)
            
            # Update parameters based on recommendations
            if interval_length is None:
                interval_length = param_info['interval_size']
            if self.lag is None:
                self.lag = param_info['lag'] 
            if self.stride is None:
                self.stride = param_info['stride']
        else:
            print(f"📋 Using Manual Parameters:")
            print(f"  - Interval length: {interval_length}")
            print(f"  - Lag: {self.lag}")
            print(f"  - Stride: {self.stride}")
            print(f"  - Regularization τ: {self.tau}")
            
        print(f"\n✅ FINAL PARAMETERS:")
        print(f"  - Dataset shape: {df.shape}")
        print(f"  - Interval length: {interval_length}")
        print(f"  - Stride: {self.stride}")
        print(f"  - Lag: {self.lag}")
        print(f"  - Regularization τ: {self.tau}")
        
        # Create time intervals using finalized parameters
        intervals = self.create_time_intervals(df, interval_length)
        print(f"  - Created {len(intervals)} time intervals")
        
        # PHASE 2: MODEL FITTING PER INTERVAL
        print(f"\n🔧 PHASE 2: MODEL FITTING ACROSS ALL INTERVALS")
        print("-" * 40)
        print(f"Fitting AR({self.lag}) models for all series across {len(intervals)} intervals...")
        
        all_models_info = []  # Will store all fitted models
        
        for interval_idx, interval in enumerate(intervals):
            print(f"  📍 Interval {interval_idx + 1:2d}/{len(intervals)}: [{interval[0]:4d}:{interval[1]:4d}]", end="")
            
            # Fit AR models for all series in this interval
            models_and_performance = self.fit_ar_models_for_interval(df, interval)
            
            if models_and_performance:
                print(f" → {len(models_and_performance)} models fitted")
                
                # Store models for this interval
                self.models_per_interval[interval_idx] = {name: model for name, (model, mse) in models_and_performance.items()}
                self.model_performance[interval_idx] = {name: mse for name, (model, mse) in models_and_performance.items()}
                
                # Add to global model collection
                for series_name, (model, mse) in models_and_performance.items():
                    all_models_info.append({
                        'interval_idx': interval_idx,
                        'series_name': series_name,
                        'model': model,
                        'mse': mse,
                        'interval': interval,
                    })
            else:
                print(f" → No valid models")
        
        total_models = len(all_models_info)
        print(f"\n✅ TOTAL MODELS FITTED: {total_models} across all intervals")
        
        if total_models == 0:
            print("❌ No models were successfully fitted. Check your parameters.")
            return self._empty_results(intervals, interval_length)
        
        # PHASE 3: GLOBAL CLUSTERING
        print(f"\n🎯 PHASE 3: GLOBAL CLUSTERING OF ALL MODELS")
        print("-" * 40)
        print(f"Performing global clustering over {total_models} models...")
        
        global_clustered_models, model_to_cluster = self.perform_global_clustering(all_models_info, df)
        
        print(f"✅ DISCOVERED {len(global_clustered_models)} UNIQUE MODEL PATTERNS")
        
        # PHASE 4: ASSIGN RESULTS BACK TO INTERVALS
        print(f"\n📊 PHASE 4: ASSIGNING CLUSTER RESULTS TO INTERVALS")
        print("-" * 40)
        
        self.assign_clusters_to_intervals(all_models_info, model_to_cluster, global_clustered_models)
        
        # Prepare final results
        all_results = []
        for interval_idx in range(len(intervals)):
            if interval_idx in self.models_per_interval and self.models_per_interval[interval_idx]:
                result = {
                    'interval': intervals[interval_idx],
                    'n_series': len(self.models_per_interval[interval_idx]),
                    'n_clusters': len(self.clustered_models.get(interval_idx, {})),
                    'series_to_cluster': self.series_to_cluster.get(interval_idx, {}),
                    'cluster_models': self.clustered_models.get(interval_idx, {}),
                    'performance': self.model_performance.get(interval_idx, {})
                }
                all_results.append(result)
        
        # SUMMARY STATISTICS
        print(f"\n" + "="*60)
        print("📈 FINAL SUMMARY STATISTICS")
        print("="*60)
        
        compression_ratio = len(global_clustered_models) / total_models if total_models > 0 else 1.0
        
        print(f"📊 Dataset Analysis:")
        print(f"  - Total time points: {len(df)}")
        print(f"  - Total time series: {len(df.columns)}")
        print(f"  - Intervals created: {len(intervals)}")
        
        print(f"\n🔧 Model Fitting Results:")
        print(f"  - Total models fitted: {total_models}")
        print(f"  - Intervals with valid models: {len(all_results)}")
        if all_results:
            print(f"  - Avg models per interval: {np.mean([r['n_series'] for r in all_results]):.1f}")
        
        print(f"\n🎯 Clustering Results:")
        print(f"  - Unique model patterns: {len(global_clustered_models)}")
        print(f"  - Model compression ratio: {compression_ratio:.3f}")
        if all_results:
            print(f"  - Avg clusters per interval: {np.mean([r['n_clusters'] for r in all_results]):.1f}")
        
        # Interpretation of results
        if compression_ratio < 0.3:
            print(f"  🎯 EXCELLENT: High model reuse - strong co-evolutionary patterns detected")
        elif compression_ratio < 0.6:
            print(f"  ✅ GOOD: Moderate model reuse - some co-evolutionary patterns found")
        else:
            print(f"  ⚠️  LIMITED: Low model reuse - consider adjusting parameters")
        
        # Validation of parameter choices
        if self.parameter_recommendations:
            self.validate_parameter_performance(all_results)
        
        return {
            'intervals': all_results,
            'models_per_interval': self.models_per_interval,
            'clustered_models': self.clustered_models,
            'series_to_cluster': self.series_to_cluster,
            'model_performance': self.model_performance,
            'global_clustered_models': global_clustered_models,
            'total_intervals': len(intervals),
            'total_models_fitted': total_models,
            'total_unique_models': len(global_clustered_models),
            'compression_ratio': compression_ratio,
            'parameter_recommendations': self.parameter_recommendations,
            'final_parameters': {
                'interval_length': interval_length,
                'lag': self.lag,
                'stride': self.stride,
                'tau': self.tau
            }
        }
    
    def _empty_results(self, intervals: List, interval_length: int) -> Dict:
        """Return empty results structure when no models are fitted"""
        return {
            'intervals': [],
            'models_per_interval': {},
            'clustered_models': {},
            'series_to_cluster': {},
            'model_performance': {},
            'global_clustered_models': {},
            'total_intervals': len(intervals),
            'total_models_fitted': 0,
            'total_unique_models': 0,
            'compression_ratio': 1.0,
            'parameter_recommendations': self.parameter_recommendations,
            'final_parameters': {
                'interval_length': interval_length,
                'lag': self.lag,
                'stride': self.stride,
                'tau': self.tau
            }
        }
    
    
    def perform_global_clustering(self, all_models_info: List[Dict], df: pd.DataFrame) -> Tuple[Dict[int, AutoRegressiveModel], Dict[str, int]]:
        """
        Perform OPTIMIZED global clustering over ALL models from ALL intervals
        
        Args:
            all_models_info: List of model information dictionaries
            df: Original dataframe for cross-prediction testing
            
        Returns:
            global_clustered_models: Dict mapping global cluster IDs to representative models
            model_to_cluster: Dict mapping model identifiers to cluster IDs
        """
        if not all_models_info:
            return {}, {}
        
        n_models = len(all_models_info)
        print(f"  🚀 OPTIMIZED clustering for {n_models} models...")
        
        # OPTIMIZATION 1: Parameter-based pre-clustering to reduce comparisons
        print(f"  🔍 Step 1: Pre-clustering by parameter similarity...")
        param_clusters = self.pre_cluster_by_parameters(all_models_info)
        print(f"  ✅ Reduced {n_models} models to {len(param_clusters)} parameter-based groups")
        
        # OPTIMIZATION 2: Smart cross-prediction sampling
        print(f"  🎯 Step 2: Smart cross-prediction testing...")
        cross_prediction_info = self.build_smart_cross_prediction_matrix(
            all_models_info, param_clusters, df
        )
        
        # OPTIMIZATION 3: Hierarchical clustering within groups
        print(f"  🔗 Step 3: Hierarchical clustering within parameter groups...")
        final_clusters = self.hierarchical_clustering_within_groups(
            param_clusters, cross_prediction_info, all_models_info
        )
        
        # OPTIMIZATION 4: Final optimization
        print(f"  ⚡ Step 4: Final loss-based optimization...")
        optimized_clusters = self.fast_optimize_clusters(
            final_clusters, all_models_info, cross_prediction_info
        )
        
        # Create global cluster representatives and assignments
        global_clustered_models = {}
        model_to_cluster = {}
        
        for cluster_id, cluster_indices in enumerate(optimized_clusters):
            if cluster_indices:
                # Select representative model for this cluster
                rep_idx = self.select_representative_from_info(
                    cluster_indices, cross_prediction_info, all_models_info
                )
                
                if rep_idx is not None:
                    representative_model = all_models_info[rep_idx]['model']
                    representative_model.model_id = cluster_id
                    global_clustered_models[cluster_id] = representative_model
                    
                    # Assign all models in cluster to this cluster ID
                    for idx in cluster_indices:
                        model_key = f"{all_models_info[idx]['interval_idx']}_{all_models_info[idx]['series_name']}"
                        model_to_cluster[model_key] = cluster_id
        
        return global_clustered_models, model_to_cluster
    
    def pre_cluster_by_parameters(self, all_models_info: List[Dict], 
                                 tolerance: float = 0.3) -> List[List[int]]:
        """
        OPTIMIZATION 1: Pre-cluster models by parameter similarity to reduce comparisons
        
        Args:
            all_models_info: List of all model information
            tolerance: Tolerance for parameter similarity
            
        Returns:
            param_clusters: List of clusters (lists of model indices)
        """
        if not all_models_info:
            return []
        
        n_models = len(all_models_info)
        param_clusters = []
        assigned = set()
        
        for i in range(n_models):
            if i in assigned:
                continue
                
            # Start new cluster
            current_cluster = [i]
            assigned.add(i)
            model_i = all_models_info[i]['model']
            
            # Find similar models by parameters
            for j in range(i + 1, n_models):
                if j in assigned:
                    continue
                    
                model_j = all_models_info[j]['model']
                similarity = model_i.calculate_similarity(model_j, tolerance)
                
                if similarity > 0.7:  # High parameter similarity
                    current_cluster.append(j)
                    assigned.add(j)
            
            param_clusters.append(current_cluster)
        
        return param_clusters
    
    def build_smart_cross_prediction_matrix(self, all_models_info: List[Dict], 
                                           param_clusters: List[List[int]], 
                                           df: pd.DataFrame) -> Dict:
        """
        OPTIMIZATION 2: Build cross-prediction matrix smartly with sampling
        
        Args:
            all_models_info: List of model information
            param_clusters: Parameter-based pre-clusters
            df: Original dataframe
            
        Returns:
            cross_prediction_info: Dict with prediction information
        """
        cross_prediction_info = {
            'within_cluster_mse': {},  # MSE within parameter clusters
            'cross_cluster_mse': {},   # MSE between parameter clusters
            'model_own_mse': {}        # Each model's performance on its own data
        }
        
        # Store each model's own performance
        for i, model_info in enumerate(all_models_info):
            cross_prediction_info['model_own_mse'][i] = model_info['mse']
        
        # Test within parameter clusters (full testing)
        for cluster_idx, cluster in enumerate(param_clusters):
            if len(cluster) <= 1:
                continue
                
            for i in cluster:
                for j in cluster:
                    if i != j:
                        key = (i, j)
                        if key not in cross_prediction_info['within_cluster_mse']:
                            model_i = all_models_info[i]['model']
                            series_name_j = all_models_info[j]['series_name']
                            interval_j = all_models_info[j]['interval']
                            
                            # Get data that model j was trained on
                            series_data_j = df[series_name_j].iloc[interval_j[0]:interval_j[1]].fillna(method='ffill').fillna(method='bfill').values
                            
                            if len(series_data_j) > self.lag:
                                mse = self.test_model_on_series(model_i, series_data_j)
                                cross_prediction_info['within_cluster_mse'][key] = mse
        
        # Test between parameter clusters (sampling approach)
        for i, cluster_i in enumerate(param_clusters):
            for j, cluster_j in enumerate(param_clusters):
                if i >= j:  # Only test upper triangle
                    continue
                    
                # Sample representatives from each cluster
                rep_i = cluster_i[0]  # Use first model as representative
                rep_j = cluster_j[0]  # Use first model as representative
                
                # Test both directions
                for model_idx, data_idx in [(rep_i, rep_j), (rep_j, rep_i)]:
                    key = (model_idx, data_idx)
                    if key not in cross_prediction_info['cross_cluster_mse']:
                        model = all_models_info[model_idx]['model']
                        series_name = all_models_info[data_idx]['series_name']
                        interval = all_models_info[data_idx]['interval']
                        
                        series_data = df[series_name].iloc[interval[0]:interval[1]].fillna(method='ffill').fillna(method='bfill').values
                        
                        if len(series_data) > self.lag:
                            mse = self.test_model_on_series(model, series_data)
                            cross_prediction_info['cross_cluster_mse'][key] = mse
        
        return cross_prediction_info
    
    def hierarchical_clustering_within_groups(self, param_clusters: List[List[int]], 
                                            cross_prediction_info: Dict,
                                            all_models_info: List[Dict]) -> List[set]:
        """
        OPTIMIZATION 3: Hierarchical clustering within parameter groups
        
        Args:
            param_clusters: Parameter-based clusters
            cross_prediction_info: Cross-prediction information
            all_models_info: Model information
            
        Returns:
            final_clusters: Refined clusters based on predictive equivalence
        """
        final_clusters = []
        
        for param_cluster in param_clusters:
            if len(param_cluster) == 1:
                # Single model cluster
                final_clusters.append(set(param_cluster))
                continue
            
            # Find predictive equivalence within this parameter cluster
            equivalence_matrix = np.zeros((len(param_cluster), len(param_cluster)), dtype=bool)
            
            for i, model_idx_i in enumerate(param_cluster):
                for j, model_idx_j in enumerate(param_cluster):
                    if i == j:
                        equivalence_matrix[i, j] = True
                    else:
                        # Check mutual predictive equivalence
                        key_ij = (model_idx_i, model_idx_j)
                        key_ji = (model_idx_j, model_idx_i)
                        
                        mse_ij = cross_prediction_info['within_cluster_mse'].get(key_ij, float('inf'))
                        mse_ji = cross_prediction_info['within_cluster_mse'].get(key_ji, float('inf'))
                        
                        # Get thresholds
                        threshold_i = cross_prediction_info['model_own_mse'][model_idx_i] * 2.0
                        threshold_j = cross_prediction_info['model_own_mse'][model_idx_j] * 2.0
                        
                        # Models are equivalent if they can predict each other's data well
                        if mse_ij <= threshold_j and mse_ji <= threshold_i:
                            equivalence_matrix[i, j] = True
            
            # Find connected components in equivalence matrix
            visited = set()
            for i in range(len(param_cluster)):
                if i not in visited:
                    cluster = set()
                    stack = [i]
                    
                    while stack:
                        current = stack.pop()
                        if current not in visited:
                            visited.add(current)
                            cluster.add(param_cluster[current])  # Convert back to original indices
                            
                            for j in range(len(param_cluster)):
                                if equivalence_matrix[current, j] and j not in visited:
                                    stack.append(j)
                    
                    final_clusters.append(cluster)
        
        return final_clusters
    
    def fast_optimize_clusters(self, initial_clusters: List[set], 
                              all_models_info: List[Dict],
                              cross_prediction_info: Dict) -> List[set]:
        """
        OPTIMIZATION 4: Fast cluster optimization using cached predictions
        
        Args:
            initial_clusters: Initial clusters
            all_models_info: Model information
            cross_prediction_info: Cached prediction information
            
        Returns:
            optimized_clusters: Optimized clusters
        """
        def calculate_fast_cluster_loss(clusters):
            total_loss = 0.0
            
            for cluster_indices in clusters:
                if len(cluster_indices) <= 1:
                    continue
                    
                cluster_list = list(cluster_indices)
                
                # Find best representative
                best_rep_idx = None
                best_rep_loss = float('inf')
                
                for rep_idx in cluster_list:
                    rep_loss = 0.0
                    
                    for model_idx in cluster_list:
                        if rep_idx == model_idx:
                            # Representative predicting its own data
                            mse = cross_prediction_info['model_own_mse'][rep_idx]
                        else:
                            # Check cached predictions
                            key = (rep_idx, model_idx)
                            if key in cross_prediction_info['within_cluster_mse']:
                                mse = cross_prediction_info['within_cluster_mse'][key]
                            elif key in cross_prediction_info['cross_cluster_mse']:
                                mse = cross_prediction_info['cross_cluster_mse'][key]
                            else:
                                # Estimate based on parameter similarity
                                model_rep = all_models_info[rep_idx]['model']
                                model_target = all_models_info[model_idx]['model']
                                similarity = model_rep.calculate_similarity(model_target)
                                base_mse = cross_prediction_info['model_own_mse'][model_idx]
                                mse = base_mse * (2.0 - similarity)  # Higher similarity = lower MSE
                        
                        rep_loss += mse
                    
                    if rep_loss < best_rep_loss:
                        best_rep_loss = rep_loss
                        best_rep_idx = rep_idx
                
                if best_rep_idx is not None:
                    # Calculate cluster MSEs for variance
                    cluster_mses = []
                    for model_idx in cluster_list:
                        if best_rep_idx == model_idx:
                            mse = cross_prediction_info['model_own_mse'][best_rep_idx]
                        else:
                            key = (best_rep_idx, model_idx)
                            if key in cross_prediction_info['within_cluster_mse']:
                                mse = cross_prediction_info['within_cluster_mse'][key]
                            elif key in cross_prediction_info['cross_cluster_mse']:
                                mse = cross_prediction_info['cross_cluster_mse'][key]
                            else:
                                # Estimate
                                model_rep = all_models_info[best_rep_idx]['model']
                                model_target = all_models_info[model_idx]['model']
                                similarity = model_rep.calculate_similarity(model_target)
                                base_mse = cross_prediction_info['model_own_mse'][model_idx]
                                mse = base_mse * (2.0 - similarity)
                        
                        cluster_mses.append(mse)
                        total_loss += mse
                    
                    # Add variance term
                    if len(cluster_mses) > 1:
                        cluster_variance = np.var(cluster_mses)
                        total_loss += self.tau * cluster_variance
            
            return total_loss
        
        # Fast optimization with limited iterations
        best_clusters = initial_clusters
        best_loss = calculate_fast_cluster_loss(initial_clusters)
        
        # Try merging small clusters only (efficiency focus)
        for i in range(len(initial_clusters)):
            for j in range(i + 1, len(initial_clusters)):
                if len(initial_clusters[i]) <= 3 and len(initial_clusters[j]) <= 3:
                    # Try merging small clusters
                    test_clusters = [cluster.copy() for cluster in initial_clusters]
                    test_clusters[i] = initial_clusters[i].union(initial_clusters[j])
                    test_clusters[j] = set()  # Empty
                    test_clusters = [cluster for cluster in test_clusters if cluster]
                    
                    test_loss = calculate_fast_cluster_loss(test_clusters)
                    
                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_clusters = test_clusters
        
        return [cluster for cluster in best_clusters if cluster]
    
    def select_representative_from_info(self, cluster_indices: set, 
                                       cross_prediction_info: Dict,
                                       all_models_info: List[Dict]) -> int:
        """
        Select representative model using cached prediction info
        
        Args:
            cluster_indices: Set of model indices in cluster
            cross_prediction_info: Cached prediction information
            all_models_info: Model information
            
        Returns:
            best_rep_idx: Index of best representative model
        """
        best_rep_idx = None
        best_total_loss = float('inf')
        
        cluster_list = list(cluster_indices)
        
        for candidate_idx in cluster_list:
            total_loss = 0.0
            
            for model_idx in cluster_list:
                if candidate_idx == model_idx:
                    mse = cross_prediction_info['model_own_mse'][candidate_idx]
                else:
                    key = (candidate_idx, model_idx)
                    if key in cross_prediction_info['within_cluster_mse']:
                        mse = cross_prediction_info['within_cluster_mse'][key]
                    elif key in cross_prediction_info['cross_cluster_mse']:
                        mse = cross_prediction_info['cross_cluster_mse'][key]
                    else:
                        # Estimate based on similarity
                        model_cand = all_models_info[candidate_idx]['model']
                        model_target = all_models_info[model_idx]['model']
                        similarity = model_cand.calculate_similarity(model_target)
                        base_mse = cross_prediction_info['model_own_mse'][model_idx]
                        mse = base_mse * (2.0 - similarity)
                
                total_loss += mse
            
            if total_loss < best_total_loss:
                best_total_loss = total_loss
                best_rep_idx = candidate_idx
        
        return best_rep_idx
    
    def find_predictive_equivalence_clusters(self, cross_prediction_matrix: np.ndarray, 
                                           all_models_info: List[Dict]) -> List[set]:
        """
        Find initial clusters based on predictive equivalence
        Two models belong to the same cluster if they can successfully predict the same datasets
        
        Args:
            cross_prediction_matrix: Matrix of cross-prediction MSEs
            all_models_info: List of all model information
            
        Returns:
            clusters: List of clusters (sets of model indices)
        """
        n_models = len(all_models_info)
        
        # Determine prediction success threshold for each model
        prediction_thresholds = []
        for i in range(n_models):
            original_mse = cross_prediction_matrix[i, i]  # Model's performance on its own data
            threshold = original_mse * 2.0  # Allow 2x degradation
            prediction_thresholds.append(threshold)
        
        # Build equivalence matrix: models that can predict similar datasets well
        equivalence_matrix = np.zeros((n_models, n_models), dtype=bool)
        
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    # Check if both models can predict each other's datasets reasonably well
                    can_i_predict_j_data = cross_prediction_matrix[i, j] <= prediction_thresholds[j]
                    can_j_predict_i_data = cross_prediction_matrix[j, i] <= prediction_thresholds[i]
                    
                    # Models are equivalent if they can both handle each other's datasets
                    if can_i_predict_j_data and can_j_predict_i_data:
                        equivalence_matrix[i, j] = True
                else:
                    equivalence_matrix[i, j] = True  # Model is equivalent to itself
        
        # Find connected components (clusters) in the equivalence graph
        clusters = []
        visited = set()
        
        for i in range(n_models):
            if i not in visited:
                # Start new cluster
                cluster = set()
                stack = [i]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster.add(current)
                        
                        # Add all equivalent models to the stack
                        for j in range(n_models):
                            if equivalence_matrix[current, j] and j not in visited:
                                stack.append(j)
                
                clusters.append(cluster)
        
        return clusters
    
    def optimize_global_clusters_with_loss_function(self, initial_clusters: List[set], 
                                                   all_models_info: List[Dict],
                                                   cross_prediction_matrix: np.ndarray) -> List[set]:
        """
        Optimize clusters using the loss function from Equation 3.3 globally
        
        Args:
            initial_clusters: Initial clustering based on predictive equivalence
            all_models_info: List of all model information
            cross_prediction_matrix: Cross-prediction MSE matrix
            
        Returns:
            optimized_clusters: Optimized clusters
        """
        def calculate_global_cluster_loss(clusters):
            total_loss = 0.0
            
            for cluster_indices in clusters:
                if not cluster_indices:
                    continue
                    
                cluster_list = list(cluster_indices)
                
                # Find best representative model for this cluster
                best_rep_idx = self.select_global_representative_model(cluster_indices, cross_prediction_matrix)
                
                if best_rep_idx is not None:
                    # MSE component: sum of MSEs when representative predicts all datasets in cluster
                    cluster_mses = []
                    for model_idx in cluster_list:
                        mse = cross_prediction_matrix[best_rep_idx, model_idx]
                        cluster_mses.append(mse)
                        total_loss += mse
                    
                    # Variance component (regularization)
                    if len(cluster_mses) > 1:
                        cluster_variance = np.var(cluster_mses)
                        total_loss += self.tau * cluster_variance
            
            return total_loss
        
        # Try different clustering configurations to minimize loss
        best_clusters = initial_clusters
        best_loss = calculate_global_cluster_loss(initial_clusters)
        
        # Simple optimization: try merging clusters if it reduces loss
        improved = True
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            current_clusters = [cluster.copy() for cluster in best_clusters]
            
            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    if not current_clusters[i] or not current_clusters[j]:
                        continue
                        
                    # Try merging clusters i and j
                    test_clusters = [cluster.copy() for cluster in current_clusters]
                    test_clusters[i] = current_clusters[i].union(current_clusters[j])
                    test_clusters[j] = set()  # Empty cluster
                    
                    # Remove empty clusters
                    test_clusters = [cluster for cluster in test_clusters if cluster]
                    
                    test_loss = calculate_global_cluster_loss(test_clusters)
                    
                    if test_loss < best_loss:
                        best_loss = test_loss
                        best_clusters = test_clusters
                        improved = True
                        print(f"    Merged clusters: loss improved to {best_loss:.4f}")
                        break
                
                if improved:
                    break
            
            iteration += 1
        
        return [cluster for cluster in best_clusters if cluster]
    
    def select_global_representative_model(self, cluster_indices: set, 
                                         cross_prediction_matrix: np.ndarray) -> int:
        """
        Select the representative model from a global cluster
        
        Args:
            cluster_indices: Set of model indices in the cluster
            cross_prediction_matrix: Matrix of cross-prediction MSEs
            
        Returns:
            best_rep_idx: Index of the best representative model
        """
        best_rep_idx = None
        best_total_mse = float('inf')
        
        cluster_list = list(cluster_indices)
        
        for candidate_idx in cluster_list:
            # Calculate total MSE when this model predicts all datasets in the cluster
            total_mse = sum(cross_prediction_matrix[candidate_idx, dataset_idx] 
                           for dataset_idx in cluster_list)
            
            if total_mse < best_total_mse:
                best_total_mse = total_mse
                best_rep_idx = candidate_idx
        
        return best_rep_idx
    
    def assign_clusters_to_intervals(self, all_models_info: List[Dict], 
                                   model_to_cluster: Dict[str, int],
                                   global_clustered_models: Dict[int, AutoRegressiveModel]):
        """
        Assign the global cluster results back to individual intervals
        
        Args:
            all_models_info: List of all model information
            model_to_cluster: Mapping from model keys to cluster IDs
            global_clustered_models: Global cluster representatives
        """
        # Initialize interval-specific storage
        for model_info in all_models_info:
            interval_idx = model_info['interval_idx']
            series_name = model_info['series_name']
            model_key = f"{interval_idx}_{series_name}"
            
            if model_key in model_to_cluster:
                cluster_id = model_to_cluster[model_key]
                
                # Initialize interval dictionaries if needed
                if interval_idx not in self.clustered_models:
                    self.clustered_models[interval_idx] = {}
                if interval_idx not in self.series_to_cluster:
                    self.series_to_cluster[interval_idx] = {}
                
                # Assign cluster representative to this interval (if not already present)
                if cluster_id not in self.clustered_models[interval_idx]:
                    self.clustered_models[interval_idx][cluster_id] = global_clustered_models[cluster_id]
                
                # Assign series to cluster
                self.series_to_cluster[interval_idx][series_name] = cluster_id
    
    def validate_parameter_performance(self, all_results: List[Dict]):
        """
        Validate how well the selected parameters performed
        
        Args:
            all_results: Results from processing all intervals
        """
        print(f"\n🔍 PARAMETER PERFORMANCE VALIDATION")
        print("="*60)
        
        if not all_results:
            print("❌ No results to validate")
            return
            
        # Calculate performance metrics
        avg_clusters_per_interval = np.mean([r['n_clusters'] for r in all_results])
        avg_series_per_interval = np.mean([r['n_series'] for r in all_results])
        
        # Calculate clustering efficiency (fewer clusters = better compression)
        avg_compression_ratio = avg_clusters_per_interval / avg_series_per_interval if avg_series_per_interval > 0 else 1
        
        # Calculate model performance statistics
        all_mse_values = []
        for result in all_results:
            all_mse_values.extend(result['performance'].values())
        
        if all_mse_values:
            avg_mse = np.mean(all_mse_values)
            std_mse = np.std(all_mse_values)
            
            print(f"✅ Clustering Performance:")
            print(f"  - Average clusters per interval: {avg_clusters_per_interval:.2f}")
            print(f"  - Compression ratio: {avg_compression_ratio:.3f} (lower is better)")
            print(f"  - Model fit quality (MSE): {avg_mse:.4f} ± {std_mse:.4f}")
            
            # Assess parameter quality
            if avg_compression_ratio < 0.3:
                print(f"  🎯 Excellent clustering - high model reuse")
            elif avg_compression_ratio < 0.6:
                print(f"  ✅ Good clustering - moderate model reuse")
            else:
                print(f"  ⚠️  Poor clustering - consider adjusting parameters")
                
            if avg_mse < 0.1:
                print(f"  🎯 Excellent model fit quality")
            elif avg_mse < 0.5:
                print(f"  ✅ Good model fit quality")
            else:
                print(f"  ⚠️  Poor model fit - consider increasing interval size or adjusting lag")
    
    def visualize_results(self, df: pd.DataFrame, results: Dict[str, any], save_plots: bool = False):
        """
        Visualize the results of Step 1
        
        Args:
            df: Original DataFrame
            results: Results from fit method
            save_plots: Whether to save plots to files
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CAARL Step 1: Model Identification Results', fontsize=16)
        
        # Plot 1: Number of clusters per interval
        intervals = [i for i in range(len(results['intervals']))]
        n_clusters = [r['n_clusters'] for r in results['intervals']]
        
        axes[0, 0].plot(intervals, n_clusters, 'bo-', linewidth=2, markersize=6)
        axes[0, 0].set_title('Number of Model Clusters per Interval')
        axes[0, 0].set_xlabel('Interval Index')
        axes[0, 0].set_ylabel('Number of Clusters')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Distribution of cluster sizes
        all_cluster_sizes = []
        for result in results['intervals']:
            for cluster_id in result['series_to_cluster'].values():
                cluster_size = list(result['series_to_cluster'].values()).count(cluster_id)
                all_cluster_sizes.append(cluster_size)
        
        axes[0, 1].hist(all_cluster_sizes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('Distribution of Cluster Sizes')
        axes[0, 1].set_xlabel('Cluster Size (Number of Series)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Model performance over time
        if self.model_performance:
            interval_avg_mse = []
            for interval_idx in sorted(self.model_performance.keys()):
                mse_values = list(self.model_performance[interval_idx].values())
                interval_avg_mse.append(np.mean(mse_values))
            
            axes[1, 0].plot(sorted(self.model_performance.keys()), interval_avg_mse, 'ro-', linewidth=2)
            axes[1, 0].set_title('Average Model Performance per Interval')
            axes[1, 0].set_xlabel('Interval Index')
            axes[1, 0].set_ylabel('Average MSE')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Series clustering heatmap (first few intervals)
        if len(results['intervals']) > 0:
            # Create a matrix showing which series belong to which clusters
            series_names = df.columns[:min(10, len(df.columns))]  # Limit to first 10 series for visibility
            intervals_to_show = min(5, len(results['intervals']))  # Show first 5 intervals
            
            cluster_matrix = np.zeros((len(series_names), intervals_to_show))
            
            for i, result in enumerate(results['intervals'][:intervals_to_show]):
                for j, series_name in enumerate(series_names):
                    if series_name in result['series_to_cluster']:
                        cluster_matrix[j, i] = result['series_to_cluster'][series_name]
                    else:
                        cluster_matrix[j, i] = -1  # Series not present
            
            im = axes[1, 1].imshow(cluster_matrix, cmap='tab10', aspect='auto')
            axes[1, 1].set_title('Series Clustering Across Intervals')
            axes[1, 1].set_xlabel('Interval Index')
            axes[1, 1].set_ylabel('Time Series')
            axes[1, 1].set_yticks(range(len(series_names)))
            axes[1, 1].set_yticklabels([name[:10] for name in series_names])
            plt.colorbar(im, ax=axes[1, 1], label='Cluster ID')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('caarl_step1_results.png', dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get a summary of all discovered models
        
        Returns:
            summary_df: DataFrame with model summary information
        """
        summary_data = []
        
        for interval_idx, cluster_models in self.clustered_models.items():
            for cluster_id, model in cluster_models.items():
                if model.fitted:
                    # Count how many series use this model
                    series_count = list(self.series_to_cluster[interval_idx].values()).count(cluster_id)
                    
                    summary_data.append({
                        'Interval': interval_idx,
                        'Cluster_ID': cluster_id,
                        'Model_ID': f"Θ{cluster_id}",
                        'Coefficients': model.coefficients.tolist(),
                        'Intercept': model.intercept,
                        'Series_Count': series_count,
                        'Avg_Performance': np.mean([mse for name, mse in self.model_performance[interval_idx].items() 
                                                   if self.series_to_cluster[interval_idx].get(name) == cluster_id])
                    })
        
        return pd.DataFrame(summary_data)




dataname = 'rock_data' #'SyD-50', 'ETTh1', 'FXs_interpolated', passengers, rock_data
path_to_data = '~/Documents/Projets/Causal-Inference-Graph-Modeling-in-CoEvolving-Time-Sequences 2/dataset/'+dataname+'.csv'
df = DataHandler(path_to_data, size=None, stride=None)._load_data(path_to_data, normalize=True)

# Example usage and testing
if __name__ == "__main__":
    
    
    print("🚀 CAARL Step 1 - Complete Implementation with Auto Parameter Selection")
    print("="*80)
    
    # Test 1: Full automatic parameter selection
    print("\n📋 TEST 1: Full Automatic Parameter Selection")
    print("-" * 50)
    caarl_auto = CAARLStep1(auto_params=True, change_detection_priority="medium")
    results_auto = caarl_auto.fit(df)
   
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
    

    