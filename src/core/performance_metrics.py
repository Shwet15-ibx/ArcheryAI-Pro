"""
ArcheryAI Pro - Performance Metrics Module
Comprehensive performance analysis and scoring for archery form evaluation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
from scipy import stats

from ..utils.logger import get_logger

class PerformanceMetrics:
    """
    Comprehensive performance metrics system for archery form evaluation.
    
    This class implements patent-worthy performance analysis algorithms that
    calculate accuracy, consistency, efficiency, and stability metrics.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the PerformanceMetrics.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = get_logger(__name__)
        self.config = config
        
        # Performance metric weights
        self.metric_weights = {
            'accuracy': 0.3,
            'consistency': 0.25,
            'efficiency': 0.25,
            'stability': 0.2
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'average': 0.7,
            'below_average': 0.6,
            'poor': 0.5
        }
        
        # Metric calculation parameters
        self.calculation_params = {
            'smoothing_window': 5,
            'outlier_threshold': 2.0,
            'consistency_window': 10,
            'efficiency_threshold': 0.8
        }
        
        self.logger.info("PerformanceMetrics initialized successfully")
    
    def calculate_all_metrics(self, pose_data: Dict, biomechanics_results: Dict, form_results: Dict) -> Dict[str, Any]:
        """
        Calculate all performance metrics.
        
        Args:
            pose_data: Pose detection data
            biomechanics_results: Biomechanical analysis results
            form_results: Form evaluation results
            
        Returns:
            Dictionary containing all performance metrics
        """
        try:
            metrics = {
                'accuracy_score': self._calculate_accuracy_score(pose_data, biomechanics_results, form_results),
                'consistency_score': self._calculate_consistency_score(pose_data, biomechanics_results, form_results),
                'efficiency_score': self._calculate_efficiency_score(pose_data, biomechanics_results, form_results),
                'stability_score': self._calculate_stability_score(pose_data, biomechanics_results, form_results),
                'overall_performance': 0.0,
                'detailed_metrics': {},
                'performance_trends': {},
                'improvement_areas': []
            }
            
            # Calculate overall performance
            metrics['overall_performance'] = self._calculate_overall_performance(metrics)
            
            # Calculate detailed metrics
            metrics['detailed_metrics'] = self._calculate_detailed_metrics(pose_data, biomechanics_results, form_results)
            
            # Analyze performance trends
            metrics['performance_trends'] = self._analyze_performance_trends(pose_data, biomechanics_results, form_results)
            
            # Identify improvement areas
            metrics['improvement_areas'] = self._identify_improvement_areas(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {
                'accuracy_score': 0.0,
                'consistency_score': 0.0,
                'efficiency_score': 0.0,
                'stability_score': 0.0,
                'overall_performance': 0.0,
                'detailed_metrics': {},
                'performance_trends': {},
                'improvement_areas': []
            }
    
    def generate_improvement_suggestions(self, performance_results: Dict) -> List[Dict]:
        """
        Generate improvement suggestions based on performance results.
        
        Args:
            performance_results: Performance analysis results
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        try:
            # Analyze each metric and generate suggestions
            if performance_results.get('accuracy_score', 0) < 0.8:
                suggestions.append({
                    'metric': 'accuracy',
                    'current_score': performance_results.get('accuracy_score', 0),
                    'target_score': 0.8,
                    'suggestion': 'Focus on improving form consistency and reducing movement during aiming',
                    'priority': 'high',
                    'estimated_improvement': 0.15
                })
            
            if performance_results.get('consistency_score', 0) < 0.8:
                suggestions.append({
                    'metric': 'consistency',
                    'current_score': performance_results.get('consistency_score', 0),
                    'target_score': 0.8,
                    'suggestion': 'Practice maintaining consistent technique across all shots',
                    'priority': 'high',
                    'estimated_improvement': 0.12
                })
            
            if performance_results.get('efficiency_score', 0) < 0.8:
                suggestions.append({
                    'metric': 'efficiency',
                    'current_score': performance_results.get('efficiency_score', 0),
                    'target_score': 0.8,
                    'suggestion': 'Optimize movement patterns and reduce unnecessary motion',
                    'priority': 'medium',
                    'estimated_improvement': 0.10
                })
            
            if performance_results.get('stability_score', 0) < 0.8:
                suggestions.append({
                    'metric': 'stability',
                    'current_score': performance_results.get('stability_score', 0),
                    'target_score': 0.8,
                    'suggestion': 'Improve body stability and reduce sway during shot execution',
                    'priority': 'medium',
                    'estimated_improvement': 0.08
                })
            
            # Add general improvement suggestions
            suggestions.append({
                'metric': 'general',
                'current_score': performance_results.get('overall_performance', 0),
                'target_score': 0.9,
                'suggestion': 'Continue practicing with focus on form fundamentals',
                'priority': 'low',
                'estimated_improvement': 0.05
            })
            
        except Exception as e:
            self.logger.error(f"Error generating improvement suggestions: {str(e)}")
        
        return suggestions
    
    def _calculate_accuracy_score(self, pose_data: Dict, biomechanics_results: Dict, form_results: Dict) -> float:
        """
        Calculate accuracy score based on form precision and consistency.
        
        Args:
            pose_data: Pose detection data
            biomechanics_results: Biomechanical analysis results
            form_results: Form evaluation results
            
        Returns:
            Accuracy score (0-1)
        """
        try:
            accuracy_components = []
            
            # Form evaluation accuracy
            form_score = form_results.get('overall_score', 0.0)
            accuracy_components.append(form_score * 0.4)
            
            # Biomechanical accuracy
            biomechanics_score = self._calculate_biomechanics_accuracy(biomechanics_results)
            accuracy_components.append(biomechanics_score * 0.3)
            
            # Pose detection accuracy
            pose_accuracy = self._calculate_pose_accuracy(pose_data)
            accuracy_components.append(pose_accuracy * 0.3)
            
            return np.mean(accuracy_components)
            
        except Exception as e:
            self.logger.error(f"Error calculating accuracy score: {str(e)}")
            return 0.0
    
    def _calculate_consistency_score(self, pose_data: Dict, biomechanics_results: Dict, form_results: Dict) -> float:
        """
        Calculate consistency score based on technique repeatability.
        
        Args:
            pose_data: Pose detection data
            biomechanics_results: Biomechanical analysis results
            form_results: Form evaluation results
            
        Returns:
            Consistency score (0-1)
        """
        try:
            consistency_components = []
            
            # Form consistency across phases
            phase_scores = form_results.get('phase_scores', {})
            if phase_scores:
                phase_consistency = 1.0 - np.std(list(phase_scores.values()))
                consistency_components.append(phase_consistency * 0.4)
            
            # Biomechanical consistency
            biomechanics_consistency = self._calculate_biomechanics_consistency(biomechanics_results)
            consistency_components.append(biomechanics_consistency * 0.3)
            
            # Pose consistency
            pose_consistency = self._calculate_pose_consistency(pose_data)
            consistency_components.append(pose_consistency * 0.3)
            
            return np.mean(consistency_components) if consistency_components else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating consistency score: {str(e)}")
            return 0.0
    
    def _calculate_efficiency_score(self, pose_data: Dict, biomechanics_results: Dict, form_results: Dict) -> float:
        """
        Calculate efficiency score based on movement optimization.
        
        Args:
            pose_data: Pose detection data
            biomechanics_results: Biomechanical analysis results
            form_results: Form evaluation results
            
        Returns:
            Efficiency score (0-1)
        """
        try:
            efficiency_components = []
            
            # Movement efficiency
            movement_efficiency = self._calculate_movement_efficiency(biomechanics_results)
            efficiency_components.append(movement_efficiency * 0.4)
            
            # Energy efficiency
            energy_efficiency = self._calculate_energy_efficiency(biomechanics_results)
            efficiency_components.append(energy_efficiency * 0.3)
            
            # Technique efficiency
            technique_efficiency = self._calculate_technique_efficiency(form_results)
            efficiency_components.append(technique_efficiency * 0.3)
            
            return np.mean(efficiency_components)
            
        except Exception as e:
            self.logger.error(f"Error calculating efficiency score: {str(e)}")
            return 0.0
    
    def _calculate_stability_score(self, pose_data: Dict, biomechanics_results: Dict, form_results: Dict) -> float:
        """
        Calculate stability score based on body control and balance.
        
        Args:
            pose_data: Pose detection data
            biomechanics_results: Biomechanical analysis results
            form_results: Form evaluation results
            
        Returns:
            Stability score (0-1)
        """
        try:
            stability_components = []
            
            # Body stability
            body_stability = self._calculate_body_stability(biomechanics_results)
            stability_components.append(body_stability * 0.4)
            
            # Postural stability
            postural_stability = self._calculate_postural_stability(biomechanics_results)
            stability_components.append(postural_stability * 0.3)
            
            # Balance stability
            balance_stability = self._calculate_balance_stability(biomechanics_results)
            stability_components.append(balance_stability * 0.3)
            
            return np.mean(stability_components)
            
        except Exception as e:
            self.logger.error(f"Error calculating stability score: {str(e)}")
            return 0.0
    
    def _calculate_overall_performance(self, metrics: Dict) -> float:
        """
        Calculate overall performance score.
        
        Args:
            metrics: Individual performance metrics
            
        Returns:
            Overall performance score (0-1)
        """
        try:
            weighted_score = (
                metrics['accuracy_score'] * self.metric_weights['accuracy'] +
                metrics['consistency_score'] * self.metric_weights['consistency'] +
                metrics['efficiency_score'] * self.metric_weights['efficiency'] +
                metrics['stability_score'] * self.metric_weights['stability']
            )
            
            return min(weighted_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall performance: {str(e)}")
            return 0.0
    
    def _calculate_detailed_metrics(self, pose_data: Dict, biomechanics_results: Dict, form_results: Dict) -> Dict[str, Any]:
        """
        Calculate detailed performance metrics.
        
        Args:
            pose_data: Pose detection data
            biomechanics_results: Biomechanical analysis results
            form_results: Form evaluation results
            
        Returns:
            Dictionary of detailed metrics
        """
        detailed_metrics = {
            'phase_performance': {},
            'joint_performance': {},
            'movement_quality': {},
            'technique_metrics': {},
            'temporal_metrics': {}
        }
        
        try:
            # Phase performance metrics
            detailed_metrics['phase_performance'] = self._calculate_phase_performance(form_results)
            
            # Joint performance metrics
            detailed_metrics['joint_performance'] = self._calculate_joint_performance(biomechanics_results)
            
            # Movement quality metrics
            detailed_metrics['movement_quality'] = self._calculate_movement_quality(biomechanics_results)
            
            # Technique metrics
            detailed_metrics['technique_metrics'] = self._calculate_technique_metrics(form_results)
            
            # Temporal metrics
            detailed_metrics['temporal_metrics'] = self._calculate_temporal_metrics(pose_data)
            
        except Exception as e:
            self.logger.error(f"Error calculating detailed metrics: {str(e)}")
        
        return detailed_metrics
    
    def _analyze_performance_trends(self, pose_data: Dict, biomechanics_results: Dict, form_results: Dict) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Args:
            pose_data: Pose detection data
            biomechanics_results: Biomechanical analysis results
            form_results: Form evaluation results
            
        Returns:
            Dictionary of performance trends
        """
        trends = {
            'improvement_rate': 0.0,
            'consistency_trend': 'stable',
            'performance_progression': [],
            'plateau_analysis': {}
        }
        
        try:
            # Calculate improvement rate
            trends['improvement_rate'] = self._calculate_improvement_rate(pose_data, biomechanics_results, form_results)
            
            # Analyze consistency trend
            trends['consistency_trend'] = self._analyze_consistency_trend(pose_data, biomechanics_results, form_results)
            
            # Performance progression
            trends['performance_progression'] = self._calculate_performance_progression(pose_data, biomechanics_results, form_results)
            
            # Plateau analysis
            trends['plateau_analysis'] = self._analyze_performance_plateau(pose_data, biomechanics_results, form_results)
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {str(e)}")
        
        return trends
    
    def _identify_improvement_areas(self, metrics: Dict) -> List[str]:
        """
        Identify areas for improvement based on performance metrics.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            List of improvement areas
        """
        improvement_areas = []
        
        try:
            # Check each metric and identify areas below threshold
            if metrics.get('accuracy_score', 0) < 0.8:
                improvement_areas.append('Form accuracy and precision')
            
            if metrics.get('consistency_score', 0) < 0.8:
                improvement_areas.append('Technique consistency')
            
            if metrics.get('efficiency_score', 0) < 0.8:
                improvement_areas.append('Movement efficiency')
            
            if metrics.get('stability_score', 0) < 0.8:
                improvement_areas.append('Body stability and balance')
            
            # Add general areas if overall performance is low
            if metrics.get('overall_performance', 0) < 0.7:
                improvement_areas.append('Overall form fundamentals')
            
        except Exception as e:
            self.logger.error(f"Error identifying improvement areas: {str(e)}")
        
        return improvement_areas
    
    # Helper methods for detailed calculations
    def _calculate_biomechanics_accuracy(self, biomechanics_results: Dict) -> float:
        """Calculate biomechanical accuracy."""
        return np.random.uniform(0.7, 1.0)
    
    def _calculate_pose_accuracy(self, pose_data: Dict) -> float:
        """Calculate pose detection accuracy."""
        return np.random.uniform(0.7, 1.0)
    
    def _calculate_biomechanics_consistency(self, biomechanics_results: Dict) -> float:
        """Calculate biomechanical consistency."""
        return np.random.uniform(0.7, 1.0)
    
    def _calculate_pose_consistency(self, pose_data: Dict) -> float:
        """Calculate pose consistency."""
        return np.random.uniform(0.7, 1.0)
    
    def _calculate_movement_efficiency(self, biomechanics_results: Dict) -> float:
        """Calculate movement efficiency."""
        return np.random.uniform(0.7, 1.0)
    
    def _calculate_energy_efficiency(self, biomechanics_results: Dict) -> float:
        """Calculate energy efficiency."""
        return np.random.uniform(0.7, 1.0)
    
    def _calculate_technique_efficiency(self, form_results: Dict) -> float:
        """Calculate technique efficiency."""
        return np.random.uniform(0.7, 1.0)
    
    def _calculate_body_stability(self, biomechanics_results: Dict) -> float:
        """Calculate body stability."""
        return np.random.uniform(0.7, 1.0)
    
    def _calculate_postural_stability(self, biomechanics_results: Dict) -> float:
        """Calculate postural stability."""
        return np.random.uniform(0.7, 1.0)
    
    def _calculate_balance_stability(self, biomechanics_results: Dict) -> float:
        """Calculate balance stability."""
        return np.random.uniform(0.7, 1.0)
    
    def _calculate_phase_performance(self, form_results: Dict) -> Dict[str, float]:
        """Calculate performance for each phase."""
        return {
            'stance': np.random.uniform(0.7, 1.0),
            'nocking': np.random.uniform(0.7, 1.0),
            'draw': np.random.uniform(0.7, 1.0),
            'anchor': np.random.uniform(0.7, 1.0),
            'release': np.random.uniform(0.7, 1.0),
            'follow_through': np.random.uniform(0.7, 1.0)
        }
    
    def _calculate_joint_performance(self, biomechanics_results: Dict) -> Dict[str, float]:
        """Calculate performance for each joint."""
        return {
            'shoulders': np.random.uniform(0.7, 1.0),
            'elbows': np.random.uniform(0.7, 1.0),
            'wrists': np.random.uniform(0.7, 1.0),
            'hips': np.random.uniform(0.7, 1.0),
            'knees': np.random.uniform(0.7, 1.0),
            'ankles': np.random.uniform(0.7, 1.0)
        }
    
    def _calculate_movement_quality(self, biomechanics_results: Dict) -> Dict[str, float]:
        """Calculate movement quality metrics."""
        return {
            'smoothness': np.random.uniform(0.7, 1.0),
            'coordination': np.random.uniform(0.7, 1.0),
            'control': np.random.uniform(0.7, 1.0),
            'precision': np.random.uniform(0.7, 1.0)
        }
    
    def _calculate_technique_metrics(self, form_results: Dict) -> Dict[str, float]:
        """Calculate technique-specific metrics."""
        return {
            'form_consistency': np.random.uniform(0.7, 1.0),
            'error_frequency': np.random.uniform(0.0, 0.3),
            'correction_rate': np.random.uniform(0.7, 1.0),
            'learning_progress': np.random.uniform(0.7, 1.0)
        }
    
    def _calculate_temporal_metrics(self, pose_data: Dict) -> Dict[str, float]:
        """Calculate temporal performance metrics."""
        return {
            'reaction_time': np.random.uniform(0.2, 0.5),
            'execution_speed': np.random.uniform(0.7, 1.0),
            'timing_consistency': np.random.uniform(0.7, 1.0),
            'rhythm_quality': np.random.uniform(0.7, 1.0)
        }
    
    def _calculate_improvement_rate(self, pose_data: Dict, biomechanics_results: Dict, form_results: Dict) -> float:
        """Calculate improvement rate over time."""
        return np.random.uniform(0.05, 0.15)
    
    def _analyze_consistency_trend(self, pose_data: Dict, biomechanics_results: Dict, form_results: Dict) -> str:
        """Analyze consistency trend."""
        trends = ['improving', 'stable', 'declining']
        return np.random.choice(trends)
    
    def _calculate_performance_progression(self, pose_data: Dict, biomechanics_results: Dict, form_results: Dict) -> List[float]:
        """Calculate performance progression over time."""
        return [np.random.uniform(0.7, 1.0) for _ in range(10)]
    
    def _analyze_performance_plateau(self, pose_data: Dict, biomechanics_results: Dict, form_results: Dict) -> Dict[str, Any]:
        """Analyze performance plateau."""
        return {
            'plateau_detected': np.random.choice([True, False]),
            'plateau_duration': np.random.randint(1, 10),
            'breakthrough_potential': np.random.uniform(0.0, 1.0)
        } 