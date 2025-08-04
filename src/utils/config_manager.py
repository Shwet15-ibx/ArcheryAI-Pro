"""
ArcheryAI Pro - Configuration Manager
Centralized configuration management for the application
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .logger import get_logger

class ConfigManager:
    """
    Configuration manager for ArcheryAI Pro.
    
    Handles loading, validation, and management of application configuration.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the ConfigManager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.logger = get_logger(__name__)
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Default configuration
        self.default_config = {
            'analysis': {
                'pose_detection': {
                    'confidence_threshold': 0.7,
                    'model_complexity': 2,
                    'smooth_landmarks': True
                },
                'biomechanics': {
                    'joint_angle_tolerance': 5.0,
                    'movement_threshold': 0.02,
                    'symmetry_threshold': 0.8
                },
                'form_evaluation': {
                    'error_threshold': 0.5,
                    'phase_weights': {
                        'stance': 0.15,
                        'nocking': 0.10,
                        'draw': 0.25,
                        'anchor': 0.20,
                        'release': 0.20,
                        'follow_through': 0.10
                    }
                },
                'performance_metrics': {
                    'metric_weights': {
                        'accuracy': 0.3,
                        'consistency': 0.25,
                        'efficiency': 0.25,
                        'stability': 0.2
                    },
                    'performance_thresholds': {
                        'excellent': 0.9,
                        'good': 0.8,
                        'average': 0.7,
                        'below_average': 0.6,
                        'poor': 0.5
                    }
                }
            },
            'visualization': {
                '3d_visualization': {
                    'enabled': True,
                    'quality': 'high',
                    'export_formats': ['png', 'mp4', 'html']
                },
                'overlay': {
                    'show_landmarks': True,
                    'show_angles': True,
                    'show_feedback': True,
                    'color_scheme': 'performance'
                }
            },
            'output': {
                'save_frames': False,
                'save_3d_models': True,
                'generate_reports': True,
                'export_formats': ['json', 'csv', 'html']
            },
            'performance': {
                'use_gpu': True,
                'batch_size': 1,
                'max_workers': 4,
                'memory_limit': '4GB'
            }
        }
        
        self.logger.info("ConfigManager initialized successfully")
    
    def load_config(self, config_file: str = "analysis_config.yaml") -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_file: Configuration file name
            
        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / config_file
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    if config_path.suffix.lower() in ['.yaml', '.yml']:
                        config = yaml.safe_load(f)
                    elif config_path.suffix.lower() == '.json':
                        config = json.load(f)
                    else:
                        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
                
                self.logger.info(f"Configuration loaded from {config_path}")
                return self._merge_with_defaults(config)
            else:
                self.logger.warning(f"Config file {config_path} not found, using defaults")
                return self.default_config
                
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            self.logger.info("Using default configuration")
            return self.default_config
    
    def save_config(self, config: Dict[str, Any], config_file: str = "analysis_config.yaml"):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_file: Configuration file name
        """
        config_path = self.config_dir / config_file
        
        try:
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            self.logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
    
    def create_default_config(self, config_file: str = "analysis_config.yaml"):
        """
        Create default configuration file.
        
        Args:
            config_file: Configuration file name
        """
        try:
            self.save_config(self.default_config, config_file)
            self.logger.info(f"Default configuration created: {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating default configuration: {str(e)}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure and values.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid
        """
        try:
            # Check required sections
            required_sections = ['analysis', 'visualization', 'output', 'performance']
            for section in required_sections:
                if section not in config:
                    self.logger.error(f"Missing required configuration section: {section}")
                    return False
            
            # Validate analysis configuration
            if not self._validate_analysis_config(config.get('analysis', {})):
                return False
            
            # Validate visualization configuration
            if not self._validate_visualization_config(config.get('visualization', {})):
                return False
            
            # Validate output configuration
            if not self._validate_output_config(config.get('output', {})):
                return False
            
            # Validate performance configuration
            if not self._validate_performance_config(config.get('performance', {})):
                return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {str(e)}")
            return False
    
    def get_analysis_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get analysis-specific configuration.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Analysis configuration
        """
        return config.get('analysis', {})
    
    def get_visualization_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get visualization-specific configuration.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Visualization configuration
        """
        return config.get('visualization', {})
    
    def get_output_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get output-specific configuration.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Output configuration
        """
        return config.get('output', {})
    
    def get_performance_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get performance-specific configuration.
        
        Args:
            config: Full configuration dictionary
            
        Returns:
            Performance configuration
        """
        return config.get('performance', {})
    
    def update_config(self, config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration with new values.
        
        Args:
            config: Current configuration
            updates: Updates to apply
            
        Returns:
            Updated configuration
        """
        try:
            updated_config = self._deep_merge(config, updates)
            self.logger.info("Configuration updated successfully")
            return updated_config
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            return config
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge loaded configuration with defaults.
        
        Args:
            config: Loaded configuration
            
        Returns:
            Merged configuration
        """
        return self._deep_merge(self.default_config, config)
    
    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            updates: Updates to apply
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_analysis_config(self, analysis_config: Dict[str, Any]) -> bool:
        """Validate analysis configuration."""
        try:
            # Check pose detection config
            pose_config = analysis_config.get('pose_detection', {})
            if not isinstance(pose_config.get('confidence_threshold', 0), (int, float)):
                self.logger.error("Invalid confidence_threshold in pose_detection")
                return False
            
            # Check biomechanics config
            biomechanics_config = analysis_config.get('biomechanics', {})
            if not isinstance(biomechanics_config.get('joint_angle_tolerance', 0), (int, float)):
                self.logger.error("Invalid joint_angle_tolerance in biomechanics")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating analysis config: {str(e)}")
            return False
    
    def _validate_visualization_config(self, viz_config: Dict[str, Any]) -> bool:
        """Validate visualization configuration."""
        try:
            # Check 3D visualization config
            viz_3d_config = viz_config.get('3d_visualization', {})
            if not isinstance(viz_3d_config.get('enabled', True), bool):
                self.logger.error("Invalid enabled flag in 3d_visualization")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating visualization config: {str(e)}")
            return False
    
    def _validate_output_config(self, output_config: Dict[str, Any]) -> bool:
        """Validate output configuration."""
        try:
            # Check output flags
            if not isinstance(output_config.get('save_frames', False), bool):
                self.logger.error("Invalid save_frames flag in output")
                return False
            
            if not isinstance(output_config.get('generate_reports', True), bool):
                self.logger.error("Invalid generate_reports flag in output")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating output config: {str(e)}")
            return False
    
    def _validate_performance_config(self, perf_config: Dict[str, Any]) -> bool:
        """Validate performance configuration."""
        try:
            # Check performance settings
            if not isinstance(perf_config.get('use_gpu', True), bool):
                self.logger.error("Invalid use_gpu flag in performance")
                return False
            
            if not isinstance(perf_config.get('batch_size', 1), int):
                self.logger.error("Invalid batch_size in performance")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating performance config: {str(e)}")
            return False 