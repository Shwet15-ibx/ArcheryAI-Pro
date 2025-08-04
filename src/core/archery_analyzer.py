"""
ArcheryAI Pro - Core Analysis Engine
Advanced Biomechanical Analysis for Archery Form Evaluation
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import time
from datetime import datetime

from .pose_detector import PoseDetector
from .biomechanics_analyzer import BiomechanicsAnalyzer
from .form_evaluator import FormEvaluator
from .performance_metrics import PerformanceMetrics
from ..utils.video_processor import VideoProcessor
from ..visualization.visualizer import Visualizer
from ..utils.logger import get_logger

class ArcheryAnalyzer:
    """
    Main analysis engine for archery form evaluation.
    
    This class implements the patent-worthy biomechanical analysis pipeline
    that processes video input and generates comprehensive 3D visualizations
    with corrective feedback.
    """
    
    def __init__(self, config: Dict, use_gpu: bool = False):
        """
        Initialize the ArcheryAnalyzer.
        
        Args:
            config: Configuration dictionary
            use_gpu: Whether to use GPU acceleration
        """
        self.logger = get_logger(__name__)
        self.config = config
        self.use_gpu = use_gpu
        
        # Initialize components
        self.pose_detector = PoseDetector(use_gpu=use_gpu)
        self.biomechanics_analyzer = BiomechanicsAnalyzer(config)
        self.form_evaluator = FormEvaluator(config)
        self.performance_metrics = PerformanceMetrics(config)
        self.video_processor = VideoProcessor()
        self.visualizer = Visualizer()
        
        # Analysis state
        self.current_analysis = None
        self.analysis_results = {}
        
        self.logger.info("ArcheryAnalyzer initialized successfully")
    
    def analyze_video(self, 
                     video_path: str, 
                     output_dir: str,
                     detailed: bool = False,
                     visualize: bool = False,
                     save_frames: bool = False) -> Dict[str, Any]:
        """
        Analyze archery form from video input.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save results
            detailed: Whether to generate detailed analysis
            visualize: Whether to show real-time visualization
            save_frames: Whether to save individual frames
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info(f"Starting video analysis: {video_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize analysis results
        analysis_results = {
            'video_path': video_path,
            'timestamp': datetime.now().isoformat(),
            'analysis_phases': {},
            'performance_metrics': {},
            'corrective_feedback': [],
            '3d_visualizations': [],
            'detailed_analysis': detailed
        }
        
        try:
            # Phase 1: Video Processing and Pose Detection
            self.logger.info("Phase 1: Video processing and pose detection")
            pose_data = self._process_video_poses(video_path, output_path, save_frames)
            
            # Phase 2: Biomechanical Analysis
            self.logger.info("Phase 2: Biomechanical analysis")
            biomechanics_results = self._analyze_biomechanics(pose_data, detailed)
            
            # Phase 3: Form Evaluation
            self.logger.info("Phase 3: Form evaluation")
            form_results = self._evaluate_form(pose_data, biomechanics_results)
            
            # Phase 4: Performance Metrics
            self.logger.info("Phase 4: Performance metrics calculation")
            performance_results = self._calculate_performance_metrics(
                pose_data, biomechanics_results, form_results
            )
            
            # Phase 5: 3D Visualization Generation
            self.logger.info("Phase 5: 3D visualization generation")
            visualization_results = self._generate_3d_visualizations(
                pose_data, biomechanics_results, output_path
            )
            
            # Phase 6: Corrective Feedback Generation
            self.logger.info("Phase 6: Corrective feedback generation")
            feedback_results = self._generate_corrective_feedback(
                biomechanics_results, form_results, performance_results
            )
            
            # Compile final results
            analysis_results.update({
                'analysis_phases': {
                    'pose_detection': pose_data,
                    'biomechanics': biomechanics_results,
                    'form_evaluation': form_results,
                    'performance_metrics': performance_results,
                    'visualization': visualization_results,
                    'feedback': feedback_results
                },
                'performance_metrics': performance_results,
                'corrective_feedback': feedback_results,
                '3d_visualizations': visualization_results
            })
            
            # Save results
            self._save_analysis_results(analysis_results, output_path)
            
            self.logger.info("Video analysis completed successfully")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error during video analysis: {str(e)}")
            raise
    
    def analyze_realtime(self, 
                        camera_index: int = 0,
                        output_dir: str = "realtime_results/",
                        visualize: bool = True):
        """
        Perform real-time analysis from camera feed.
        
        Args:
            camera_index: Camera device index
            output_dir: Directory to save results
            visualize: Whether to show real-time visualization
        """
        self.logger.info(f"Starting real-time analysis from camera {camera_index}")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                pose_landmarks = self.pose_detector.detect_pose(frame)
                
                if pose_landmarks is not None:
                    # Real-time biomechanical analysis
                    biomechanics = self.biomechanics_analyzer.analyze_frame(pose_landmarks)
                    
                    # Real-time form evaluation
                    form_score = self.form_evaluator.evaluate_frame(pose_landmarks, biomechanics)
                    
                    # Real-time visualization
                    if visualize:
                        annotated_frame = self.visualizer.draw_realtime_analysis(
                            frame, pose_landmarks, biomechanics, form_score
                        )
                        cv2.imshow('ArcheryAI Pro - Real-time Analysis', annotated_frame)
                
                frame_count += 1
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Save real-time analysis summary
            duration = time.time() - start_time
            summary = {
                'frames_processed': frame_count,
                'duration_seconds': duration,
                'fps': frame_count / duration if duration > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(output_path / 'realtime_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
    
    def _process_video_poses(self, video_path: str, output_path: Path, save_frames: bool) -> Dict:
        """Process video and extract pose data for each frame."""
        self.logger.info("Processing video poses...")
        
        pose_data = {
            'frames': [],
            'pose_landmarks': [],
            'confidence_scores': [],
            'frame_timestamps': []
        }
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect pose in frame
            pose_landmarks = self.pose_detector.detect_pose(frame)
            
            if pose_landmarks is not None:
                pose_data['frames'].append(frame_count)
                pose_data['pose_landmarks'].append(pose_landmarks)
                pose_data['confidence_scores'].append(
                    self.pose_detector.get_confidence_score(pose_landmarks)
                )
                pose_data['frame_timestamps'].append(frame_count / cap.get(cv2.CAP_PROP_FPS))
                
                # Save frame if requested
                if save_frames:
                    frame_path = output_path / f"frame_{frame_count:04d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
            
            frame_count += 1
        
        cap.release()
        
        self.logger.info(f"Processed {len(pose_data['frames'])} frames with pose data")
        return pose_data
    
    def _analyze_biomechanics(self, pose_data: Dict, detailed: bool) -> Dict:
        """Analyze biomechanical aspects of archery form."""
        self.logger.info("Analyzing biomechanics...")
        
        biomechanics_results = {
            'stance_analysis': {},
            'draw_phase_analysis': {},
            'release_analysis': {},
            'follow_through_analysis': {},
            'joint_angles': [],
            'movement_patterns': [],
            'symmetry_analysis': {}
        }
        
        # Analyze each frame
        for i, landmarks in enumerate(pose_data['pose_landmarks']):
            frame_analysis = self.biomechanics_analyzer.analyze_frame(landmarks)
            
            # Store joint angles
            biomechanics_results['joint_angles'].append(frame_analysis['joint_angles'])
            
            # Store movement patterns
            biomechanics_results['movement_patterns'].append(frame_analysis['movement_patterns'])
        
        # Perform detailed analysis if requested
        if detailed:
            biomechanics_results.update(
                self.biomechanics_analyzer.perform_detailed_analysis(pose_data)
            )
        
        return biomechanics_results
    
    def _evaluate_form(self, pose_data: Dict, biomechanics_results: Dict) -> Dict:
        """Evaluate overall archery form and technique."""
        self.logger.info("Evaluating archery form...")
        
        form_results = {
            'overall_score': 0.0,
            'phase_scores': {},
            'error_detection': [],
            'technique_assessment': {},
            'consistency_metrics': {}
        }
        
        # Evaluate each phase
        phases = ['stance', 'nocking', 'draw', 'anchor', 'release', 'follow_through']
        
        for phase in phases:
            phase_score = self.form_evaluator.evaluate_phase(
                pose_data, biomechanics_results, phase
            )
            form_results['phase_scores'][phase] = phase_score
        
        # Calculate overall score
        form_results['overall_score'] = np.mean(list(form_results['phase_scores'].values()))
        
        # Detect errors
        form_results['error_detection'] = self.form_evaluator.detect_errors(
            pose_data, biomechanics_results
        )
        
        return form_results
    
    def _calculate_performance_metrics(self, pose_data: Dict, biomechanics_results: Dict, form_results: Dict) -> Dict:
        """Calculate comprehensive performance metrics."""
        self.logger.info("Calculating performance metrics...")
        
        performance_results = {
            'accuracy_score': 0.0,
            'consistency_score': 0.0,
            'efficiency_score': 0.0,
            'stability_score': 0.0,
            'overall_performance': 0.0,
            'detailed_metrics': {}
        }
        
        # Calculate various performance metrics
        performance_results.update(
            self.performance_metrics.calculate_all_metrics(
                pose_data, biomechanics_results, form_results
            )
        )
        
        return performance_results
    
    def _generate_3d_visualizations(self, pose_data: Dict, biomechanics_results: Dict, output_path: Path) -> List[str]:
        """Generate 3D visualizations of archery form."""
        self.logger.info("Generating 3D visualizations...")
        
        visualization_paths = []
        
        # Generate different types of 3D visualizations
        viz_types = ['skeletal', 'biomechanical', 'performance_overlay', 'comparative']
        
        for viz_type in viz_types:
            viz_path = self.visualizer.create_3d_visualization(
                pose_data, biomechanics_results, viz_type, output_path
            )
            if viz_path:
                visualization_paths.append(str(viz_path))
        
        return visualization_paths
    
    def _generate_corrective_feedback(self, biomechanics_results: Dict, form_results: Dict, performance_results: Dict) -> List[Dict]:
        """Generate corrective feedback based on analysis results."""
        self.logger.info("Generating corrective feedback...")
        
        feedback = []
        
        # Generate feedback for each detected issue
        for error in form_results['error_detection']:
            correction = self.form_evaluator.generate_correction(error)
            feedback.append({
                'issue': error,
                'correction': correction,
                'priority': self.form_evaluator.get_priority(error),
                'video_timestamp': error.get('timestamp', 0)
            })
        
        # Add performance improvement suggestions
        performance_feedback = self.performance_metrics.generate_improvement_suggestions(
            performance_results
        )
        feedback.extend(performance_feedback)
        
        return feedback
    
    def _save_analysis_results(self, results: Dict, output_path: Path):
        """Save analysis results to files."""
        self.logger.info("Saving analysis results...")
        
        # Save main results
        with open(output_path / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary = {
            'overall_score': results['performance_metrics'].get('overall_performance', 0),
            'form_score': results['analysis_phases']['form_evaluation']['overall_score'],
            'feedback_count': len(results['corrective_feedback']),
            'visualization_count': len(results['3d_visualizations']),
            'timestamp': results['timestamp']
        }
        
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def get_analysis_summary(self) -> Dict:
        """Get a summary of the current analysis."""
        if not self.analysis_results:
            return {}
        
        return {
            'overall_score': self.analysis_results.get('performance_metrics', {}).get('overall_performance', 0),
            'form_score': self.analysis_results.get('analysis_phases', {}).get('form_evaluation', {}).get('overall_score', 0),
            'feedback_count': len(self.analysis_results.get('corrective_feedback', [])),
            'visualization_count': len(self.analysis_results.get('3d_visualizations', []))
        } 