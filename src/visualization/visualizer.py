"""
ArcheryAI Pro - Visualization Module
Advanced 3D visualization and real-time analysis display
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

from ..utils.logger import get_logger

class Visualizer:
    """
    Advanced visualization system for ArcheryAI Pro.
    
    This class implements patent-worthy 3D visualization techniques
    for archery form analysis, including skeletal reconstruction,
    biomechanical overlays, and performance metrics visualization.
    """
    
    def __init__(self):
        """Initialize the Visualizer."""
        self.logger = get_logger(__name__)
        
        # Color schemes for different visualization types
        self.color_schemes = {
            'performance': {
                'excellent': (0, 255, 0),    # Green
                'good': (0, 255, 255),       # Yellow
                'average': (0, 165, 255),    # Orange
                'below_average': (0, 69, 255), # Red
                'poor': (0, 0, 255)          # Dark Red
            },
            'biomechanics': {
                'optimal': (0, 255, 0),      # Green
                'acceptable': (0, 255, 255),  # Yellow
                'suboptimal': (0, 165, 255),  # Orange
                'poor': (0, 0, 255)          # Red
            },
            'feedback': {
                'positive': (0, 255, 0),     # Green
                'neutral': (255, 255, 0),    # Yellow
                'negative': (0, 0, 255)      # Red
            }
        }
        
        # Visualization settings
        self.viz_settings = {
            'skeleton_line_width': 2,
            'landmark_size': 5,
            'text_size': 0.6,
            'text_thickness': 1,
            'overlay_alpha': 0.7
        }
        
        self.logger.info("Visualizer initialized successfully")
    
    def create_3d_visualization(self, pose_data: Dict, biomechanics_results: Dict, 
                               viz_type: str, output_path: Path) -> Optional[Path]:
        """
        Create 3D visualization of archery form.
        
        Args:
            pose_data: Pose detection data
            biomechanics_results: Biomechanical analysis results
            viz_type: Type of visualization ('skeletal', 'biomechanical', 'performance_overlay', 'comparative')
            output_path: Output directory path
            
        Returns:
            Path to generated visualization file
        """
        try:
            if viz_type == 'skeletal':
                return self._create_skeletal_3d_viz(pose_data, output_path)
            elif viz_type == 'biomechanical':
                return self._create_biomechanical_3d_viz(pose_data, biomechanics_results, output_path)
            elif viz_type == 'performance_overlay':
                return self._create_performance_overlay_3d_viz(pose_data, biomechanics_results, output_path)
            elif viz_type == 'comparative':
                return self._create_comparative_3d_viz(pose_data, biomechanics_results, output_path)
            else:
                self.logger.warning(f"Unknown visualization type: {viz_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating 3D visualization: {str(e)}")
            return None
    
    def draw_realtime_analysis(self, frame: np.ndarray, pose_landmarks: Dict, 
                              biomechanics: Dict, form_score: float) -> np.ndarray:
        """
        Draw real-time analysis overlay on frame.
        
        Args:
            frame: Input frame
            pose_landmarks: Pose landmarks
            biomechanics: Biomechanical analysis results
            form_score: Form evaluation score
            
        Returns:
            Frame with analysis overlay
        """
        try:
            annotated_frame = frame.copy()
            
            # Draw pose landmarks
            if pose_landmarks and 'landmarks' in pose_landmarks:
                annotated_frame = self._draw_pose_landmarks(annotated_frame, pose_landmarks['landmarks'])
            
            # Draw biomechanical information
            if biomechanics:
                annotated_frame = self._draw_biomechanical_overlay(annotated_frame, biomechanics)
            
            # Draw form score
            annotated_frame = self._draw_form_score(annotated_frame, form_score)
            
            # Draw performance indicators
            annotated_frame = self._draw_performance_indicators(annotated_frame, biomechanics)
            
            return annotated_frame
            
        except Exception as e:
            self.logger.error(f"Error drawing real-time analysis: {str(e)}")
            return frame
    
    def create_performance_chart(self, performance_data: Dict, output_path: Path) -> Optional[Path]:
        """
        Create performance analysis charts.
        
        Args:
            performance_data: Performance metrics data
            output_path: Output directory path
            
        Returns:
            Path to generated chart file
        """
        try:
            # Create subplots for different metrics
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Overall Performance', 'Accuracy Score', 'Consistency Score', 'Efficiency Score'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Add performance metrics
            metrics = ['overall_performance', 'accuracy_score', 'consistency_score', 'efficiency_score']
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for metric, pos in zip(metrics, positions):
                if metric in performance_data:
                    value = performance_data[metric]
                    fig.add_trace(
                        go.Indicator(
                            mode="gauge+number+delta",
                            value=value * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': metric.replace('_', ' ').title()},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 60], 'color': "lightgray"},
                                    {'range': [60, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ),
                        row=pos[0], col=pos[1]
                    )
            
            # Update layout
            fig.update_layout(
                title_text="Archery Performance Analysis",
                showlegend=False,
                height=800
            )
            
            # Save chart
            chart_path = output_path / "performance_chart.html"
            fig.write_html(str(chart_path))
            
            self.logger.info(f"Performance chart created: {chart_path}")
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Error creating performance chart: {str(e)}")
            return None
    
    def create_biomechanics_analysis_plot(self, biomechanics_data: Dict, output_path: Path) -> Optional[Path]:
        """
        Create biomechanics analysis plots.
        
        Args:
            biomechanics_data: Biomechanical analysis data
            output_path: Output directory path
            
        Returns:
            Path to generated plot file
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Biomechanical Analysis', fontsize=16)
            
            # Joint angles over time
            if 'joint_angles' in biomechanics_data and biomechanics_data['joint_angles']:
                joint_angles = biomechanics_data['joint_angles']
                time_points = range(len(joint_angles))
                
                # Plot shoulder angles
                if joint_angles and 'left_shoulder' in joint_angles[0]:
                    left_shoulder = [frame.get('left_shoulder', 0) for frame in joint_angles]
                    right_shoulder = [frame.get('right_shoulder', 0) for frame in joint_angles]
                    
                    axes[0, 0].plot(time_points, left_shoulder, label='Left Shoulder', color='blue')
                    axes[0, 0].plot(time_points, right_shoulder, label='Right Shoulder', color='red')
                    axes[0, 0].set_title('Shoulder Angles Over Time')
                    axes[0, 0].set_xlabel('Frame')
                    axes[0, 0].set_ylabel('Angle (degrees)')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True)
                
                # Plot elbow angles
                if joint_angles and 'left_elbow' in joint_angles[0]:
                    left_elbow = [frame.get('left_elbow', 0) for frame in joint_angles]
                    right_elbow = [frame.get('right_elbow', 0) for frame in joint_angles]
                    
                    axes[0, 1].plot(time_points, left_elbow, label='Left Elbow', color='blue')
                    axes[0, 1].plot(time_points, right_elbow, label='Right Elbow', color='red')
                    axes[0, 1].set_title('Elbow Angles Over Time')
                    axes[0, 1].set_xlabel('Frame')
                    axes[0, 1].set_ylabel('Angle (degrees)')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True)
            
            # Symmetry analysis
            if 'symmetry_analysis' in biomechanics_data:
                symmetry = biomechanics_data['symmetry_analysis']
                symmetry_metrics = ['shoulder_symmetry', 'arm_symmetry', 'hip_symmetry', 'leg_symmetry']
                symmetry_values = [symmetry.get(metric, 0) for metric in symmetry_metrics]
                
                axes[1, 0].bar(symmetry_metrics, symmetry_values, color=['green', 'blue', 'orange', 'red'])
                axes[1, 0].set_title('Body Symmetry Analysis')
                axes[1, 0].set_ylabel('Symmetry Score')
                axes[1, 0].set_ylim(0, 1)
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Movement patterns
            if 'movement_patterns' in biomechanics_data and biomechanics_data['movement_patterns']:
                movement_patterns = biomechanics_data['movement_patterns']
                time_points = range(len(movement_patterns))
                
                if movement_patterns and 'overall_stability' in movement_patterns[0]:
                    stability = [frame.get('overall_stability', 0) for frame in movement_patterns]
                    
                    axes[1, 1].plot(time_points, stability, color='purple', linewidth=2)
                    axes[1, 1].set_title('Movement Stability Over Time')
                    axes[1, 1].set_xlabel('Frame')
                    axes[1, 1].set_ylabel('Stability Score')
                    axes[1, 1].grid(True)
            
            # Save plot
            plot_path = output_path / "biomechanics_analysis.png"
            plt.tight_layout()
            plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Biomechanics analysis plot created: {plot_path}")
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error creating biomechanics analysis plot: {str(e)}")
            return None
    
    def _create_skeletal_3d_viz(self, pose_data: Dict, output_path: Path) -> Optional[Path]:
        """Create 3D skeletal visualization."""
        try:
            # Extract 3D landmarks
            landmarks_3d = self._extract_3d_landmarks(pose_data)
            
            if not landmarks_3d:
                return None
            
            # Create 3D plot
            fig = go.Figure()
            
            # Add skeletal connections
            connections = self._get_skeletal_connections()
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx in landmarks_3d and end_idx in landmarks_3d:
                    start_point = landmarks_3d[start_idx]
                    end_point = landmarks_3d[end_idx]
                    
                    fig.add_trace(go.Scatter3d(
                        x=[start_point[0], end_point[0]],
                        y=[start_point[1], end_point[1]],
                        z=[start_point[2], end_point[2]],
                        mode='lines',
                        line=dict(color='blue', width=3),
                        showlegend=False
                    ))
            
            # Add landmarks
            x_coords = [landmarks_3d[idx][0] for idx in landmarks_3d.keys()]
            y_coords = [landmarks_3d[idx][1] for idx in landmarks_3d.keys()]
            z_coords = [landmarks_3d[idx][2] for idx in landmarks_3d.keys()]
            
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='markers',
                marker=dict(color='red', size=5),
                name='Landmarks'
            ))
            
            # Update layout
            fig.update_layout(
                title='3D Skeletal Visualization',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                showlegend=True
            )
            
            # Save visualization
            viz_path = output_path / "skeletal_3d.html"
            fig.write_html(str(viz_path))
            
            self.logger.info(f"3D skeletal visualization created: {viz_path}")
            return viz_path
            
        except Exception as e:
            self.logger.error(f"Error creating skeletal 3D visualization: {str(e)}")
            return None
    
    def _create_biomechanical_3d_viz(self, pose_data: Dict, biomechanics_results: Dict, output_path: Path) -> Optional[Path]:
        """Create 3D biomechanical visualization."""
        try:
            # This would create a more detailed 3D visualization with biomechanical data
            # For now, return a simplified version
            return self._create_skeletal_3d_viz(pose_data, output_path)
            
        except Exception as e:
            self.logger.error(f"Error creating biomechanical 3D visualization: {str(e)}")
            return None
    
    def _create_performance_overlay_3d_viz(self, pose_data: Dict, biomechanics_results: Dict, output_path: Path) -> Optional[Path]:
        """Create 3D visualization with performance overlay."""
        try:
            # This would create a 3D visualization with performance metrics overlay
            # For now, return a simplified version
            return self._create_skeletal_3d_viz(pose_data, output_path)
            
        except Exception as e:
            self.logger.error(f"Error creating performance overlay 3D visualization: {str(e)}")
            return None
    
    def _create_comparative_3d_viz(self, pose_data: Dict, biomechanics_results: Dict, output_path: Path) -> Optional[Path]:
        """Create comparative 3D visualization."""
        try:
            # This would create a side-by-side comparison of different shots or techniques
            # For now, return a simplified version
            return self._create_skeletal_3d_viz(pose_data, output_path)
            
        except Exception as e:
            self.logger.error(f"Error creating comparative 3D visualization: {str(e)}")
            return None
    
    def _draw_pose_landmarks(self, frame: np.ndarray, landmarks: Dict) -> np.ndarray:
        """Draw pose landmarks on frame."""
        try:
            for landmark_id, landmark_data in landmarks.items():
                x, y = int(landmark_data['x']), int(landmark_data['y'])
                confidence = landmark_data.get('visibility', 0)
                
                # Color based on confidence
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green
                elif confidence > 0.6:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                cv2.circle(frame, (x, y), self.viz_settings['landmark_size'], color, -1)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error drawing pose landmarks: {str(e)}")
            return frame
    
    def _draw_biomechanical_overlay(self, frame: np.ndarray, biomechanics: Dict) -> np.ndarray:
        """Draw biomechanical information overlay."""
        try:
            # Draw joint angles if available
            if 'joint_angles' in biomechanics and biomechanics['joint_angles']:
                joint_angles = biomechanics['joint_angles'][-1]  # Latest frame
                
                # Display key angles
                y_offset = 30
                for joint, angle in joint_angles.items():
                    if angle > 0:  # Only show valid angles
                        text = f"{joint}: {angle:.1f}°"
                        cv2.putText(frame, text, (10, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += 20
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error drawing biomechanical overlay: {str(e)}")
            return frame
    
    def _draw_form_score(self, frame: np.ndarray, form_score: float) -> np.ndarray:
        """Draw form score on frame."""
        try:
            # Determine color based on score
            if form_score >= 0.8:
                color = self.color_schemes['performance']['excellent']
            elif form_score >= 0.6:
                color = self.color_schemes['performance']['good']
            else:
                color = self.color_schemes['performance']['poor']
            
            # Draw score
            text = f"Form Score: {form_score:.2f}"
            cv2.putText(frame, text, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error drawing form score: {str(e)}")
            return frame
    
    def _draw_performance_indicators(self, frame: np.ndarray, biomechanics: Dict) -> np.ndarray:
        """Draw performance indicators on frame."""
        try:
            # Draw stability indicator
            if 'movement_patterns' in biomechanics and biomechanics['movement_patterns']:
                stability = biomechanics['movement_patterns'][-1].get('overall_stability', 0)
                
                # Create stability bar
                bar_width = 100
                bar_height = 10
                bar_x = frame.shape[1] - bar_width - 10
                bar_y = 10
                
                # Background bar
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                             (100, 100, 100), -1)
                
                # Stability level
                stability_width = int(bar_width * stability)
                if stability >= 0.8:
                    color = (0, 255, 0)  # Green
                elif stability >= 0.6:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + stability_width, bar_y + bar_height), 
                             color, -1)
                
                # Label
                cv2.putText(frame, "Stability", (bar_x, bar_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error drawing performance indicators: {str(e)}")
            return frame
    
    def _extract_3d_landmarks(self, pose_data: Dict) -> Dict[int, Tuple[float, float, float]]:
        """Extract 3D landmarks from pose data."""
        try:
            landmarks_3d = {}
            
            if 'pose_landmarks' in pose_data and pose_data['pose_landmarks']:
                # Use world landmarks if available
                if 'pose_world_landmarks' in pose_data and pose_data['pose_world_landmarks']:
                    world_landmarks = pose_data['pose_world_landmarks']
                    for i, landmark in enumerate(world_landmarks.landmark):
                        landmarks_3d[i] = (landmark.x, landmark.y, landmark.z)
                else:
                    # Use regular landmarks with estimated Z coordinates
                    landmarks = pose_data['pose_landmarks'][-1]  # Latest frame
                    for landmark_id, landmark_data in landmarks.items():
                        x = landmark_data['x']
                        y = landmark_data['y']
                        z = landmark_data.get('z', 0)  # Use Z if available, otherwise 0
                        landmarks_3d[landmark_id] = (x, y, z)
            
            return landmarks_3d
            
        except Exception as e:
            self.logger.error(f"Error extracting 3D landmarks: {str(e)}")
            return {}
    
    def _get_skeletal_connections(self) -> List[Tuple[int, int]]:
        """Get skeletal connection pairs."""
        # MediaPipe pose connections
        return [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Face
            (0, 4), (4, 5), (5, 6),  # Face
            (10, 9), (9, 8), (8, 6),  # Face
            (6, 12), (12, 11), (11, 10),  # Face
            (12, 14), (14, 16),  # Right arm
            (11, 13), (13, 15),  # Left arm
            (12, 24), (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
            (11, 23), (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
            (24, 23),  # Hips
            (12, 11)   # Shoulders
        ] 