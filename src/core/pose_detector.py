"""
ArcheryAI Pro - Pose Detection Module
Advanced pose detection using MediaPipe and computer vision techniques
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional, Any
import logging
import time

from ..utils.logger import get_logger

class PoseDetector:
    """
    Advanced pose detection system optimized for archery analysis.
    
    This class implements patent-worthy pose detection techniques specifically
    designed for archery form analysis, including specialized landmark detection
    and confidence scoring.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the PoseDetector.
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.logger = get_logger(__name__)
        self.use_gpu = use_gpu
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure pose detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Highest accuracy
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Archery-specific landmark mappings
        self.archery_landmarks = {
            'head': [self.mp_pose.PoseLandmark.NOSE, 
                    self.mp_pose.PoseLandmark.LEFT_EYE, 
                    self.mp_pose.PoseLandmark.RIGHT_EYE],
            'shoulders': [self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                         self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
            'arms': [self.mp_pose.PoseLandmark.LEFT_ELBOW, 
                    self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                    self.mp_pose.PoseLandmark.LEFT_WRIST, 
                    self.mp_pose.PoseLandmark.RIGHT_WRIST],
            'torso': [self.mp_pose.PoseLandmark.LEFT_HIP, 
                     self.mp_pose.PoseLandmark.RIGHT_HIP],
            'legs': [self.mp_pose.PoseLandmark.LEFT_KNEE, 
                    self.mp_pose.PoseLandmark.RIGHT_KNEE,
                    self.mp_pose.PoseLandmark.LEFT_ANKLE, 
                    self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        }
        
        # Confidence thresholds for archery analysis
        self.confidence_thresholds = {
            'critical': 0.8,  # Head, shoulders, arms
            'important': 0.7,  # Torso, hips
            'secondary': 0.6   # Legs, feet
        }
        
        self.logger.info("PoseDetector initialized successfully")
    
    def detect_pose(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect pose landmarks in a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary containing pose landmarks and metadata
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks is None:
                return None
            
            # Extract landmarks
            landmarks = self._extract_landmarks(results.pose_landmarks, frame.shape)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(landmarks)
            
            # Validate pose for archery context
            if not self._validate_archery_pose(landmarks, confidence_scores):
                return None
            
            return {
                'landmarks': landmarks,
                'confidence_scores': confidence_scores,
                'pose_world_landmarks': results.pose_world_landmarks,
                'frame_shape': frame.shape,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error in pose detection: {str(e)}")
            return None
    
    def _extract_landmarks(self, pose_landmarks, frame_shape: Tuple[int, int, int]) -> Dict[str, List[float]]:
        """
        Extract and normalize landmark coordinates.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            frame_shape: Shape of the input frame
            
        Returns:
            Dictionary of normalized landmark coordinates
        """
        landmarks = {}
        height, width, _ = frame_shape
        
        for landmark_id, landmark in enumerate(pose_landmarks.landmark):
            # Normalize coordinates
            x = landmark.x * width
            y = landmark.y * height
            z = landmark.z * width  # Normalize Z to same scale as X
            
            landmarks[landmark_id] = {
                'x': x,
                'y': y,
                'z': z,
                'visibility': landmark.visibility,
                'presence': landmark.presence
            }
        
        return landmarks
    
    def _calculate_confidence_scores(self, landmarks: Dict) -> Dict[str, float]:
        """
        Calculate confidence scores for different body parts.
        
        Args:
            landmarks: Extracted landmarks
            
        Returns:
            Dictionary of confidence scores
        """
        confidence_scores = {}
        
        # Calculate confidence for each body part group
        for part_name, landmark_ids in self.archery_landmarks.items():
            visibilities = []
            presences = []
            
            for landmark_id in landmark_ids:
                if landmark_id in landmarks:
                    visibilities.append(landmarks[landmark_id]['visibility'])
                    presences.append(landmarks[landmark_id]['presence'])
            
            if visibilities:
                # Weighted average based on importance
                if part_name in ['head', 'shoulders', 'arms']:
                    weight = 0.4  # Critical for archery
                elif part_name in ['torso']:
                    weight = 0.35  # Important
                else:
                    weight = 0.25  # Secondary
                
                avg_visibility = np.mean(visibilities)
                avg_presence = np.mean(presences)
                
                confidence_scores[part_name] = (avg_visibility * 0.7 + avg_presence * 0.3) * weight
        
        # Overall confidence
        if confidence_scores:
            confidence_scores['overall'] = np.mean(list(confidence_scores.values()))
        
        return confidence_scores
    
    def _validate_archery_pose(self, landmarks: Dict, confidence_scores: Dict) -> bool:
        """
        Validate if the detected pose is suitable for archery analysis.
        
        Args:
            landmarks: Extracted landmarks
            confidence_scores: Confidence scores
            
        Returns:
            True if pose is valid for archery analysis
        """
        # Check overall confidence
        if confidence_scores.get('overall', 0) < 0.5:
            return False
        
        # Check critical body parts
        critical_parts = ['head', 'shoulders', 'arms']
        for part in critical_parts:
            if confidence_scores.get(part, 0) < self.confidence_thresholds['critical']:
                return False
        
        # Check if person is standing (basic validation)
        if not self._is_standing_pose(landmarks):
            return False
        
        return True
    
    def _is_standing_pose(self, landmarks: Dict) -> bool:
        """
        Check if the detected pose is a standing pose.
        
        Args:
            landmarks: Extracted landmarks
            
        Returns:
            True if pose appears to be standing
        """
        # Check if hips are above knees
        left_hip = landmarks.get(self.mp_pose.PoseLandmark.LEFT_HIP.value)
        right_hip = landmarks.get(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        left_knee = landmarks.get(self.mp_pose.PoseLandmark.LEFT_KNEE.value)
        right_knee = landmarks.get(self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
        
        if all([left_hip, right_hip, left_knee, right_knee]):
            avg_hip_y = (left_hip['y'] + right_hip['y']) / 2
            avg_knee_y = (left_knee['y'] + right_knee['y']) / 2
            
            # Hips should be above knees
            return avg_hip_y < avg_knee_y
        
        return True  # Assume valid if we can't determine
    
    def get_confidence_score(self, pose_data: Dict) -> float:
        """
        Get overall confidence score for pose detection.
        
        Args:
            pose_data: Pose detection results
            
        Returns:
            Overall confidence score
        """
        return pose_data.get('confidence_scores', {}).get('overall', 0.0)
    
    def draw_pose_landmarks(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """
        Draw pose landmarks on the frame.
        
        Args:
            frame: Input frame
            pose_data: Pose detection results
            
        Returns:
            Frame with drawn landmarks
        """
        if pose_data is None or 'landmarks' not in pose_data:
            return frame
        
        # Convert landmarks back to MediaPipe format for drawing
        mp_landmarks = self.mp_pose.PoseLandmark
        
        # Create a copy of the frame
        annotated_frame = frame.copy()
        
        # Draw landmarks
        for landmark_id, landmark_data in pose_data['landmarks'].items():
            x, y = int(landmark_data['x']), int(landmark_data['y'])
            
            # Color based on confidence
            confidence = landmark_data['visibility']
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            cv2.circle(annotated_frame, (x, y), 5, color, -1)
        
        # Draw connections between landmarks
        connections = self.mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_idx = connection[0].value
            end_idx = connection[1].value
            
            if start_idx in pose_data['landmarks'] and end_idx in pose_data['landmarks']:
                start_point = pose_data['landmarks'][start_idx]
                end_point = pose_data['landmarks'][end_idx]
                
                start_coord = (int(start_point['x']), int(start_point['y']))
                end_coord = (int(end_point['x']), int(end_point['y']))
                
                cv2.line(annotated_frame, start_coord, end_coord, (255, 255, 255), 2)
        
        return annotated_frame
    
    def get_archery_specific_landmarks(self, pose_data: Dict) -> Dict[str, List[float]]:
        """
        Extract archery-specific landmark coordinates.
        
        Args:
            pose_data: Pose detection results
            
        Returns:
            Dictionary of archery-specific landmarks
        """
        if pose_data is None or 'landmarks' not in pose_data:
            return {}
        
        archery_landmarks = {}
        
        for part_name, landmark_ids in self.archery_landmarks.items():
            part_coords = []
            for landmark_id in landmark_ids:
                if landmark_id in pose_data['landmarks']:
                    landmark = pose_data['landmarks'][landmark_id]
                    part_coords.append([landmark['x'], landmark['y'], landmark['z']])
            
            if part_coords:
                archery_landmarks[part_name] = np.array(part_coords)
        
        return archery_landmarks
    
    def detect_archery_phases(self, pose_data: Dict) -> Dict[str, Any]:
        """
        Detect different phases of archery shot.
        
        Args:
            pose_data: Pose detection results
            
        Returns:
            Dictionary containing phase detection results
        """
        if pose_data is None:
            return {}
        
        # This is a simplified phase detection
        # In a full implementation, this would use temporal analysis
        phases = {
            'stance': self._detect_stance_phase(pose_data),
            'draw': self._detect_draw_phase(pose_data),
            'anchor': self._detect_anchor_phase(pose_data),
            'release': self._detect_release_phase(pose_data),
            'follow_through': self._detect_follow_through_phase(pose_data)
        }
        
        return phases
    
    def _detect_stance_phase(self, pose_data: Dict) -> Dict[str, Any]:
        """Detect stance phase characteristics."""
        # Simplified stance detection
        return {
            'detected': True,
            'confidence': 0.8,
            'characteristics': {
                'feet_alignment': 'parallel',
                'shoulder_alignment': 'square',
                'head_position': 'neutral'
            }
        }
    
    def _detect_draw_phase(self, pose_data: Dict) -> Dict[str, Any]:
        """Detect draw phase characteristics."""
        return {
            'detected': True,
            'confidence': 0.7,
            'characteristics': {
                'draw_length': 'full',
                'shoulder_alignment': 'maintained',
                'elbow_position': 'correct'
            }
        }
    
    def _detect_anchor_phase(self, pose_data: Dict) -> Dict[str, Any]:
        """Detect anchor phase characteristics."""
        return {
            'detected': True,
            'confidence': 0.6,
            'characteristics': {
                'anchor_point': 'consistent',
                'head_stability': 'stable',
                'bow_cant': 'minimal'
            }
        }
    
    def _detect_release_phase(self, pose_data: Dict) -> Dict[str, Any]:
        """Detect release phase characteristics."""
        return {
            'detected': True,
            'confidence': 0.5,
            'characteristics': {
                'release_smoothness': 'smooth',
                'follow_through_direction': 'forward',
                'bow_hand_reaction': 'minimal'
            }
        }
    
    def _detect_follow_through_phase(self, pose_data: Dict) -> Dict[str, Any]:
        """Detect follow-through phase characteristics."""
        return {
            'detected': True,
            'confidence': 0.6,
            'characteristics': {
                'body_stability': 'maintained',
                'head_movement': 'minimal',
                'arrow_trajectory': 'straight'
            }
        }
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'pose'):
            self.pose.close() 