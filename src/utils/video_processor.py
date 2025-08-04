"""
ArcheryAI Pro - Video Processor Utility
Video processing and frame manipulation utilities
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Generator
import logging
from datetime import datetime

from .logger import get_logger

class VideoProcessor:
    """
    Video processing utility for ArcheryAI Pro.
    
    Handles video input/output, frame processing, and video manipulation.
    """
    
    def __init__(self):
        """Initialize the VideoProcessor."""
        self.logger = get_logger(__name__)
        self.logger.info("VideoProcessor initialized successfully")
    
    def load_video(self, video_path: str) -> Optional[cv2.VideoCapture]:
        """
        Load a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoCapture object or None if failed
        """
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                self.logger.error(f"Video file not found: {video_path}")
                return None
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return None
            
            self.logger.info(f"Video loaded successfully: {video_path}")
            return cap
            
        except Exception as e:
            self.logger.error(f"Error loading video {video_path}: {str(e)}")
            return None
    
    def get_video_info(self, cap: cv2.VideoCapture) -> Dict[str, Any]:
        """
        Get video information.
        
        Args:
            cap: VideoCapture object
            
        Returns:
            Dictionary containing video information
        """
        try:
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
            }
            
            self.logger.info(f"Video info: {info['width']}x{info['height']}, {info['fps']:.2f} fps, {info['duration']:.2f}s")
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting video info: {str(e)}")
            return {}
    
    def extract_frames(self, cap: cv2.VideoCapture, start_frame: int = 0, end_frame: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Extract frames from video.
        
        Args:
            cap: VideoCapture object
            start_frame: Starting frame number
            end_frame: Ending frame number (None for all frames)
            
        Yields:
            Tuple of (frame_number, frame_array)
        """
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_count = start_frame
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if end_frame is not None and frame_count >= end_frame:
                    break
                
                yield frame_count, frame
                frame_count += 1
                
        except Exception as e:
            self.logger.error(f"Error extracting frames: {str(e)}")
    
    def save_frame(self, frame: np.ndarray, output_path: str, frame_number: Optional[int] = None):
        """
        Save a single frame to file.
        
        Args:
            frame: Frame array
            output_path: Output file path
            frame_number: Frame number for filename
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if frame_number is not None:
                # Add frame number to filename
                stem = output_path.stem
                suffix = output_path.suffix
                output_path = output_path.parent / f"{stem}_{frame_number:04d}{suffix}"
            
            success = cv2.imwrite(str(output_path), frame)
            if success:
                self.logger.debug(f"Frame saved: {output_path}")
            else:
                self.logger.error(f"Failed to save frame: {output_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving frame: {str(e)}")
    
    def create_video_writer(self, output_path: str, width: int, height: int, fps: float = 30.0, codec: str = 'mp4v') -> Optional[cv2.VideoWriter]:
        """
        Create a video writer.
        
        Args:
            output_path: Output video path
            width: Video width
            height: Video height
            fps: Frames per second
            codec: Video codec
            
        Returns:
            VideoWriter object or None if failed
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if not writer.isOpened():
                self.logger.error(f"Could not create video writer: {output_path}")
                return None
            
            self.logger.info(f"Video writer created: {output_path}")
            return writer
            
        except Exception as e:
            self.logger.error(f"Error creating video writer: {str(e)}")
            return None
    
    def process_video_frames(self, cap: cv2.VideoCapture, processor_func, output_path: Optional[str] = None, 
                           save_processed: bool = False, progress_callback=None) -> List[Any]:
        """
        Process video frames with a custom function.
        
        Args:
            cap: VideoCapture object
            processor_func: Function to process each frame
            output_path: Output video path (if saving processed video)
            save_processed: Whether to save processed video
            progress_callback: Callback function for progress updates
            
        Returns:
            List of processing results
        """
        try:
            results = []
            video_info = self.get_video_info(cap)
            total_frames = video_info.get('frame_count', 0)
            
            # Create video writer if needed
            writer = None
            if save_processed and output_path:
                writer = self.create_video_writer(
                    output_path, 
                    video_info['width'], 
                    video_info['height'], 
                    video_info['fps']
                )
            
            # Process frames
            for frame_num, frame in self.extract_frames(cap):
                # Process frame
                result = processor_func(frame, frame_num)
                results.append(result)
                
                # Save processed frame if needed
                if save_processed and writer and result is not None:
                    if isinstance(result, np.ndarray):
                        writer.write(result)
                    else:
                        # Assume result contains processed frame
                        processed_frame = result.get('frame', frame)
                        writer.write(processed_frame)
                
                # Progress callback
                if progress_callback and total_frames > 0:
                    progress = (frame_num + 1) / total_frames
                    progress_callback(progress)
            
            # Clean up
            if writer:
                writer.release()
            
            self.logger.info(f"Processed {len(results)} frames")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing video frames: {str(e)}")
            return []
    
    def resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize frame to target size.
        
        Args:
            frame: Input frame
            target_size: Target (width, height)
            
        Returns:
            Resized frame
        """
        try:
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        except Exception as e:
            self.logger.error(f"Error resizing frame: {str(e)}")
            return frame
    
    def crop_frame(self, frame: np.ndarray, crop_region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop frame to specified region.
        
        Args:
            frame: Input frame
            crop_region: (x, y, width, height)
            
        Returns:
            Cropped frame
        """
        try:
            x, y, w, h = crop_region
            return frame[y:y+h, x:x+w]
        except Exception as e:
            self.logger.error(f"Error cropping frame: {str(e)}")
            return frame
    
    def apply_filters(self, frame: np.ndarray, filters: List[str]) -> np.ndarray:
        """
        Apply filters to frame.
        
        Args:
            frame: Input frame
            filters: List of filter names to apply
            
        Returns:
            Filtered frame
        """
        try:
            processed_frame = frame.copy()
            
            for filter_name in filters:
                if filter_name == 'blur':
                    processed_frame = cv2.GaussianBlur(processed_frame, (5, 5), 0)
                elif filter_name == 'sharpen':
                    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    processed_frame = cv2.filter2D(processed_frame, -1, kernel)
                elif filter_name == 'grayscale':
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                elif filter_name == 'brightness':
                    processed_frame = cv2.convertScaleAbs(processed_frame, alpha=1.2, beta=10)
                elif filter_name == 'contrast':
                    processed_frame = cv2.convertScaleAbs(processed_frame, alpha=1.5, beta=0)
            
            return processed_frame
            
        except Exception as e:
            self.logger.error(f"Error applying filters: {str(e)}")
            return frame
    
    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """
        Extract audio from video file.
        
        Args:
            video_path: Input video path
            audio_path: Output audio path
            
        Returns:
            True if successful
        """
        try:
            # This would require additional audio processing libraries
            # For now, return False as placeholder
            self.logger.warning("Audio extraction not implemented")
            return False
            
        except Exception as e:
            self.logger.error(f"Error extracting audio: {str(e)}")
            return False
    
    def create_video_summary(self, video_path: str, output_path: str, num_frames: int = 10) -> bool:
        """
        Create a video summary with key frames.
        
        Args:
            video_path: Input video path
            output_path: Output summary video path
            num_frames: Number of frames to include in summary
            
        Returns:
            True if successful
        """
        try:
            cap = self.load_video(video_path)
            if not cap:
                return False
            
            video_info = self.get_video_info(cap)
            total_frames = video_info.get('frame_count', 0)
            
            if total_frames == 0:
                return False
            
            # Calculate frame intervals
            interval = max(1, total_frames // num_frames)
            key_frames = []
            
            # Extract key frames
            for i in range(0, total_frames, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    key_frames.append(frame)
                
                if len(key_frames) >= num_frames:
                    break
            
            cap.release()
            
            # Create summary video
            if key_frames:
                writer = self.create_video_writer(
                    output_path,
                    video_info['width'],
                    video_info['height'],
                    video_info['fps']
                )
                
                if writer:
                    for frame in key_frames:
                        writer.write(frame)
                    writer.release()
                    
                    self.logger.info(f"Video summary created: {output_path}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error creating video summary: {str(e)}")
            return False
    
    def get_frame_at_time(self, cap: cv2.VideoCapture, time_seconds: float) -> Optional[np.ndarray]:
        """
        Get frame at specific time.
        
        Args:
            cap: VideoCapture object
            time_seconds: Time in seconds
            
        Returns:
            Frame at specified time or None
        """
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(time_seconds * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                return frame
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting frame at time {time_seconds}s: {str(e)}")
            return None
    
    def cleanup(self, cap: Optional[cv2.VideoCapture] = None):
        """
        Clean up video resources.
        
        Args:
            cap: VideoCapture object to release
        """
        try:
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Video resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 