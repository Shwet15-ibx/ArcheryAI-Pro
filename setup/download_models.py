#!/usr/bin/env python3
"""
ArcheryAI Pro - Model Download Script
Download and configure pre-trained models for the analysis system
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import logging
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger

class ModelDownloader:
    """Download and setup pre-trained models for ArcheryAI Pro."""
    
    def __init__(self):
        """Initialize the ModelDownloader."""
        self.logger = setup_logger("ModelDownloader")
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Model URLs and configurations
        self.models = {
            'mediapipe_pose': {
                'url': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
                'filename': 'pose_landmarker_lite.task',
                'description': 'MediaPipe Pose Landmarker Model'
            },
            'yolo_archery': {
                'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
                'filename': 'yolov8n_archery.pt',
                'description': 'YOLO Object Detection Model (for archery equipment)'
            }
        }
    
    def download_models(self):
        """Download all required models."""
        self.logger.info("Starting model download process...")
        
        for model_name, model_info in self.models.items():
            self.logger.info(f"Downloading {model_info['description']}...")
            
            model_path = self.models_dir / model_info['filename']
            
            if model_path.exists():
                self.logger.info(f"Model {model_name} already exists, skipping...")
                continue
            
            try:
                self._download_file(model_info['url'], model_path)
                self.logger.info(f"Successfully downloaded {model_name}")
                
            except Exception as e:
                self.logger.error(f"Error downloading {model_name}: {str(e)}")
        
        self.logger.info("Model download process completed")
    
    def _download_file(self, url: str, filepath: Path):
        """Download a file with progress bar."""
        try:
            # Create a progress bar
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                def progress_hook(count, block_size, total_size):
                    pbar.total = total_size
                    pbar.update(block_size)
                
                urllib.request.urlretrieve(url, filepath, progress_hook)
                
        except Exception as e:
            self.logger.error(f"Error downloading file: {str(e)}")
            raise
    
    def verify_models(self):
        """Verify that all required models are present and valid."""
        self.logger.info("Verifying downloaded models...")
        
        all_valid = True
        
        for model_name, model_info in self.models.items():
            model_path = self.models_dir / model_info['filename']
            
            if not model_path.exists():
                self.logger.error(f"Model {model_name} not found: {model_path}")
                all_valid = False
            else:
                file_size = model_path.stat().st_size
                self.logger.info(f"Model {model_name} verified: {file_size} bytes")
        
        if all_valid:
            self.logger.info("All models verified successfully")
        else:
            self.logger.error("Some models are missing or invalid")
        
        return all_valid
    
    def setup_environment(self):
        """Setup the environment for model usage."""
        self.logger.info("Setting up environment...")
        
        # Create necessary directories
        directories = ['data', 'results', 'logs', 'reports', 'temp']
        
        for dir_name in directories:
            dir_path = Path(dir_name)
            dir_path.mkdir(exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")
        
        # Create model configuration file
        self._create_model_config()
        
        self.logger.info("Environment setup completed")
    
    def _create_model_config(self):
        """Create model configuration file."""
        config_content = {
            'models': {
                'pose_detection': {
                    'type': 'mediapipe',
                    'path': str(self.models_dir / 'pose_landmarker_lite.task'),
                    'enabled': True
                },
                'object_detection': {
                    'type': 'yolo',
                    'path': str(self.models_dir / 'yolov8n_archery.pt'),
                    'enabled': True
                }
            },
            'settings': {
                'use_gpu': True,
                'model_cache': True,
                'optimization_level': 'high'
            }
        }
        
        config_path = self.models_dir / 'model_config.json'
        
        import json
        with open(config_path, 'w') as f:
            json.dump(config_content, f, indent=2)
        
        self.logger.info(f"Model configuration created: {config_path}")

def main():
    """Main function."""
    downloader = ModelDownloader()
    
    try:
        # Download models
        downloader.download_models()
        
        # Verify models
        if downloader.verify_models():
            # Setup environment
            downloader.setup_environment()
            print("✅ Model setup completed successfully!")
        else:
            print("❌ Model verification failed. Please check the logs.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error during model setup: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 