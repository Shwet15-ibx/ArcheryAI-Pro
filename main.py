#!/usr/bin/env python3
"""
ArcheryAI Pro - Main Entry Point
Advanced Biomechanical Analysis System for Archery Form Evaluation
"""

import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.core.archery_analyzer import ArcheryAnalyzer
from src.utils.video_processor import VideoProcessor
from src.utils.config_manager import ConfigManager
from src.visualization.report_generator import ReportGenerator
from src.utils.logger import setup_logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ArcheryAI Pro - Advanced Biomechanical Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video archery_video.mp4 --output results/
  python main.py --input_folder videos/ --output results/
  python main.py --realtime --camera 0
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video", 
        type=str, 
        help="Path to single video file for analysis"
    )
    input_group.add_argument(
        "--input_folder", 
        type=str, 
        help="Path to folder containing multiple videos"
    )
    input_group.add_argument(
        "--realtime", 
        action="store_true", 
        help="Enable real-time analysis from camera"
    )
    
    # Output options
    parser.add_argument(
        "--output", 
        type=str, 
        default="results/",
        help="Output directory for analysis results (default: results/)"
    )
    
    # Camera options
    parser.add_argument(
        "--camera", 
        type=int, 
        default=0,
        help="Camera device index for real-time analysis (default: 0)"
    )
    
    # Analysis options
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/analysis_config.yaml",
        help="Path to analysis configuration file"
    )
    
    parser.add_argument(
        "--detailed", 
        action="store_true",
        help="Generate detailed analysis with all metrics"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Show real-time visualization during analysis"
    )
    
    parser.add_argument(
        "--save_frames", 
        action="store_true",
        help="Save individual frames with analysis overlays"
    )
    
    # Performance options
    parser.add_argument(
        "--gpu", 
        action="store_true",
        help="Use GPU acceleration if available"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size for processing (default: 1)"
    )
    
    return parser.parse_args()

def setup_environment():
    """Setup environment and check dependencies."""
    logger = logging.getLogger(__name__)
    
    # Check if required directories exist
    required_dirs = ["config", "models", "data", "results"]
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("GPU not available, using CPU")
    except ImportError:
        logger.warning("PyTorch not available, GPU check skipped")
    
    return True

def process_single_video(video_path, output_dir, config, args):
    """Process a single video file."""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    logger.info(f"Processing video: {video_path}")
    
    # Initialize analyzer
    analyzer = ArcheryAnalyzer(config, use_gpu=args.gpu)
    
    # Process video
    results = analyzer.analyze_video(
        video_path=video_path,
        output_dir=output_dir,
        detailed=args.detailed,
        visualize=args.visualize,
        save_frames=args.save_frames
    )
    
    # Generate report
    report_gen = ReportGenerator()
    report_path = report_gen.generate_report(results, output_dir)
    
    logger.info(f"Analysis complete. Report saved to: {report_path}")
    return True

def process_video_folder(input_folder, output_dir, config, args):
    """Process all videos in a folder."""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(input_folder):
        logger.error(f"Input folder not found: {input_folder}")
        return False
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(input_folder).glob(f"*{ext}"))
        video_files.extend(Path(input_folder).glob(f"*{ext.upper()}"))
    
    if not video_files:
        logger.error(f"No video files found in: {input_folder}")
        return False
    
    logger.info(f"Found {len(video_files)} video files to process")
    
    # Process each video
    success_count = 0
    for video_file in video_files:
        video_output_dir = Path(output_dir) / video_file.stem
        video_output_dir.mkdir(exist_ok=True)
        
        if process_single_video(str(video_file), str(video_output_dir), config, args):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count}/{len(video_files)} videos")
    return success_count == len(video_files)

def run_realtime_analysis(camera_index, output_dir, config, args):
    """Run real-time analysis from camera feed."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting real-time analysis from camera {camera_index}")
    
    # Initialize analyzer
    analyzer = ArcheryAnalyzer(config, use_gpu=args.gpu)
    
    # Start real-time analysis
    analyzer.analyze_realtime(
        camera_index=camera_index,
        output_dir=output_dir,
        visualize=True
    )

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logger()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ArcheryAI Pro - Advanced Biomechanical Analysis System")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup environment
    if not setup_environment():
        logger.error("Environment setup failed")
        sys.exit(1)
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.video:
            # Process single video
            success = process_single_video(args.video, str(output_dir), config, args)
            
        elif args.input_folder:
            # Process video folder
            success = process_video_folder(args.input_folder, str(output_dir), config, args)
            
        elif args.realtime:
            # Real-time analysis
            run_realtime_analysis(args.camera, str(output_dir), config, args)
            success = True
            
        else:
            logger.error("No input specified")
            success = False
        
        if success:
            logger.info("Analysis completed successfully!")
            logger.info(f"Results saved to: {output_dir}")
        else:
            logger.error("Analysis failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 