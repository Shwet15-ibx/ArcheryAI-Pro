#!/usr/bin/env python3
"""
ArcheryAI Pro - Demo Script
Demonstration of the advanced biomechanical analysis system
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.core.archery_analyzer import ArcheryAnalyzer
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logger
from src.visualization.report_generator import ReportGenerator

def create_sample_video():
    """
    Create a sample video for demonstration purposes.
    This would normally use actual archery footage.
    """
    print("🎬 Creating sample video for demonstration...")
    
    # In a real implementation, this would create or use actual archery footage
    # For demo purposes, we'll create a placeholder
    sample_video_path = Path("data/sample_archery_video.mp4")
    sample_video_path.parent.mkdir(exist_ok=True)
    
    # Create a placeholder file
    with open(sample_video_path, 'w') as f:
        f.write("# This is a placeholder for sample archery video\n")
        f.write("# In a real implementation, this would be actual video footage\n")
    
    print(f"✅ Sample video placeholder created: {sample_video_path}")
    return str(sample_video_path)

def run_demo_analysis(video_path: str, output_dir: str = "demo_results"):
    """
    Run a demonstration analysis.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for results
    """
    print("🏹 Starting ArcheryAI Pro Demo Analysis...")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logger("ArcheryAI_Demo")
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Initialize analyzer
        print("🔧 Initializing ArcheryAI Pro Analyzer...")
        analyzer = ArcheryAnalyzer(config, use_gpu=False)  # Use CPU for demo
        
        # Run analysis
        print("📊 Running comprehensive analysis...")
        results = analyzer.analyze_video(
            video_path=video_path,
            output_dir=output_dir,
            detailed=True,
            visualize=False,
            save_frames=False
        )
        
        # Generate report
        print("📋 Generating comprehensive report...")
        report_generator = ReportGenerator()
        report_path = report_generator.generate_report(results, Path(output_dir))
        
        # Display results
        print("\n" + "=" * 60)
        print("🎯 ANALYSIS RESULTS")
        print("=" * 60)
        
        # Performance metrics
        performance = results.get('performance_metrics', {})
        print(f"Overall Performance: {performance.get('overall_performance', 0):.2%}")
        print(f"Accuracy Score: {performance.get('accuracy_score', 0):.2%}")
        print(f"Consistency Score: {performance.get('consistency_score', 0):.2%}")
        print(f"Efficiency Score: {performance.get('efficiency_score', 0):.2%}")
        print(f"Stability Score: {performance.get('stability_score', 0):.2%}")
        
        # Form evaluation
        form_eval = results.get('analysis_phases', {}).get('form_evaluation', {})
        print(f"\nForm Score: {form_eval.get('overall_score', 0):.2%}")
        
        # Phase scores
        phase_scores = form_eval.get('phase_scores', {})
        print("\nPhase-by-Phase Analysis:")
        for phase, score in phase_scores.items():
            print(f"  {phase.replace('_', ' ').title()}: {score:.2%}")
        
        # Feedback
        feedback = results.get('corrective_feedback', [])
        print(f"\nCorrective Feedback ({len(feedback)} items):")
        for i, item in enumerate(feedback[:3], 1):  # Show first 3
            issue = item.get('issue', {}).get('description', 'Unknown issue')
            correction = item.get('correction', {}).get('description', 'No correction available')
            print(f"  {i}. Issue: {issue}")
            print(f"     Correction: {correction}")
        
        # Visualizations
        visualizations = results.get('3d_visualizations', [])
        print(f"\n3D Visualizations Generated: {len(visualizations)}")
        
        # Report location
        print(f"\n📄 Comprehensive Report: {report_path}")
        
        print("\n" + "=" * 60)
        print("✅ Demo Analysis Completed Successfully!")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"Error during demo analysis: {str(e)}")
        print(f"❌ Demo analysis failed: {str(e)}")
        return None

def run_realtime_demo():
    """Run real-time analysis demo."""
    print("🎥 Starting Real-time Analysis Demo...")
    print("Press 'q' to quit the real-time analysis")
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Initialize analyzer
        analyzer = ArcheryAnalyzer(config, use_gpu=False)
        
        # Start real-time analysis
        analyzer.analyze_realtime(
            camera_index=0,
            output_dir="realtime_demo_results",
            visualize=True
        )
        
    except Exception as e:
        print(f"❌ Real-time demo failed: {str(e)}")

def show_system_info():
    """Display system information and capabilities."""
    print("🏹 ArcheryAI Pro - Advanced Biomechanical Analysis System")
    print("=" * 60)
    print("🔬 Patent-Worthy Features:")
    print("  • Multi-Phase Biomechanical Analysis Pipeline")
    print("  • Real-time 3D Skeletal Reconstruction")
    print("  • Advanced Joint Angle Calculations")
    print("  • Movement Pattern Analysis")
    print("  • Symmetry Detection Algorithms")
    print("  • Predictive Performance Modeling")
    print("  • Adaptive Learning Capabilities")
    print("  • Comprehensive Scoring Methodology")
    print("  • Corrective Feedback Generation")
    print("  • 3D Visualization Engine")
    print()
    print("📊 Analysis Components:")
    print("  • Stance & Posture Analysis")
    print("  • Nocking & Set-up Evaluation")
    print("  • Draw Phase Symmetry Detection")
    print("  • Anchor & Aiming Consistency")
    print("  • Release Mechanics Analysis")
    print("  • Follow-through Trajectory Tracking")
    print()
    print("🎯 Performance Metrics:")
    print("  • Accuracy Score")
    print("  • Consistency Score")
    print("  • Efficiency Score")
    print("  • Stability Score")
    print("  • Overall Performance Rating")
    print()
    print("📈 Output Capabilities:")
    print("  • 3D Visualizations (HTML, PNG, MP4)")
    print("  • Performance Charts")
    print("  • Biomechanical Analysis Plots")
    print("  • Comprehensive Reports (HTML, JSON, CSV)")
    print("  • Real-time Analysis Overlays")
    print("  • Corrective Feedback Recommendations")
    print("=" * 60)

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="ArcheryAI Pro Demo")
    parser.add_argument("--mode", choices=["analysis", "realtime", "info"], 
                       default="analysis", help="Demo mode")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--output", type=str, default="demo_results", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    if args.mode == "info":
        show_system_info()
        return
    
    elif args.mode == "realtime":
        run_realtime_demo()
        return
    
    elif args.mode == "analysis":
        # Use provided video or create sample
        video_path = args.video
        if not video_path:
            video_path = create_sample_video()
        
        # Run analysis
        results = run_demo_analysis(video_path, args.output)
        
        if results:
            print("\n🎉 Demo completed successfully!")
            print("📁 Check the output directory for detailed results and reports.")
        else:
            print("\n❌ Demo failed. Check the logs for details.")
    
    else:
        print("Invalid demo mode. Use --help for options.")

if __name__ == "__main__":
    main() 