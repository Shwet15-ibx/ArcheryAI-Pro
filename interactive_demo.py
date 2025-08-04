#!/usr/bin/env python3
"""
ArcheryAI Pro - Interactive Demo with Visual Output
Comprehensive demonstration with visual analysis results and web reporting

Developed by: Sports Biomechanics Research Team
Version: 1.0.0
Copyright (c) 2024 ArcheryAI Pro Development Team

This interactive version provides visual feedback, web reports,
and comprehensive analysis visualization capabilities.
"""

import os
import sys
import json
import time
import random
import webbrowser
from pathlib import Path
from datetime import datetime
import argparse

# Available dependencies
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as patches

# Import our simplified components
sys.path.append(str(Path(__file__).parent))
from simple_demo import SimplifiedPoseDetector, SimplifiedBiomechanicsAnalyzer, SimplifiedVisualizer, create_sample_frame

def create_web_report(analysis_results, keypoints, output_dir):
    """Create an HTML report that can be viewed in browser"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ArcheryAI Pro - Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(45deg, #2c3e50, #3498db);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .header p {{
                margin: 10px 0 0 0;
                font-size: 1.2em;
                opacity: 0.9;
            }}
            .content {{
                padding: 30px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .metric-card {{
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                border-left: 5px solid #3498db;
                transition: transform 0.3s ease;
            }}
            .metric-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #2c3e50;
                margin: 10px 0;
            }}
            .metric-label {{
                color: #7f8c8d;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .score-circle {{
                width: 150px;
                height: 150px;
                border-radius: 50%;
                margin: 20px auto;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 2em;
                font-weight: bold;
                color: white;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .score-excellent {{ background: linear-gradient(45deg, #27ae60, #2ecc71); }}
            .score-good {{ background: linear-gradient(45deg, #f39c12, #e67e22); }}
            .score-needs-work {{ background: linear-gradient(45deg, #e74c3c, #c0392b); }}
            .feedback-section {{
                background: #ecf0f1;
                border-radius: 10px;
                padding: 25px;
                margin: 30px 0;
            }}
            .feedback-title {{
                font-size: 1.5em;
                color: #2c3e50;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
            }}
            .feedback-title::before {{
                content: "💡";
                margin-right: 10px;
                font-size: 1.2em;
            }}
            .feedback-item {{
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                border-left: 4px solid #3498db;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .timestamp {{
                text-align: center;
                color: #7f8c8d;
                font-style: italic;
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #ecf0f1;
            }}
            .visualization-section {{
                text-align: center;
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }}
            .btn {{
                display: inline-block;
                padding: 12px 24px;
                background: linear-gradient(45deg, #3498db, #2980b9);
                color: white;
                text-decoration: none;
                border-radius: 6px;
                margin: 10px;
                transition: all 0.3s ease;
                border: none;
                cursor: pointer;
                font-size: 1em;
            }}
            .btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏹 ArcheryAI Pro</h1>
                <p>Advanced Biomechanical Analysis Report</p>
            </div>
            
            <div class="content">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Overall Form Score</div>
                        <div class="score-circle {'score-excellent' if analysis_results['form_score'] > 85 else 'score-good' if analysis_results['form_score'] > 70 else 'score-needs-work'}">
                            {analysis_results['form_score']:.0f}%
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Draw Angle</div>
                        <div class="metric-value">{analysis_results['draw_angle']:.1f}°</div>
                        <small>Target: 90°</small>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Shoulder Alignment</div>
                        <div class="metric-value">{abs(analysis_results['shoulder_alignment']):.1f}°</div>
                        <small>Deviation from level</small>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Elbow Position</div>
                        <div class="metric-value">{analysis_results['elbow_position']}</div>
                        <small>Drawing arm evaluation</small>
                    </div>
                </div>
                
                <div class="visualization-section">
                    <h3>📊 Detailed Analysis Visualization</h3>
                    <p>A comprehensive 4-panel analysis chart has been generated showing pose detection, metrics, and feedback.</p>
                    <button class="btn" onclick="alert('Visualization saved as analysis_results.png in demo_output folder')">
                        View Detailed Charts
                    </button>
                </div>
                
                <div class="feedback-section">
                    <div class="feedback-title">Corrective Feedback & Recommendations</div>
                    {''.join([f'<div class="feedback-item">{feedback}</div>' for feedback in analysis_results['feedback']])}
                </div>
                
                <div class="timestamp">
                    Analysis completed on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                </div>
            </div>
        </div>
        
        <script>
            // Add some interactivity
            document.addEventListener('DOMContentLoaded', function() {{
                const cards = document.querySelectorAll('.metric-card');
                cards.forEach(card => {{
                    card.addEventListener('click', function() {{
                        this.style.transform = 'scale(1.05)';
                        setTimeout(() => {{
                            this.style.transform = '';
                        }}, 200);
                    }});
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    html_path = output_dir / "analysis_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_path

def display_results_in_console(analysis_results, keypoints):
    """Display results in a formatted console output"""
    print("\n" + "="*60)
    print("🏹 ARCHERYAI PRO - ANALYSIS RESULTS")
    print("="*60)
    
    # Form Score with visual indicator
    score = analysis_results['form_score']
    if score > 85:
        score_indicator = "🟢 EXCELLENT"
        score_color = "\033[92m"  # Green
    elif score > 70:
        score_indicator = "🟡 GOOD"
        score_color = "\033[93m"  # Yellow
    else:
        score_indicator = "🔴 NEEDS WORK"
        score_color = "\033[91m"  # Red
    
    print(f"\n📊 OVERALL FORM SCORE: {score_color}{score:.1f}%\033[0m {score_indicator}")
    
    # Create a visual score bar
    bar_length = 40
    filled_length = int(bar_length * score / 100)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    print(f"    [{bar}] {score:.0f}%")
    
    print(f"\n🎯 DETAILED METRICS:")
    print(f"   • Draw Angle:        {analysis_results['draw_angle']:.1f}° (Target: 90°)")
    print(f"   • Shoulder Alignment: {analysis_results['shoulder_alignment']:.1f}° from level")
    print(f"   • Elbow Position:     {analysis_results['elbow_position']}")
    print(f"   • Stance Width:       {analysis_results['stance_width']:.1f} pixels")
    
    print(f"\n💡 CORRECTIVE FEEDBACK:")
    for i, feedback in enumerate(analysis_results['feedback'], 1):
        print(f"   {i}. {feedback}")
    
    print(f"\n📍 KEY POSE POINTS DETECTED:")
    important_joints = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
    for joint in important_joints:
        if joint in keypoints:
            x, y = keypoints[joint]
            print(f"   • {joint.replace('_', ' ').title()}: ({x:.0f}, {y:.0f})")
    
    print("\n" + "="*60)

def run_interactive_demo():
    """Run interactive demo with visual output"""
    print("🏹 ArcheryAI Pro - Interactive Demo with Visual Output")
    print("=" * 60)
    print("Initializing biomechanical analysis system...")
    
    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    print("🔧 Loading analysis components...")
    pose_detector = SimplifiedPoseDetector()
    biomechanics_analyzer = SimplifiedBiomechanicsAnalyzer()
    visualizer = SimplifiedVisualizer()
    
    # Create sample frame
    print("📸 Creating sample archery scene...")
    frame = create_sample_frame()
    
    # Show the sample frame
    print("🖼️  Displaying sample archery scene...")
    cv2.imshow('ArcheryAI Pro - Sample Scene', frame)
    cv2.waitKey(2000)  # Show for 2 seconds
    cv2.destroyAllWindows()
    
    # Detect pose
    print("🎯 Analyzing archer pose...")
    keypoints = pose_detector.detect_pose(frame)
    
    # Analyze biomechanics
    print("🔬 Performing biomechanical analysis...")
    analysis_results = biomechanics_analyzer.analyze_form(keypoints)
    
    # Display results in console
    display_results_in_console(analysis_results, keypoints)
    
    # Create visualization
    print("\n📊 Generating analysis visualization...")
    viz_path = output_dir / "analysis_results.png"
    visualizer.create_analysis_visualization(frame, keypoints, analysis_results, viz_path)
    
    # Create web report
    print("🌐 Creating interactive web report...")
    html_path = create_web_report(analysis_results, keypoints, output_dir)
    
    # Save JSON results
    results_path = output_dir / "analysis_results.json"
    with open(results_path, 'w') as f:
        json_results = {}
        for key, value in analysis_results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, np.floating):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'analysis_results': json_results,
            'keypoints': {k: list(v) for k, v in keypoints.items()}
        }, f, indent=2)
    
    # Show the visualization
    print(f"\n🖼️  Opening analysis visualization...")
    try:
        # Try to display the matplotlib visualization
        img = cv2.imread(str(viz_path))
        if img is not None:
            # Resize if too large
            height, width = img.shape[:2]
            if width > 1200:
                scale = 1200 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            cv2.imshow('ArcheryAI Pro - Analysis Results', img)
            print("📱 Press any key to close the visualization window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"Note: Could not display image directly: {e}")
    
    # Open web report
    print(f"\n🌐 Opening interactive web report...")
    try:
        webbrowser.open(f'file://{html_path.absolute()}')
        print("✅ Web report opened in your default browser!")
    except Exception as e:
        print(f"Note: Could not open browser automatically: {e}")
        print(f"You can manually open: {html_path}")
    
    print(f"\n📁 All results saved to '{output_dir}' folder:")
    print(f"   • 📊 Visualization: {viz_path}")
    print(f"   • 🌐 Web Report: {html_path}")
    print(f"   • 📄 JSON Data: {results_path}")
    
    print(f"\n🎉 Interactive demo completed successfully!")
    print("The ArcheryAI Pro system has analyzed the archery form and provided")
    print("comprehensive feedback with visual results you can now see!")
    
    return str(viz_path), str(html_path), str(results_path)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ArcheryAI Pro - Interactive Demo")
    parser.add_argument("--output", "-o", default="demo_output", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        viz_path, html_path, results_path = run_interactive_demo()
        
        print(f"\n🎯 SUCCESS! All outputs are now visible:")
        print(f"   📊 Charts: {viz_path}")
        print(f"   🌐 Report: {html_path}")
        print(f"   📄 Data: {results_path}")
        
        input("\nPress Enter to exit...")
        
    except Exception as e:
        print(f"❌ Error running interactive demo: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
