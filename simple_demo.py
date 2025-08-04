#!/usr/bin/env python3
"""
ArcheryAI Pro - Simplified Demo
A working demonstration of the core concepts without heavy dependencies
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
import argparse

# Available dependencies
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as patches

class SimplifiedPoseDetector:
    """Simplified pose detection simulation"""
    
    def __init__(self):
        self.joint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def detect_pose(self, frame):
        """Simulate pose detection with realistic archery positions"""
        height, width = frame.shape[:2]
        
        # Simulate archery stance keypoints
        keypoints = {}
        
        # Head position (centered, slightly left)
        keypoints['nose'] = (width * 0.45, height * 0.15)
        keypoints['left_eye'] = (width * 0.43, height * 0.13)
        keypoints['right_eye'] = (width * 0.47, height * 0.13)
        
        # Shoulders (archer facing sideways)
        keypoints['left_shoulder'] = (width * 0.35, height * 0.25)
        keypoints['right_shoulder'] = (width * 0.55, height * 0.25)
        
        # Arms (drawing bow)
        keypoints['left_elbow'] = (width * 0.25, height * 0.35)
        keypoints['right_elbow'] = (width * 0.75, height * 0.30)
        keypoints['left_wrist'] = (width * 0.20, height * 0.40)
        keypoints['right_wrist'] = (width * 0.85, height * 0.25)
        
        # Torso and hips
        keypoints['left_hip'] = (width * 0.40, height * 0.55)
        keypoints['right_hip'] = (width * 0.50, height * 0.55)
        
        # Legs (stable stance)
        keypoints['left_knee'] = (width * 0.38, height * 0.75)
        keypoints['right_knee'] = (width * 0.52, height * 0.75)
        keypoints['left_ankle'] = (width * 0.36, height * 0.95)
        keypoints['right_ankle'] = (width * 0.54, height * 0.95)
        
        # Add some realistic variation
        for joint in keypoints:
            x, y = keypoints[joint]
            x += random.uniform(-10, 10)
            y += random.uniform(-5, 5)
            keypoints[joint] = (max(0, min(width, x)), max(0, min(height, y)))
        
        return keypoints

class SimplifiedBiomechanicsAnalyzer:
    """Simplified biomechanics analysis"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_form(self, keypoints):
        """Analyze archery form from keypoints"""
        results = {}
        
        # Calculate key angles
        results['draw_angle'] = self._calculate_draw_angle(keypoints)
        results['stance_width'] = self._calculate_stance_width(keypoints)
        results['shoulder_alignment'] = self._calculate_shoulder_alignment(keypoints)
        results['elbow_position'] = self._analyze_elbow_position(keypoints)
        
        # Overall form score
        results['form_score'] = self._calculate_form_score(results)
        
        # Generate feedback
        results['feedback'] = self._generate_feedback(results)
        
        return results
    
    def _calculate_draw_angle(self, keypoints):
        """Calculate bow draw angle"""
        if 'right_shoulder' in keypoints and 'right_elbow' in keypoints and 'right_wrist' in keypoints:
            shoulder = np.array(keypoints['right_shoulder'])
            elbow = np.array(keypoints['right_elbow'])
            wrist = np.array(keypoints['right_wrist'])
            
            v1 = shoulder - elbow
            v2 = wrist - elbow
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle)
        return 90.0
    
    def _calculate_stance_width(self, keypoints):
        """Calculate stance width"""
        if 'left_ankle' in keypoints and 'right_ankle' in keypoints:
            left_ankle = np.array(keypoints['left_ankle'])
            right_ankle = np.array(keypoints['right_ankle'])
            return np.linalg.norm(right_ankle - left_ankle)
        return 100.0
    
    def _calculate_shoulder_alignment(self, keypoints):
        """Calculate shoulder alignment"""
        if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
            left_shoulder = np.array(keypoints['left_shoulder'])
            right_shoulder = np.array(keypoints['right_shoulder'])
            
            # Calculate angle from horizontal
            diff = right_shoulder - left_shoulder
            angle = np.arctan2(diff[1], diff[0])
            return np.degrees(angle)
        return 0.0
    
    def _analyze_elbow_position(self, keypoints):
        """Analyze elbow position"""
        if 'right_elbow' in keypoints and 'right_shoulder' in keypoints:
            elbow = np.array(keypoints['right_elbow'])
            shoulder = np.array(keypoints['right_shoulder'])
            
            # Check if elbow is at proper height
            height_diff = abs(elbow[1] - shoulder[1])
            return "Good" if height_diff < 50 else "Needs Adjustment"
        return "Unknown"
    
    def _calculate_form_score(self, results):
        """Calculate overall form score"""
        score = 100
        
        # Deduct points for poor form
        if abs(results['draw_angle'] - 90) > 15:
            score -= 20
        
        if abs(results['shoulder_alignment']) > 10:
            score -= 15
        
        if results['elbow_position'] != "Good":
            score -= 10
        
        return max(0, score)
    
    def _generate_feedback(self, results):
        """Generate corrective feedback"""
        feedback = []
        
        if abs(results['draw_angle'] - 90) > 15:
            feedback.append("Adjust your draw angle - aim for 90 degrees at full draw")
        
        if abs(results['shoulder_alignment']) > 10:
            feedback.append("Keep your shoulders level and square to the target")
        
        if results['elbow_position'] != "Good":
            feedback.append("Raise your drawing elbow to shoulder height")
        
        if results['form_score'] > 85:
            feedback.append("Excellent form! Keep it up!")
        elif results['form_score'] > 70:
            feedback.append("Good form with room for minor improvements")
        else:
            feedback.append("Focus on the key areas mentioned above")
        
        return feedback

class SimplifiedVisualizer:
    """Simplified visualization system"""
    
    def __init__(self):
        self.colors = {
            'excellent': 'green',
            'good': 'yellow',
            'needs_work': 'red'
        }
    
    def create_analysis_visualization(self, frame, keypoints, analysis_results, output_path):
        """Create visualization of the analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ArcheryAI Pro - Biomechanical Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Pose visualization
        ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._draw_skeleton(ax1, keypoints)
        ax1.set_title('Pose Detection & Skeleton')
        ax1.axis('off')
        
        # 2. Form metrics
        self._plot_form_metrics(ax2, analysis_results)
        
        # 3. Score visualization
        self._plot_score_gauge(ax3, analysis_results['form_score'])
        
        # 4. Feedback text
        self._display_feedback(ax4, analysis_results['feedback'])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Analysis visualization saved: {output_path}")
    
    def _draw_skeleton(self, ax, keypoints):
        """Draw skeleton on the pose"""
        # Draw keypoints
        for joint, (x, y) in keypoints.items():
            ax.plot(x, y, 'ro', markersize=8)
            ax.text(x+5, y-5, joint, fontsize=8, color='white', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        # Draw connections
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee'),
            ('left_knee', 'left_ankle'),
            ('right_knee', 'right_ankle')
        ]
        
        for joint1, joint2 in connections:
            if joint1 in keypoints and joint2 in keypoints:
                x1, y1 = keypoints[joint1]
                x2, y2 = keypoints[joint2]
                ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2)
    
    def _plot_form_metrics(self, ax, results):
        """Plot form metrics"""
        metrics = ['Draw Angle', 'Shoulder Alignment', 'Form Score']
        values = [results['draw_angle'], abs(results['shoulder_alignment']), results['form_score']]
        colors = ['green' if v > 80 else 'yellow' if v > 60 else 'red' for v in [90-abs(results['draw_angle']-90), 90-abs(results['shoulder_alignment']), results['form_score']]]
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_title('Form Metrics')
        ax.set_ylabel('Score/Angle')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}', ha='center', va='bottom')
    
    def _plot_score_gauge(self, ax, score):
        """Plot score as a gauge"""
        # Create a semi-circular gauge
        theta = np.linspace(0, np.pi, 100)
        
        # Background arc
        ax.plot(np.cos(theta), np.sin(theta), 'lightgray', linewidth=10)
        
        # Score arc
        score_theta = np.linspace(0, np.pi * (score/100), int(score))
        color = 'green' if score > 85 else 'yellow' if score > 70 else 'red'
        ax.plot(np.cos(score_theta), np.sin(score_theta), color, linewidth=10)
        
        # Score text
        ax.text(0, -0.3, f'{score:.0f}%', ha='center', va='center', 
               fontsize=24, fontweight='bold')
        ax.text(0, -0.5, 'Overall Form Score', ha='center', va='center', fontsize=12)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.7, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Performance Score')
    
    def _display_feedback(self, ax, feedback):
        """Display feedback text"""
        ax.text(0.05, 0.95, 'Corrective Feedback:', transform=ax.transAxes, 
               fontsize=14, fontweight='bold', va='top')
        
        for i, fb in enumerate(feedback):
            ax.text(0.05, 0.85 - i*0.15, f"• {fb}", transform=ax.transAxes, 
                   fontsize=11, va='top', wrap=True)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

def create_sample_frame():
    """Create a sample frame for analysis"""
    # Create a simple background
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
    
    # Add some archery range elements
    cv2.rectangle(frame, (50, 400), (590, 470), (139, 69, 19), -1)  # Ground
    cv2.rectangle(frame, (500, 100), (580, 400), (101, 67, 33), -1)  # Target stand
    cv2.circle(frame, (540, 200), 40, (255, 255, 255), -1)  # Target
    cv2.circle(frame, (540, 200), 35, (255, 0, 0), 3)  # Target rings
    cv2.circle(frame, (540, 200), 25, (255, 255, 0), 3)
    cv2.circle(frame, (540, 200), 15, (0, 255, 0), 3)
    cv2.circle(frame, (540, 200), 5, (255, 0, 0), -1)  # Bullseye
    
    return frame

def run_simplified_demo():
    """Run the simplified ArcheryAI Pro demo"""
    print("🏹 ArcheryAI Pro - Simplified Demo")
    print("=" * 50)
    print("Demonstrating core biomechanical analysis concepts...")
    print()
    
    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    pose_detector = SimplifiedPoseDetector()
    biomechanics_analyzer = SimplifiedBiomechanicsAnalyzer()
    visualizer = SimplifiedVisualizer()
    
    # Create sample frame
    print("📸 Creating sample archery scene...")
    frame = create_sample_frame()
    
    # Detect pose
    print("🎯 Detecting archer pose...")
    keypoints = pose_detector.detect_pose(frame)
    
    # Analyze biomechanics
    print("🔬 Analyzing biomechanics...")
    analysis_results = biomechanics_analyzer.analyze_form(keypoints)
    
    # Create visualization
    print("📊 Creating analysis visualization...")
    viz_path = output_dir / "analysis_results.png"
    visualizer.create_analysis_visualization(frame, keypoints, analysis_results, viz_path)
    
    # Save results to JSON
    results_path = output_dir / "analysis_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
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
    
    print(f"💾 Results saved to: {results_path}")
    
    # Display summary
    print("\n📋 Analysis Summary:")
    print(f"   Form Score: {analysis_results['form_score']:.1f}%")
    print(f"   Draw Angle: {analysis_results['draw_angle']:.1f}°")
    print(f"   Shoulder Alignment: {analysis_results['shoulder_alignment']:.1f}°")
    print(f"   Elbow Position: {analysis_results['elbow_position']}")
    
    print("\n💡 Feedback:")
    for feedback in analysis_results['feedback']:
        print(f"   • {feedback}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📁 Check the '{output_dir}' folder for detailed results and visualizations.")
    
    return str(viz_path), str(results_path)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ArcheryAI Pro - Simplified Demo")
    parser.add_argument("--output", "-o", default="demo_output", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        viz_path, results_path = run_simplified_demo()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📊 Visualization: {viz_path}")
        print(f"📄 Results: {results_path}")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
