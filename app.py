#!/usr/bin/env python3
"""
ArcheryAI Pro - Production Web Application
Professional archery form analysis platform with web interface

Developed by: Sports Biomechanics Research Team
Version: 2.0.0
Copyright (c) 2024 ArcheryAI Pro Development Team

Production-ready web application for comprehensive archery analysis
with real-time processing, user management, and professional reporting.
"""

import os
import sys
import json
import uuid
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import queue

# Web framework and utilities
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3

# Data processing
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import base64
from io import BytesIO

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our analysis modules
from simple_demo import SimplifiedPoseDetector, SimplifiedBiomechanicsAnalyzer, SimplifiedVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/archeryai_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ArcheryAIApp:
    """Production ArcheryAI Pro Web Application"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = os.environ.get('SECRET_KEY', 'archeryai-pro-2024-secure-key')
        
        # Configuration
        self.app.config.update(
            UPLOAD_FOLDER='uploads',
            RESULTS_FOLDER='results',
            MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100MB max file size
            PERMANENT_SESSION_LIFETIME=timedelta(hours=24)
        )
        
        # Initialize components
        self.pose_detector = SimplifiedPoseDetector()
        self.biomechanics_analyzer = SimplifiedBiomechanicsAnalyzer()
        self.visualizer = SimplifiedVisualizer()
        
        # Analysis queue for background processing
        self.analysis_queue = queue.Queue()
        self.analysis_results = {}
        
        # Setup directories
        self._setup_directories()
        
        # Initialize database
        self._init_database()
        
        # Setup routes
        self._setup_routes()
        
        # Start background worker
        self._start_background_worker()
        
        logger.info("ArcheryAI Pro Application initialized successfully")
    
    def _setup_directories(self):
        """Create necessary directories"""
        dirs = ['uploads', 'results', 'logs', 'static/images', 'static/reports']
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database for user sessions and analysis history"""
        conn = sqlite3.connect('archeryai.db')
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Analysis sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER,
                filename TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                form_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard"""
            return render_template('index.html')
        
        @self.app.route('/analyze')
        def analyze_page():
            """Analysis page"""
            return render_template('analyze.html')
        
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            """Handle file upload and start analysis"""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                if file and self._allowed_file(file.filename):
                    # Generate unique session ID
                    session_id = str(uuid.uuid4())
                    
                    # Save file
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    unique_filename = f"{timestamp}_{session_id[:8]}_{filename}"
                    filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], unique_filename)
                    file.save(filepath)
                    
                    # Store session info
                    self._store_analysis_session(session_id, unique_filename, 'video_upload')
                    
                    # Queue for analysis
                    self.analysis_queue.put({
                        'session_id': session_id,
                        'filepath': filepath,
                        'filename': unique_filename,
                        'analysis_type': 'comprehensive'
                    })
                    
                    return jsonify({
                        'success': True,
                        'session_id': session_id,
                        'message': 'File uploaded successfully. Analysis started.'
                    })
                
                return jsonify({'error': 'Invalid file type'}), 400
                
            except Exception as e:
                logger.error(f"Upload error: {str(e)}")
                return jsonify({'error': 'Upload failed'}), 500
        
        @self.app.route('/status/<session_id>')
        def check_status(session_id):
            """Check analysis status"""
            try:
                conn = sqlite3.connect('archeryai.db')
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT status, form_score, completed_at FROM analysis_sessions WHERE id = ?',
                    (session_id,)
                )
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    status, form_score, completed_at = result
                    return jsonify({
                        'status': status,
                        'form_score': form_score,
                        'completed_at': completed_at,
                        'results_available': status == 'completed'
                    })
                
                return jsonify({'error': 'Session not found'}), 404
                
            except Exception as e:
                logger.error(f"Status check error: {str(e)}")
                return jsonify({'error': 'Status check failed'}), 500
        
        @self.app.route('/results/<session_id>')
        def get_results(session_id):
            """Get analysis results"""
            try:
                if session_id in self.analysis_results:
                    results = self.analysis_results[session_id]
                    
                    # Get detailed metrics from database
                    conn = sqlite3.connect('archeryai.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT metric_name, metric_value, metric_unit 
                        FROM analysis_results 
                        WHERE session_id = ?
                    ''', (session_id,))
                    metrics = cursor.fetchall()
                    conn.close()
                    
                    # Format metrics
                    detailed_metrics = {}
                    for name, value, unit in metrics:
                        detailed_metrics[name] = {
                            'value': value,
                            'unit': unit or ''
                        }
                    
                    return jsonify({
                        'success': True,
                        'results': results,
                        'metrics': detailed_metrics,
                        'visualization_url': f'/visualization/{session_id}',
                        'report_url': f'/report/{session_id}'
                    })
                
                return jsonify({'error': 'Results not found'}), 404
                
            except Exception as e:
                logger.error(f"Results retrieval error: {str(e)}")
                return jsonify({'error': 'Failed to retrieve results'}), 500
        
        @self.app.route('/visualization/<session_id>')
        def get_visualization(session_id):
            """Get analysis visualization"""
            try:
                viz_path = Path(f'results/{session_id}_visualization.png')
                if viz_path.exists():
                    return send_file(str(viz_path), mimetype='image/png')
                
                return jsonify({'error': 'Visualization not found'}), 404
                
            except Exception as e:
                logger.error(f"Visualization error: {str(e)}")
                return jsonify({'error': 'Failed to load visualization'}), 500
        
        @self.app.route('/report/<session_id>')
        def get_report(session_id):
            """Get HTML report"""
            try:
                report_path = Path(f'results/{session_id}_report.html')
                if report_path.exists():
                    return send_file(str(report_path))
                
                return jsonify({'error': 'Report not found'}), 404
                
            except Exception as e:
                logger.error(f"Report error: {str(e)}")
                return jsonify({'error': 'Failed to load report'}), 500
        
        @self.app.route('/api/live-analysis', methods=['POST'])
        def live_analysis():
            """Real-time analysis endpoint"""
            try:
                data = request.get_json()
                if not data or 'image_data' not in data:
                    return jsonify({'error': 'No image data provided'}), 400
                
                # Decode base64 image
                image_data = data['image_data'].split(',')[1]
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Perform quick analysis
                keypoints = self.pose_detector.detect_pose(frame)
                analysis = self.biomechanics_analyzer.analyze_form(keypoints)
                
                return jsonify({
                    'success': True,
                    'form_score': analysis['form_score'],
                    'draw_angle': analysis['draw_angle'],
                    'shoulder_alignment': analysis['shoulder_alignment'],
                    'feedback': analysis['feedback'][:2]  # Top 2 feedback items
                })
                
            except Exception as e:
                logger.error(f"Live analysis error: {str(e)}")
                return jsonify({'error': 'Analysis failed'}), 500
        
        @self.app.route('/dashboard')
        def dashboard():
            """User dashboard with analysis history"""
            try:
                conn = sqlite3.connect('archeryai.db')
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, filename, analysis_type, status, form_score, created_at, completed_at
                    FROM analysis_sessions 
                    ORDER BY created_at DESC 
                    LIMIT 20
                ''')
                sessions = cursor.fetchall()
                conn.close()
                
                return render_template('dashboard.html', sessions=sessions)
                
            except Exception as e:
                logger.error(f"Dashboard error: {str(e)}")
                return render_template('dashboard.html', sessions=[], error="Failed to load dashboard")
    
    def _allowed_file(self, filename):
        """Check if file type is allowed"""
        allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'jpg', 'jpeg', 'png'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
    def _store_analysis_session(self, session_id, filename, analysis_type):
        """Store analysis session in database"""
        try:
            conn = sqlite3.connect('archeryai.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analysis_sessions (id, filename, analysis_type, status)
                VALUES (?, ?, ?, 'pending')
            ''', (session_id, filename, analysis_type))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
    
    def _start_background_worker(self):
        """Start background worker for analysis processing"""
        def worker():
            while True:
                try:
                    # Get analysis job from queue
                    job = self.analysis_queue.get(timeout=1)
                    self._process_analysis_job(job)
                    self.analysis_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Worker error: {str(e)}")
        
        # Start worker thread
        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        logger.info("Background worker started")
    
    def _process_analysis_job(self, job):
        """Process analysis job in background"""
        session_id = job['session_id']
        filepath = job['filepath']
        
        try:
            logger.info(f"Processing analysis for session {session_id}")
            
            # Update status to processing
            self._update_session_status(session_id, 'processing')
            
            # Determine if it's an image or video
            file_ext = Path(filepath).suffix.lower()
            
            if file_ext in ['.jpg', '.jpeg', '.png']:
                # Process image
                frame = cv2.imread(filepath)
                results = self._analyze_frame(frame, session_id)
            else:
                # For video files, analyze first frame (simplified for demo)
                cap = cv2.VideoCapture(filepath)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    results = self._analyze_frame(frame, session_id)
                else:
                    raise Exception("Could not read video file")
            
            # Store results
            self.analysis_results[session_id] = results
            
            # Update database
            self._update_session_status(session_id, 'completed', results['form_score'])
            self._store_detailed_metrics(session_id, results)
            
            logger.info(f"Analysis completed for session {session_id}")
            
        except Exception as e:
            logger.error(f"Analysis failed for session {session_id}: {str(e)}")
            self._update_session_status(session_id, 'failed')
    
    def _analyze_frame(self, frame, session_id):
        """Analyze a single frame"""
        # Detect pose
        keypoints = self.pose_detector.detect_pose(frame)
        
        # Analyze biomechanics
        analysis = self.biomechanics_analyzer.analyze_form(keypoints)
        
        # Create visualization
        viz_path = Path(f'results/{session_id}_visualization.png')
        self.visualizer.create_analysis_visualization(frame, keypoints, analysis, str(viz_path))
        
        # Create HTML report
        self._create_html_report(session_id, analysis, keypoints)
        
        return analysis
    
    def _create_html_report(self, session_id, analysis, keypoints):
        """Create comprehensive HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ArcheryAI Pro - Analysis Report #{session_id[:8]}</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ padding: 30px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
                .score {{ font-size: 2.5em; font-weight: bold; color: #2c3e50; }}
                .feedback {{ background: #ecf0f1; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .timestamp {{ text-align: center; color: #7f8c8d; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ecf0f1; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏹 ArcheryAI Pro Analysis Report</h1>
                    <p>Session ID: {session_id[:8]}</p>
                </div>
                <div class="content">
                    <div class="metric-grid">
                        <div class="metric-card">
                            <h3>Overall Form Score</h3>
                            <div class="score">{analysis['form_score']:.0f}%</div>
                        </div>
                        <div class="metric-card">
                            <h3>Draw Angle</h3>
                            <div class="score">{analysis['draw_angle']:.1f}°</div>
                        </div>
                        <div class="metric-card">
                            <h3>Shoulder Alignment</h3>
                            <div class="score">{abs(analysis['shoulder_alignment']):.1f}°</div>
                        </div>
                        <div class="metric-card">
                            <h3>Elbow Position</h3>
                            <div class="score">{analysis['elbow_position']}</div>
                        </div>
                    </div>
                    
                    <div class="feedback">
                        <h3>💡 Professional Feedback</h3>
                        {''.join([f'<p>• {feedback}</p>' for feedback in analysis['feedback']])}
                    </div>
                    
                    <div class="timestamp">
                        Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br>
                        <small>ArcheryAI Pro v2.0 - Professional Biomechanical Analysis System</small>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        report_path = Path(f'results/{session_id}_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _update_session_status(self, session_id, status, form_score=None):
        """Update session status in database"""
        try:
            conn = sqlite3.connect('archeryai.db')
            cursor = conn.cursor()
            
            if form_score is not None:
                cursor.execute('''
                    UPDATE analysis_sessions 
                    SET status = ?, form_score = ?, completed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (status, form_score, session_id))
            else:
                cursor.execute('''
                    UPDATE analysis_sessions 
                    SET status = ?
                    WHERE id = ?
                ''', (status, session_id))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database update error: {str(e)}")
    
    def _store_detailed_metrics(self, session_id, results):
        """Store detailed metrics in database"""
        try:
            conn = sqlite3.connect('archeryai.db')
            cursor = conn.cursor()
            
            metrics = [
                ('draw_angle', results['draw_angle'], 'degrees'),
                ('shoulder_alignment', results['shoulder_alignment'], 'degrees'),
                ('stance_width', results['stance_width'], 'pixels'),
                ('form_score', results['form_score'], 'percentage')
            ]
            
            for name, value, unit in metrics:
                cursor.execute('''
                    INSERT INTO analysis_results (session_id, metric_name, metric_value, metric_unit)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, name, value, unit))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Metrics storage error: {str(e)}")
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the application"""
        logger.info(f"Starting ArcheryAI Pro Application on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def create_app():
    """Application factory"""
    return ArcheryAIApp()

if __name__ == '__main__':
    # Create and run the application
    app = create_app()
    
    # Get configuration from environment
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(host=host, port=port, debug=debug)
