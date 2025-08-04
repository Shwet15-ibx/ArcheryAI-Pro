#!/usr/bin/env python3
"""
ArcheryAI Pro - Production Server Launcher
Professional deployment script for the archery analysis platform

Developed by: Sports Biomechanics Research Team
Version: 2.0.0
Copyright (c) 2024 ArcheryAI Pro Development Team

Production-ready server with proper configuration, logging, and monitoring.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app import create_app
from config import config

def setup_production_environment():
    """Setup production environment and configuration"""
    
    # Set default environment
    os.environ.setdefault('FLASK_ENV', 'production')
    os.environ.setdefault('FLASK_CONFIG', 'production')
    
    # Create necessary directories
    directories = ['logs', 'uploads', 'results', 'static/images', 'static/reports']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/archeryai_production.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("ArcheryAI Pro Production Environment Setup Complete")
    
    return logger

def main():
    """Main production server entry point"""
    
    # Setup environment
    logger = setup_production_environment()
    
    try:
        # Create Flask application
        logger.info("Initializing ArcheryAI Pro Application...")
        app = create_app()
        
        # Get configuration from environment
        host = os.environ.get('HOST', '0.0.0.0')
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('DEBUG', 'False').lower() == 'true'
        
        logger.info(f"Starting ArcheryAI Pro Production Server")
        logger.info(f"Host: {host}")
        logger.info(f"Port: {port}")
        logger.info(f"Debug: {debug}")
        
        # Production server recommendations
        if not debug:
            logger.info("Production Mode: Consider using Gunicorn for better performance")
            logger.info("Example: gunicorn -w 4 -b 0.0.0.0:5000 'run_production:create_app()'")
        
        # Start production server
        print("🚀 Starting ArcheryAI Pro production server...")
        app.run(host=host, port=port, debug=debug)
        
    except Exception as e:
        logger.error(f"Failed to start ArcheryAI Pro: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
