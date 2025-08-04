#!/usr/bin/env python3
"""
ArcheryAI Pro - Production Configuration
Configuration settings for production deployment

Developed by: Sports Biomechanics Research Team
Version: 2.0.0
Copyright (c) 2024 ArcheryAI Pro Development Team
"""

import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Application Settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'archeryai-pro-production-key-2024'
    
    # Database Configuration
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///archeryai_production.db'
    
    # File Upload Settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER') or 'results'
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))  # 100MB
    
    # Analysis Settings
    ANALYSIS_TIMEOUT = int(os.environ.get('ANALYSIS_TIMEOUT', 300))  # 5 minutes
    MAX_CONCURRENT_ANALYSES = int(os.environ.get('MAX_CONCURRENT_ANALYSES', 5))
    
    # Security Settings
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'True').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'logs/archeryai_production.log')
    
    # Performance Settings
    ENABLE_CACHING = os.environ.get('ENABLE_CACHING', 'True').lower() == 'true'
    CACHE_TIMEOUT = int(os.environ.get('CACHE_TIMEOUT', 3600))  # 1 hour
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create necessary directories
        directories = [
            Config.UPLOAD_FOLDER,
            Config.RESULTS_FOLDER,
            'logs',
            'static/images',
            'static/reports'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    SESSION_COOKIE_SECURE = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True
    PERMANENT_SESSION_LIFETIME = 86400  # 24 hours
    
    # Production database (can be PostgreSQL, MySQL, etc.)
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///archeryai_production.db'
    
    # Production logging
    LOG_LEVEL = 'WARNING'
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Production-specific initialization
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Setup rotating file handler
        if not app.debug and not app.testing:
            if not os.path.exists('logs'):
                os.mkdir('logs')
            
            file_handler = RotatingFileHandler(
                'logs/archeryai_production.log',
                maxBytes=10240000,  # 10MB
                backupCount=10
            )
            
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            app.logger.setLevel(logging.INFO)
            app.logger.info('ArcheryAI Pro startup')

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
