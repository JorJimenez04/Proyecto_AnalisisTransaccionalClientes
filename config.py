import os
from typing import Dict, Any

class Config:
    """Base configuration class"""
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Dash specific
    DASH_HOST = os.environ.get('DASH_HOST', '0.0.0.0')
    DASH_PORT = int(os.environ.get('PORT', 8050))
    DASH_DEBUG = os.environ.get('DASH_DEBUG', 'False').lower() == 'true'
    
    # File upload limits
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    @staticmethod
    def init_app(app):
        pass

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
    # Redis for session storage (optional)
    REDIS_URL = os.environ.get('REDIS_URL')
    
    # Database (if needed)
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Log to stderr in production
        import logging
        from logging import StreamHandler
        file_handler = StreamHandler()
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    DASH_DEBUG = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}