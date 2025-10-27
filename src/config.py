#!/usr/bin/env python3
"""
Configuration Management for Stock Prediction System

This module handles all configuration settings for the stock prediction system,
including data paths, model parameters, and system settings.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Data-related configuration settings"""
    stock_list_path: str = "data/stock_list.csv"
    price_data_path: str = "data/price_data/"
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    min_data_points: int = 100
    max_missing_days: int = 10
    date_format: str = "%Y-%m-%d"
    
    def __post_init__(self):
        """Validate data configuration"""
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1.0) > 0.01:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")

@dataclass
class ModelConfig:
    """Neural network model configuration"""
    short_term_window: int = 5
    long_term_window: int = 25
    lookback_period: int = 60
    prediction_horizon: int = 1
    
    # Neural network architecture
    hidden_layers: List[int] = None
    dropout_rate: float = 0.2
    activation: str = "relu"
    output_activation: str = "linear"
    
    # Training parameters
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    
    # Optimization
    optimizer: str = "adam"
    loss_function: str = "mse"
    metrics: List[str] = None
    
    def __post_init__(self):
        """Set default values for list fields"""
        if self.hidden_layers is None:
            self.hidden_layers = [64, 32, 16]
        if self.metrics is None:
            self.metrics = ["mae", "mse"]

@dataclass
class PredictionConfig:
    """Prediction and confidence calculation settings"""
    confidence_threshold: float = 0.7
    update_frequency: str = "daily"
    ensemble_size: int = 5
    monte_carlo_samples: int = 1000
    bailout_volatility_multiplier: float = 2.0
    bailout_return_multiplier: float = 3.0
    bailout_fixed_pct: float = 0.08
    
    # Trend detection settings
    trend_threshold: float = 0.001
    reversal_sensitivity: float = 0.02
    volatility_window: int = 20
    
    # Output settings
    decimal_places: int = 2
    include_intermediate_values: bool = False
    save_confidence_details: bool = True

@dataclass
class SystemConfig:
    """System-level configuration settings"""
    log_level: str = "INFO"
    log_file: str = "logs/system.log"
    model_save_path: str = "models/"
    output_path: str = "output/"
    
    # Performance settings
    n_jobs: int = -1  # -1 uses all available cores
    memory_limit_gb: float = 8.0
    batch_processing_size: int = 100
    
    # Update settings
    auto_update_enabled: bool = False
    update_schedule: str = "0 2 * * *"  # Daily at 2 AM (cron format)
    backup_models: bool = True
    max_model_versions: int = 5

class Config:
    """
    Main configuration class for the Stock Prediction System
    
    This class manages all configuration settings and provides methods to load,
    save, and validate configuration data.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.config_path = config_path or "config.json"
        
        # Initialize configuration sections
        self.data = DataConfig()
        self.model = ModelConfig()
        self.prediction = PredictionConfig()
        self.system = SystemConfig()
        
        # Load configuration if file exists
        if os.path.exists(self.config_path):
            self.load_config()
        else:
            logger.warning(f"Configuration file {self.config_path} not found. Using defaults.")
            self.save_config()
    
    def load_config(self) -> None:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update configuration sections
            if 'data' in config_data:
                self.data = DataConfig(**config_data['data'])
            
            if 'model' in config_data:
                self.model = ModelConfig(**config_data['model'])
            
            if 'prediction' in config_data:
                self.prediction = PredictionConfig(**config_data['prediction'])
            
            if 'system' in config_data:
                self.system = SystemConfig(**config_data['system'])
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def save_config(self) -> None:
        """Save current configuration to JSON file"""
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            config_data = {
                'data': asdict(self.data),
                'model': asdict(self.model),
                'prediction': asdict(self.prediction),
                'system': asdict(self.system)
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def validate_config(self) -> bool:
        """
        Validate configuration settings
        
        Returns:
            bool: True if configuration is valid
        """
        try:
            # Validate data paths
            self._validate_paths()
            
            # Validate model parameters
            self._validate_model_params()
            
            # Validate prediction settings
            self._validate_prediction_params()
            
            # Validate system settings
            self._validate_system_params()
            
            logger.info("Configuration validation passed")
            return True
            
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _validate_paths(self) -> None:
        """Validate data paths and create directories if needed"""
        # Create necessary directories
        directories = [
            os.path.dirname(self.data.stock_list_path),
            self.data.price_data_path,
            self.system.model_save_path,
            self.system.output_path,
            os.path.dirname(self.system.log_file)
        ]
        
        for directory in directories:
            if directory:  # Skip empty strings
                os.makedirs(directory, exist_ok=True)
    
    def _validate_model_params(self) -> None:
        """Validate model parameters"""
        if self.model.short_term_window <= 0:
            raise ValueError("Short term window must be positive")
        
        if self.model.long_term_window <= self.model.short_term_window:
            raise ValueError("Long term window must be greater than short term window")
        
        if self.model.lookback_period <= self.model.long_term_window:
            raise ValueError("Lookback period must be greater than long term window")
        
        if not 0 < self.model.dropout_rate < 1:
            raise ValueError("Dropout rate must be between 0 and 1")
        
        if self.model.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
    
    def _validate_prediction_params(self) -> None:
        """Validate prediction parameters"""
        if not 0 <= self.prediction.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if self.prediction.ensemble_size <= 0:
            raise ValueError("Ensemble size must be positive")
        
        if self.prediction.monte_carlo_samples <= 0:
            raise ValueError("Monte Carlo samples must be positive")
    
    def _validate_system_params(self) -> None:
        """Validate system parameters"""
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.system.log_level not in valid_log_levels:
            raise ValueError(f"Log level must be one of {valid_log_levels}")
        
        if self.system.memory_limit_gb <= 0:
            raise ValueError("Memory limit must be positive")
        
        if self.system.batch_processing_size <= 0:
            raise ValueError("Batch processing size must be positive")
    
    def get_model_path(self, model_name: str) -> str:
        """
        Get the full path for a model file
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            str: Full path to model file
        """
        return os.path.join(self.system.model_save_path, f"{model_name}.h5")
    
    def get_output_path(self, filename: str) -> str:
        """
        Get the full path for an output file
        
        Args:
            filename (str): Name of the output file
            
        Returns:
            str: Full path to output file
        """
        return os.path.join(self.system.output_path, filename)
    
    def update_config(self, section: str, **kwargs) -> None:
        """
        Update configuration section with new values
        
        Args:
            section (str): Configuration section ('data', 'model', 'prediction', 'system')
            **kwargs: Key-value pairs to update
        """
        if section == 'data':
            for key, value in kwargs.items():
                if hasattr(self.data, key):
                    setattr(self.data, key, value)
        elif section == 'model':
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        elif section == 'prediction':
            for key, value in kwargs.items():
                if hasattr(self.prediction, key):
                    setattr(self.prediction, key, value)
        elif section == 'system':
            for key, value in kwargs.items():
                if hasattr(self.system, key):
                    setattr(self.system, key, value)
        else:
            raise ValueError(f"Unknown configuration section: {section}")
        
        # Validate and save updated configuration
        if self.validate_config():
            self.save_config()
            logger.info(f"Configuration section '{section}' updated")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values"""
        self.data = DataConfig()
        self.model = ModelConfig()
        self.prediction = PredictionConfig()
        self.system = SystemConfig()
        
        self.save_config()
        logger.info("Configuration reset to defaults")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            dict: Configuration as dictionary
        """
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'prediction': asdict(self.prediction),
            'system': asdict(self.system)
        }
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def __repr__(self) -> str:
        """Detailed representation of configuration"""
        return f"Config(config_path='{self.config_path}')"

def create_default_config(config_path: str = "config.json") -> Config:
    """
    Create a default configuration file
    
    Args:
        config_path (str): Path where to save the configuration
        
    Returns:
        Config: Initialized configuration object
    """
    config = Config(config_path)
    config.save_config()
    return config

# Example usage and testing
if __name__ == "__main__":
    # Create default configuration
    config = create_default_config("example_config.json")
    
    # Print configuration
    print("Default Configuration:")
    print(config)
    
    # Update some settings
    config.update_config('model', learning_rate=0.01, epochs=200)
    config.update_config('data', train_ratio=0.75, validation_ratio=0.15)
    
    # Validate configuration
    is_valid = config.validate_config()
    print(f"\nConfiguration valid: {is_valid}")
    
    # Print updated configuration
    print("\nUpdated Configuration:")
    print(json.dumps(config.to_dict(), indent=2))