#!/usr/bin/env python3
"""
Stock Price Trend Prediction System
Main entry point for the ML-based stock prediction system

Features:
- Neural network training and prediction
- Short & long trend analysis (5-day & 25-day moving averages)
- Trend reversal price prediction
- Confidence rating system
- Model update capability
- Data splitting and validation
"""

import os
import sys
import argparse
import logging
import re
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Core modules
from data_manager import DataManager
from model_trainer import ModelTrainer
from predictor import Predictor
from trend_analyzer import TrendAnalyzer
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

def setup_logging(log_level: str) -> None:
    """
    Configure logging for the application
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class StockPredictionSystem:
    """
    Main class for the stock prediction system
    """
    
    def __init__(self, config_path="/app/config.json"):
        """
        Initialize the stock prediction system
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = Config(config_path)
        self.data_manager = DataManager(self.config)
        self.trend_analyzer = TrendAnalyzer(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.predictor = Predictor(self.config)
        self.predictor.set_dependencies(self.trend_analyzer, self.data_manager)
        
        # Get logger after logging is configured
        self.logger = logging.getLogger(__name__)
        self.logger.info("Stock Prediction System initialized")
    
    def load_data(self):
        """
        Load and prepare stock data
        """
        self.logger.info("Loading stock data...")
        
        # Load stock list
        stock_list = self.data_manager.load_stock_list()
        self.logger.info(f"Loaded {len(stock_list)} stocks from stock list")
        
        # Load price data for all stocks (limited to first 50 for testing)
        #Limited data version
        if False:
            limited_stocks = stock_list["Code"].head(100).tolist()
            price_data = self.data_manager.load_price_data(limited_stocks)

        #Full data version
        if True:
            stocks = stock_list["Code"].tolist()
            price_data = self.data_manager.load_price_data(stocks)
        
        return stock_list, price_data
    
    def prepare_training_data(self, price_data):
        """
        Prepare data for model training
        
        Args:
            price_data: Dictionary of stock price data
            
        Returns:
            Tuple of (train_data, test_data)
        """
        self.logger.info("Preparing training data...")
        # src/main.py - prepare_training_data
        self.logger.info("Adding features to price data...")
        featured_data = {}
        for symbol, data in price_data.items():
            if len(data) >= self.config.data.min_data_points:
                featured_data[symbol] = self.trend_analyzer.calculate_indicators(data) 

        self.logger.info("Splitting data for training, validation and testing...")
        train_data, validation_data, test_data = self.data_manager.split_data(
            featured_data,
            train_ratio=self.config.data.train_ratio,
            validation_ratio=self.config.data.validation_ratio
        )
        
        self.logger.info(f"Data split - Train: {len(train_data)}, Val: {len(validation_data)}, Test: {len(test_data)}")
        
        # Return all three sets
        return train_data, validation_data, test_data
    
    def train_models(self, train_data=None):
        """
        Train neural network models
        
        Args:
            train_data: Training data dictionary (optional, will load if not provided)
        """
        self.logger.info("Training models...")
        
        if train_data is None:
            stock_list, price_data = self.load_data()
            train_data, validation_data, test_data = self.prepare_training_data(price_data)
        
        # Train trend classification models
        self.model_trainer.train_short_term_model(train_data, validation_data)
        self.model_trainer.train_long_term_model(train_data, validation_data)
        self.model_trainer.train_reversal_models(train_data, validation_data)
        
        # Save models
        self.model_trainer.save_models()
        
        self.logger.info("Model training completed")
        return {"status": "Training completed"}
    
    def make_predictions(self, stock_symbols=None):
        """
        Generate predictions for specified stocks
        
        Args:
            stock_symbols (list): List of stock symbols to predict (None for all)
            
        Returns:
            dict: Predictions for each stock
        """
        logger.info("Making predictions...")
        logger.info("Loading stock data...")
        
        # Load stock list and price data
        stock_list, price_data = self.load_data()
        logger.info(f"Loaded price data for {len(price_data)} stocks")
        
        # --- FIX: DO NOT load models in the parent process ---
        # This was causing the ProcessPoolExecutor to hang.
        # Models will be loaded by each child process instead.
        #
        # models_loaded = self.predictor.load_models()
        # logger.info(f"Loaded {models_loaded} models from disk" if models_loaded else "No models loaded, using fallback methods")
        
        predictions = {}
        
        self.logger.info("Adding features to price data for prediction...")
        featured_data = {}
        total_stocks = len(price_data) # Get total count
        
        for i, (symbol, data) in enumerate(price_data.items()):
            # --- ADD THIS LINE ---
            self.logger.info(f"[{i+1}/{total_stocks}] Calculating indicators for {symbol}...")
            # --- END ---
            
            featured_data[symbol] = self.trend_analyzer.calculate_indicators(data)

        self.logger.info(f"Generating batch predictions for {len(featured_data)} stocks...")
        # Predict on all stocks at once
        predictions = self.predictor.batch_predict(featured_data)
                
        logger.info(f"Generated predictions for {len(price_data)} stocks")
        return predictions
    def update_models(self):
        """
        Update models with latest data
        """
        self.logger.info("Updating models...")
        
        # Load latest data
        stock_list, price_data = self.load_data()
        train_data, test_data = self.prepare_training_data(price_data)
        
        # Retrain models
        self.train_models(train_data)
        
        # Evaluate performance
        metrics = self.evaluate_models(test_data)
        
        self.logger.info("Models updated successfully")
        return metrics
    
    def evaluate_models(self, test_data):
        """
        Evaluate model performance
        
        Args:
            test_data: Test data dictionary
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating models...")
        
        # Load trained models
        self.predictor.load_models()
        
        # Evaluate on test data
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
        
        # This would contain actual evaluation logic
        # For now, returning placeholder metrics
        
        return metrics
    
    def run_full_pipeline(self):
        """
        Run the complete pipeline: load data, train models, make predictions
        """
        self.logger.info("Running full pipeline...")
        
        try:
            # 1. Load data
            stock_list, price_data = self.load_data()
            
            # 2. Prepare training data
            train_data, validation_data, test_data = self.prepare_training_data(price_data)
            
            # 3. Train models
            self.train_models(train_data)
            
            # 4. Evaluate models
            metrics = self.evaluate_models(test_data)
            
            # 5. Generate sample predictions
            sample_stocks = list(price_data.keys())[:10]  # Predict for first 10 stocks
            predictions = self.make_predictions(sample_stocks)
            
            self.logger.info("Full pipeline completed successfully")
            return {
                'metrics': metrics,
                'sample_predictions': predictions
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Stock Prediction System - Predict trends and reversal prices for stocks'
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'predict', 'update', 'full', 'test'],
        default='predict',
        help='Operation mode (default: predict)'
    )
    
    parser.add_argument(
        '--config',
        default='/app/config.json',
        help='Path to configuration file (default: config.json)'
    )
    
    parser.add_argument(
        '--output',
        default='predictions.json',
        help='Output filename for predictions (default: predictions.json)'
    )
    
    parser.add_argument(
        '--stocks',
        nargs='+',
        help='Specific stock symbols to predict (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()

def ensure_output_directory(config: Config, output_filename: str) -> str:
    """
    Ensure output directory exists and return full output path
    
    Args:
        config (Config): System configuration
        output_filename (str): Name of the output file
        
    Returns:
        str: Full path to output file
    """
    # Get configured output directory
    output_dir = config.system.output_path
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine directory and filename
    full_output_path = os.path.join(output_dir, output_filename)
    
    return full_output_path

def save_predictions_as_csv(predictions: dict[str, dict[str, any]], output_path: str) -> str:
    """
    Save stock predictions as CSV instead of JSON format
    
    Args:
        predictions (Dict[str, Dict[str, Any]]): Prediction results dictionary
        output_path (str): Path where to save the CSV file
        
    Returns:
        str: Full path to the saved CSV file
    """
    # Convert predictions dictionary to a list of records for DataFrame
    prediction_records = []
    
    for stock_code, prediction in predictions.items():
        if 'error' in prediction:
            # Handle error cases
            record = {
                'stock_code': stock_code,
                'error': prediction['error'],
                'timestamp': prediction.get('timestamp', datetime.now(JST).isoformat()),
                'short_trend': None,
                'long_trend': None,
                'short_reversal_price': None,
                'long_reversal_price': None,
                'short_bailout_point': None,
                'long_bailout_point': None,
                'short_confidence': None,
                'long_confidence': None,
                'current_price': None,
                'analysis_date': None
            }
        else:
            # Handle successful predictions
            record = {
                'stock_code': stock_code,
                'error': None,
                'timestamp': prediction.get('timestamp', datetime.now(JST).isoformat()),
                'short_trend': prediction.get('short_trend'),
                'long_trend': prediction.get('long_trend'),
                'short_reversal_price': prediction.get('short_reversal_price'),
                'long_reversal_price': prediction.get('long_reversal_price'),
                'short_bailout_point': prediction.get('short_bailout_point'),
                'long_bailout_point': prediction.get('long_bailout_point'),
                'short_confidence': prediction.get('short_confidence'),
                'long_confidence': prediction.get('long_confidence'),
                'current_price': prediction.get('current_price'),
                'analysis_date': prediction.get('analysis_date')
            }
        
        prediction_records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(prediction_records)
    
    # Sort by stock_code for consistent output
    df = df.sort_values('stock_code')
    
    # Ensure output path has .csv extension
    if not output_path.endswith('.csv'):
        output_path = output_path.replace('.json', '.csv')
        if not output_path.endswith('.csv'):
            output_path += '.csv'
    
    # Save to CSV
    df.to_csv("/app/output/predictions.csv", index=False, encoding='utf-8')
    
    return output_path

def main():
    """Main entry point"""
    args = parse_arguments()
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # Setup logging
        setup_logging(config.system.log_level)
        
        # Get logger after setup
        logger = logging.getLogger(__name__)
        
        # Initialize system
        system = StockPredictionSystem(args.config)
        
        logger.info(f"Stock Prediction System started in {args.mode} mode")
        
        if args.mode == 'train':
            logger.info("Training mode selected")
            metrics = system.train_models()
            print("Training completed. Metrics:", metrics)
            
        elif args.mode == 'predict':
            logger.info("Prediction mode selected")
            predictions = system.make_predictions(args.stocks)
            
            # Ensure output directory exists and get full path
            full_output_path = ensure_output_directory(config, args.output)
            
            # Save predictions as CSV instead of JSON
            try:
                final_output_path = save_predictions_as_csv(predictions, full_output_path)
                print(f"Predictions saved to {final_output_path}")
            except Exception as e:
                logger.error(f"Error saving predictions to CSV: {e}")
                # Fallback to JSON if CSV fails
                fallback_path = full_output_path.replace('.csv', '.json') if '.csv' in full_output_path else full_output_path
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    json.dump(predictions, f, indent=2, ensure_ascii=False)
                print(f"Predictions saved to {fallback_path} (JSON fallback)")
            
            # Display sample results
            print(f"\nSample predictions (showing first 5):")
            for symbol, pred in list(predictions.items())[:5]:
                if 'error' not in pred:
                    print(f"\n{symbol}:")
                    print(f"  Current Price: ${pred['current_price']:.2f}")
                    print(f"  Short trend: {pred['short_trend']}")
                    print(f"  Long trend: {pred['long_trend']}")
                    print(f"  Short reversal: ${pred['short_reversal_price']:.2f}")
                    print(f"  Long reversal: ${pred['long_reversal_price']:.2f}")
                    if pred.get('short_bailout_point') is not None:
                        print(f"  Short Bailout: ${pred['short_bailout_point']:.2f}")
                    if pred.get('long_bailout_point') is not None:
                        print(f"  Long Bailout: ${pred['long_bailout_point']:.2f}")
                    print(f"  Confidence: {pred['short_confidence']:.2f}% / {pred['long_confidence']:.2f}%")
                else:
                    print(f"\n{symbol}: {pred['error']}")
            
            total_predictions = len(predictions)
            successful_predictions = len([p for p in predictions.values() if 'error' not in p])
            print(f"\nTotal predictions: {total_predictions}")
            print(f"Successful predictions: {successful_predictions}")
            print(f"Success rate: {successful_predictions/total_predictions*100:.1f}%")
            
        elif args.mode == 'update':
            logger.info("Update mode selected")
            metrics = system.update_models()
            print("Models updated. New metrics:", metrics)
            
        elif args.mode == 'full':
            logger.info("Full pipeline mode selected")
            results = system.run_full_pipeline()
            print("Full pipeline completed:", results['metrics'])
            
        elif args.mode == 'test':
            logger.info("Test mode selected")
            # Test system components
            from src.predictor import test_predictor
            test_predictor()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"System error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()