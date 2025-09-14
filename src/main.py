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
import logging
import pandas as pd
from datetime import datetime
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

class StockPredictionSystem:
    """
    Main class for the stock prediction system
    """
    
    def __init__(self, config_path="config.json"):
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
        
        logger.info("Stock Prediction System initialized")
    
    def load_data(self):
        """
        Load and prepare stock data
        """
        logger.info("Loading stock data...")
        
        # Load stock list
        stock_list = self.data_manager.load_stock_list()
        logger.info(f"Loaded {len(stock_list)} stocks from stock list")
        
        # Load price data for all stocks
        Limited_stocks = stock_list["Code"].head(50).tolist()
        price_data = self.data_manager.load_price_data(Limited_stocks)
        logger.info(f"Loaded price data for {len(price_data)} stocks")
        
        return stock_list, price_data
    
    def prepare_training_data(self, price_data):
        """
        Prepare data for model training
        
        Args:
            price_data (dict): Stock price data
            
        Returns:
            tuple: Training and testing datasets
        """
        logger.info("Preparing training data...")
        
        # Calculate technical indicators
        processed_data = self.trend_analyzer.calculate_moving_averages(price_data)
        processed_data = self.trend_analyzer.calculate_trend_features(processed_data)
        
        # Split data for training and testing
        train_data, validation_data, test_data = self.data_manager.split_data(
            processed_data, 
            train_ratio=self.config.data.train_ratio
        )
        
        logger.info(f"Training data: {len(train_data)} stocks")
        logger.info(f"Validation data: {len(validation_data)} stocks")
        logger.info(f"Testing data: {len(test_data)} stocks")
        
        return train_data, test_data
    
    def train_models(self, train_data):
        """
        Train the neural network models
        
        Args:
            train_data (dict): Training dataset
        """
        logger.info("Training neural network models...")
        
        # Train short-term trend model
        self.model_trainer.train_short_term_model(train_data)
        logger.info("Short-term trend model trained")
        
        # Train long-term trend model
        self.model_trainer.train_long_term_model(train_data)
        logger.info("Long-term trend model trained")
        
        # Train reversal price prediction models
        self.model_trainer.train_reversal_models(train_data)
        logger.info("Reversal price prediction models trained")
        
        # Save trained models
        self.model_trainer.save_models()
        logger.info("All models saved successfully")
    
    def evaluate_models(self, test_data):
        """
        Evaluate model performance on test data
        
        Args:
            test_data (dict): Testing dataset
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        metrics = self.model_trainer.evaluate_models(test_data)
        
        logger.info("Model evaluation completed:")
        for model_name, model_metrics in metrics.items():
            logger.info(f"{model_name}: {model_metrics}")
        
        return metrics
    
    def make_predictions(self, stock_symbols=None):
        """
        Generate predictions for specified stocks
        
        Args:
            stock_symbols (list): List of stock symbols to predict (None for all)
            
        Returns:
            dict: Predictions for each stock
        """
        logger.info("Generating predictions...")
        
        # If no specific stocks provided, load all stocks from stock list
        if stock_symbols is None:
            try:
                # Load stock list if not already loaded
                if self.data_manager.stock_list is None:
                    stock_list = self.data_manager.load_stock_list()
                    logger.info(f"Loaded stock list with {len(stock_list)} stocks")
                
                # Get stock codes from the stock list
                if hasattr(self.data_manager.stock_list, 'Code'):
                    stock_symbols = self.data_manager.stock_list['Code'].tolist()
                elif hasattr(self.data_manager.stock_list, 'code'):
                    stock_symbols = self.data_manager.stock_list['code'].tolist()
                else:
                    # Fallback: try to discover available stock data files
                    stock_symbols = self.data_manager._discover_available_stocks()
                    
                logger.info(f"Will generate predictions for {len(stock_symbols)} stocks")
                
            except Exception as e:
                logger.error(f"Error loading stock list: {e}")
                return {}
        
        # Load latest data for the specified stocks
        current_data = self.data_manager.get_latest_data(stock_symbols)
        logger.info(f"Loaded data for {len(current_data)} stocks")
        
        if not current_data:
            logger.warning("No stock data loaded - predictions cannot be generated")
            return {}
        
        predictions = {}
        
        for symbol, data in current_data.items():
            try:
                # Calculate current trends
                short_trend = self.trend_analyzer.calculate_short_trend(data)
                long_trend = self.trend_analyzer.calculate_long_trend(data)
                
                # Predict trend reversals
                short_reversal = self.predictor.predict_short_reversal(data)
                long_reversal = self.predictor.predict_long_reversal(data)
                
                # Calculate confidence ratings
                short_confidence = self.predictor.calculate_confidence(data, 'short')
                long_confidence = self.predictor.calculate_confidence(data, 'long')
                
                predictions[symbol] = {
                    'short_trend': short_trend,
                    'long_trend': long_trend,
                    'short_reversal_price': short_reversal['price'],
                    'long_reversal_price': long_reversal['price'],
                    'short_confidence': short_confidence,
                    'long_confidence': long_confidence,
                    'timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error predicting for {symbol}: {e}")
                predictions[symbol] = {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        logger.info(f"Generated predictions for {len(predictions)} stocks")
        return predictions
    
    def update_models(self):
        """
        Update the prediction models with new data
        """
        logger.info("Updating models with latest data...")
        
        # Load latest data
        stock_list, price_data = self.load_data()
        
        # Prepare training data
        train_data, test_data = self.prepare_training_data(price_data)
        
        # Retrain models
        self.train_models(train_data)
        
        # Evaluate updated models
        metrics = self.evaluate_models(test_data)
        
        logger.info("Model update completed")
        return metrics
    
    def run_full_pipeline(self):
        """
        Run the complete training and prediction pipeline
        """
        logger.info("Starting full pipeline...")
        
        try:
            # 1. Load data
            stock_list, price_data = self.load_data()
            
            # 2. Prepare training data
            train_data, test_data = self.prepare_training_data(price_data)
            
            # 3. Train models
            self.train_models(train_data)
            
            # 4. Evaluate models
            metrics = self.evaluate_models(test_data)
            
            # 5. Generate sample predictions
            sample_stocks = list(price_data.keys())[:10]  # Predict for first 10 stocks
            predictions = self.make_predictions(sample_stocks)
            
            logger.info("Full pipeline completed successfully")
            return {
                'metrics': metrics,
                'sample_predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

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
    
    def __init__(self, config_path="config.json"):
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
        limited_stocks = stock_list["Code"].head(50).tolist()
        price_data = self.data_manager.load_price_data(limited_stocks)
        
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
        
        # Add technical indicators and features
        processed_data = {}
        for symbol, data in price_data.items():
            if len(data) >= self.config.data.min_data_points:
                processed_data[symbol] = self.trend_analyzer.calculate_indicators(data)
        
        # Split data for training and testing
        train_data = {}
        test_data = {}
        
        for symbol, data in processed_data.items():
            split_index = int(len(data) * self.config.data.train_ratio)
            train_data[symbol] = data[:split_index].copy()
            test_data[symbol] = data[split_index:].copy()
        
        self.logger.info(f"Training data prepared for {len(train_data)} stocks")
        return train_data, test_data
    
    def train_models(self, train_data=None):
        """
        Train neural network models
        
        Args:
            train_data: Training data dictionary (optional, will load if not provided)
        """
        self.logger.info("Training models...")
        
        if train_data is None:
            stock_list, price_data = self.load_data()
            train_data, _ = self.prepare_training_data(price_data)
        
        # Train trend classification models
        self.model_trainer.train_trend_models(train_data)
        
        # Train reversal prediction models
        self.model_trainer.train_reversal_models(train_data)
        
        # Save models
        self.model_trainer.save_all_models()
        
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
        
        # Load trained models into predictor
        models_loaded = self.predictor.load_models(self.model_trainer)
        logger.info(f"Loaded {models_loaded} models from disk" if models_loaded else "No models loaded, using fallback methods")
        
        predictions = {}
        
        for stock_code, stock_data in price_data.items():
            try:
                # Process data through the analysis pipeline
                # First calculate moving averages
                processed_data = {stock_code: stock_data}
                ma_data = self.trend_analyzer.calculate_moving_averages(processed_data)
                
                # Then calculate all trend features
                trend_data = self.trend_analyzer.calculate_trend_features(ma_data)
                
                # Get the processed data for this stock
                analyzed_data = trend_data[stock_code]
                
                # Generate complete prediction using the predictor
                prediction = self.predictor.predict_stock(analyzed_data, stock_code)
                
                # Store the prediction
                predictions[stock_code] = prediction
                
            except Exception as e:
                logger.error(f"Error predicting {stock_code}: {e}")
                predictions[stock_code] = str(e)
        
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
            train_data, test_data = self.prepare_training_data(price_data)
            
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
        default='config.json',
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
                'timestamp': prediction.get('timestamp', datetime.now().isoformat()),
                'short_trend': None,
                'long_trend': None,
                'short_reversal_price': None,
                'long_reversal_price': None,
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
                'timestamp': prediction.get('timestamp', datetime.now().isoformat()),
                'short_trend': prediction.get('short_trend'),
                'long_trend': prediction.get('long_trend'),
                'short_reversal_price': prediction.get('short_reversal_price'),
                'long_reversal_price': prediction.get('long_reversal_price'),
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
    df.to_csv(output_path, index=False, encoding='utf-8')
    
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
                    print(f"  Short trend: {pred['short_trend']}")
                    print(f"  Long trend: {pred['long_trend']}")
                    print(f"  Short reversal: {pred['short_reversal_price']:.2f}")
                    print(f"  Long reversal: {pred['long_reversal_price']:.2f}")
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