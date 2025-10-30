#!/usr/bin/env python3
"""
Prediction Engine for Stock Prediction System

This module handles real-time predictions using trained neural network models
to generate trend forecasts, reversal price predictions, and confidence ratings
according to the system specifications.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timezone, timedelta
import json
from scipy import stats
from scipy.signal import find_peaks
import warnings
import os  # This was missing and causing the "name 'os' is not defined" error
import joblib  # For model serialization if not using TensorFlow
import concurrent.futures
import pandas as pd
from typing import Dict, Any

# TensorFlow/Keras imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. Using fallback prediction methods.")
    TENSORFLOW_AVAILABLE = False
    # Create dummy classes to prevent import errors
    class keras:
        @staticmethod
        class models:
            @staticmethod
            def load_model(path):
                return None

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

class Predictor:
    """
    Prediction Engine for stock trend and reversal price forecasting
    
    Uses trained neural network models to generate:
    - Short-term and long-term trend predictions
    - Trend reversal price predictions
    - Confidence ratings for all predictions
    """
    
    def __init__(self, config):
        """
        Initialize Predictor
        
        Args:
            config: Configuration object containing prediction settings
        """
        self.config = config
        self.models = {}
        self.trend_analyzer = None
        self.data_manager = None
        
        # Prediction parameters
        self.short_window = config.model.short_term_window
        self.long_window = config.model.long_term_window
        self.lookback_period = config.model.lookback_period
        self.ensemble_size = config.prediction.ensemble_size
        self.monte_carlo_samples = config.prediction.monte_carlo_samples
        
        logger.info("Predictor initialized")
    
    def set_dependencies(self, trend_analyzer, data_manager):
        """
        Set dependencies for trend analysis and data management
        
        Args:
            trend_analyzer: TrendAnalyzer instance
            data_manager: DataManager instance
        """
        self.trend_analyzer = trend_analyzer
        self.data_manager = data_manager
    
    def load_models(self) -> bool:
        """
        Load trained models for prediction
        
        Args:
            model_trainer: ModelTrainer instance (optional)
            
        Returns:
            bool: True if models loaded successfully
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available for prediction")
            return False
        
        try:
            
            # Load from disk
            model_names = [
                'short_trend_classifier',
                'long_trend_classifier', 
                'short_reversal_regressor',
                'long_reversal_regressor',
                'confidence_estimator'
            ]
            
            loaded_count = 0
            for model_name in model_names:
                model_path = self.config.get_model_path(model_name)
                try:
                    if os.path.exists(model_path):
                        self.models[model_name] = keras.models.load_model(model_path, compile=False)
                        loaded_count += 1
                        logger.info(f"Loaded {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
            
            logger.info(f"Loaded {loaded_count} models from disk")
            return loaded_count > 0
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def prepare_prediction_data(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Prepare data for neural network prediction
        
        Args:
            data (pd.DataFrame): Stock price data with features
            
        Returns:
            np.ndarray: Prepared data for model input
        """
        try:
            # Check if we have enough data
            if len(data) < self.lookback_period:
                logger.warning(f"Insufficient data for prediction: {len(data)} < {self.lookback_period}")
                return None
            
            # Feature columns expected by the model
            base_features = [
                'open', 'high', 'low', 'close', 'volume',
                'return_1d', 'volatility_5d', 'volatility_20d',
                'rsi', 'macd', 'macd_signal',
                'short_trend_slope', 'long_trend_slope',
                'short_trend_strength', 'long_trend_strength',
                'trend_agreement', 'ma_convergence'
            ]
            
            # Filter features that exist in the data
            available_features = [col for col in base_features if col in data.columns]
            
            if not available_features:
                logger.warning("No features available for prediction")
                return None
            
            # Extract features and handle missing values
            feature_data = data[available_features].ffill().fillna(0)
            
            # Get the most recent sequence
            if len(feature_data) >= self.lookback_period:
                sequence = feature_data.tail(self.lookback_period).values
                # Reshape for model input: (1, timesteps, features)
                return sequence.reshape(1, sequence.shape[0], sequence.shape[1])
            
            return None
            
        except Exception as e:
            logger.error(f"Error preparing prediction data: {e}")
            return None
    
    def predict_trend(self, data: pd.DataFrame, trend_type: str = 'short') -> Dict[str, Any]:
        """
        Predict trend direction using neural network models
        
        Args:
            data (pd.DataFrame): Stock price data
            trend_type (str): 'short' or 'long' trend prediction
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        try:
            model_name = f'{trend_type}_trend_classifier'
            
            if model_name not in self.models:
                # Fallback to traditional trend analysis
                logger.warning(f"Model {model_name} not available, using fallback method")
                return self._fallback_trend_prediction(data, trend_type)
            
            # Prepare input data
            input_data = self.prepare_prediction_data(data)
            if input_data is None:
                return self._fallback_trend_prediction(data, trend_type)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(input_data, verbose=0)
            
            # Convert prediction to trend direction
            predicted_class = np.argmax(prediction[0])  # 0, 1, or 2
            trend_direction = predicted_class - 1  # Convert to -1, 0, 1
            
            # Get prediction probabilities
            probabilities = prediction[0]
            confidence = float(np.max(probabilities))
            
            # Convert to +/- format
            trend_str = '+' if trend_direction == 1 else '-' if trend_direction == -1 else '0'
            
            return {
                'trend': trend_str,
                'trend_numeric': int(trend_direction),
                'confidence': confidence,
                'probabilities': {
                    'down': float(probabilities[0]),
                    'sideways': float(probabilities[1]),
                    'up': float(probabilities[2])
                },
                'method': 'neural_network'
            }
            
        except Exception as e:
            logger.error(f"Error in trend prediction: {e}")
            return self._fallback_trend_prediction(data, trend_type)
    
    def _fallback_trend_prediction(self, data: pd.DataFrame, trend_type: str) -> Dict[str, Any]:
        """
        Fallback trend prediction using traditional analysis
        
        Args:
            data (pd.DataFrame): Stock price data
            trend_type (str): 'short' or 'long'
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        try:
            if self.trend_analyzer is None:
                # Basic trend calculation
                window = self.short_window if trend_type == 'short' else self.long_window
                if len(data) < window:
                    return {'trend': '0', 'confidence': 0.3, 'method': 'insufficient_data'}
                
                ma = data['close'].rolling(window=window).mean()
                recent_ma = ma.dropna().tail(3)
                
                if len(recent_ma) < 2:
                    return {'trend': '0', 'confidence': 0.3, 'method': 'insufficient_ma'}
                
                slope = (recent_ma.iloc[-1] - recent_ma.iloc[0]) / len(recent_ma)
                threshold = self.config.prediction.trend_threshold
                
                if slope > threshold:
                    trend = '+'
                elif slope < -threshold:
                    trend = '-'
                else:
                    trend = '0'
                
                confidence = min(abs(slope) / threshold, 1.0) * 0.7  # Max 70% for fallback
                
            else:
                # Use trend analyzer
                if trend_type == 'short':
                    trend = self.trend_analyzer.calculate_short_trend(data)
                else:
                    trend = self.trend_analyzer.calculate_long_trend(data)
                
                confidence = 0.6  # Moderate confidence for traditional analysis
            
            return {
                'trend': trend,
                'confidence': confidence,
                'method': 'fallback_analysis'
            }
            
        except Exception as e:
            logger.error(f"Error in fallback trend prediction: {e}")
            return {'trend': '0', 'confidence': 0.0, 'method': 'error'}
    
    def predict_reversal_price(self, data: pd.DataFrame, trend_type: str = 'short') -> Dict[str, Any]:
        """
        Predict trend reversal price using neural network models
        
        Args:
            data (pd.DataFrame): Stock price data
            trend_type (str): 'short' or 'long' reversal prediction
            
        Returns:
            Dict[str, Any]: Reversal price prediction
        """
        try:
            model_name = f'{trend_type}_reversal_regressor'
            
            if model_name not in self.models:
                # Fallback to traditional reversal analysis
                logger.warning(f"Model {model_name} not available, using fallback method")
                return self._fallback_reversal_prediction(data, trend_type)
            
            # Prepare input data
            input_data = self.prepare_prediction_data(data)
            if input_data is None:
                return self._fallback_reversal_prediction(data, trend_type)
            
            # Make prediction (returns percentage change)
            model = self.models[model_name]
            prediction = model.predict(input_data, verbose=0)
            
            predicted_change = float(prediction[0][0])
            current_price = float(data['close'].iloc[-1])
            
            # Calculate reversal price
            reversal_price = current_price * (1 + predicted_change)
            
            # Ensure price is positive and reasonable
            reversal_price = max(reversal_price, current_price * 0.5)
            reversal_price = min(reversal_price, current_price * 2.0)
            
            # Estimate confidence using prediction uncertainty
            confidence = self._estimate_reversal_confidence(data, predicted_change, trend_type)
            
            return {
                'price': reversal_price,
                'current_price': current_price,
                'predicted_change_pct': predicted_change * 100,
                'confidence': confidence,
                'method': 'neural_network'
            }
            
        except Exception as e:
            logger.error(f"Error in reversal price prediction: {e}")
            return self._fallback_reversal_prediction(data, trend_type)
    
    def _fallback_reversal_prediction(self, data: pd.DataFrame, trend_type: str) -> Dict[str, Any]:
        """
        Fallback reversal prediction using traditional analysis
        
        Args:
            data (pd.DataFrame): Stock price data
            trend_type (str): 'short' or 'long'
            
        Returns:
            Dict[str, Any]: Reversal price prediction
        """
        try:
            current_price = float(data['close'].iloc[-1])
            
            if self.trend_analyzer is None:
                # Simple statistical approach
                window = self.short_window if trend_type == 'short' else self.long_window
                returns = data['close'].pct_change().dropna()
                
                if len(returns) < window:
                    return {
                        'price': current_price,
                        'confidence': 0.3,
                        'method': 'insufficient_data'
                    }
                
                # Use recent volatility to estimate potential reversal
                recent_vol = returns.tail(window * 2).std()
                reversal_price = current_price * (1 + recent_vol * np.random.choice([-1, 1]))
                confidence = 0.5
                
            else:
                # Use trend analyzer
                reversal_info = self.trend_analyzer.predict_next_reversal_price(data, trend_type)
                reversal_price = reversal_info.get('price', current_price)
                confidence = reversal_info.get('confidence', 0.5)
            
            return {
                'price': reversal_price,
                'current_price': current_price,
                'confidence': confidence,
                'method': 'fallback_analysis'
            }
            
        except Exception as e:
            logger.error(f"Error in fallback reversal prediction: {e}")
            return {
                'price': data['close'].iloc[-1] if len(data) > 0 else 100.0,
                'confidence': 0.0,
                'method': 'error'
            }
    
    def _estimate_reversal_confidence(self, data: pd.DataFrame, predicted_change: float, trend_type: str) -> float:
        """
        Estimate confidence in reversal price prediction
        
        Args:
            data (pd.DataFrame): Stock price data
            predicted_change (float): Predicted price change
            trend_type (str): 'short' or 'long'
            
        Returns:
            float: Confidence score (0-1)
        """
        try:
            # Use confidence estimator model if available
            if 'confidence_estimator' in self.models:
                input_data = self.prepare_prediction_data(data)
                if input_data is not None:
                    confidence_pred = self.models['confidence_estimator'].predict(input_data, verbose=0)
                    base_confidence = float(confidence_pred[0][0])
                else:
                    base_confidence = 0.5
            else:
                base_confidence = 0.5
            
            # Adjust confidence based on prediction reasonableness
            if abs(predicted_change) > 0.5:  # Very large change (>50%)
                base_confidence *= 0.7
            elif abs(predicted_change) < 0.01:  # Very small change (<1%)
                base_confidence *= 0.8
            
            # Consider recent volatility
            recent_returns = data['close'].pct_change().tail(20).std()
            if recent_returns > 0.05:  # High volatility
                base_confidence *= 0.9
            elif recent_returns < 0.01:  # Low volatility
                base_confidence *= 1.1
            
            return np.clip(base_confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error estimating reversal confidence: {e}")
            return 0.5
    
    def _calculate_bailout_point(self, data: pd.DataFrame, current_price: float, trend_direction: str) -> Optional[float]:
        """
        Calculate the bailout point based on recent volatility.
        This is a stop-loss or invalidation point for the trend prediction.
        
        Args:
            data (pd.DataFrame): Stock price data with features
            current_price (float): The current closing price
            trend_direction (str): The predicted trend ('+', '-', '0')
            
        Returns:
            Optional[float]: The calculated bailout price
        """
        try:
            # Get multipliers from config, with defaults
            vol_multiplier = self.config.prediction.bailout_volatility_multiplier
            ret_multiplier = self.config.prediction.bailout_return_multiplier
            fixed_pct = self.config.prediction.bailout_fixed_pct
            
            bailout_offset = 0.0
            
            # 1. Try to use 'price_volatility' (rolling std dev of price from trend_analyzer)
            if 'price_volatility' in data.columns and not pd.isna(data['price_volatility'].iloc[-1]):
                bailout_offset = data['price_volatility'].iloc[-1] * vol_multiplier
            
            # 2. Fallback to 'volatility_20d' (std dev of returns)
            elif 'volatility_20d' in data.columns and not pd.isna(data['volatility_20d'].iloc[-1]):
                volatility_pct = data['volatility_20d'].iloc[-1]
                bailout_offset = current_price * (volatility_pct * ret_multiplier)
            
            # 3. Last resort: a fixed percentage
            else:
                bailout_offset = current_price * fixed_pct

            if pd.isna(bailout_offset) or bailout_offset <= 0.0:
                 bailout_offset = current_price * fixed_pct

            bailout_price = None
            if trend_direction == '+':
                # For an UPTREND, the bailout is a price DROP
                bailout_price = current_price - bailout_offset
            elif trend_direction == '-':
                # For a DOWNTREND, the bailout is a price RISE
                bailout_price = current_price + bailout_offset
            else:
                # For a SIDEWAYS trend, a bailout point isn't as clear.
                return None # Return None for '0' trend

            # Ensure bailout price is positive
            return max(0.0, bailout_price) if bailout_price is not None else None
        
        except Exception as e:
            logger.warning(f"Error calculating bailout point: {e}")
            # Fallback to fixed percentage on error
            try:
                fixed_pct = self.config.prediction.bailout_fixed_pct
                if trend_direction == '+':
                    return max(0.0, current_price * (1 - fixed_pct))
                elif trend_direction == '-':
                    return current_price * (1 + fixed_pct)
                else:
                    return None
            except:
                return None # Final fallback

    def calculate_confidence(self, data: pd.DataFrame, prediction_type: str = 'short') -> float:
        """
        Calculate confidence rating for predictions
        
        Args:
            data (pd.DataFrame): Stock price data
            prediction_type (str): 'short' or 'long' term prediction
            
        Returns:
            float: Confidence percentage (0-100)
        """
        try:
            # Use neural network confidence estimator if available
            if 'confidence_estimator' in self.models:
                input_data = self.prepare_prediction_data(data)  # <-- This is now correct!
                if input_data is not None:
                    confidence_pred = self.models['confidence_estimator'].predict(input_data, verbose=0)
                    base_confidence = float(confidence_pred[0][0])
                else:
                    base_confidence = 0.5
            else:
                base_confidence = 0.5
            
            # Adjust confidence based on data quality factors
            confidence_factors = []
            
            # Data completeness factor
            required_data_points = self.long_window + 10
            if len(data) >= required_data_points:
                completeness_factor = 1.0
            elif len(data) >= self.short_window:
                completeness_factor = 0.7
            else:
                completeness_factor = 0.3
            
            confidence_factors.append(completeness_factor)
            
            # Volatility factor (lower volatility = higher confidence)
            if len(data) >= 20:
                recent_volatility = data['close'].pct_change().tail(20).std()
                historical_volatility = data['close'].pct_change().std()
                
                if historical_volatility > 0:
                    vol_ratio = recent_volatility / historical_volatility
                    vol_factor = max(0.3, 1.0 - (vol_ratio - 1.0) * 0.5)  # Lower confidence for higher vol
                else:
                    vol_factor = 0.7
                
                confidence_factors.append(vol_factor)
            
            # Trend consistency factor
            if prediction_type == 'short' and len(data) >= self.short_window:
                window = self.short_window
            elif prediction_type == 'long' and len(data) >= self.long_window:
                window = self.long_window
            else:
                window = min(10, len(data))
            
            if window > 0:
                recent_data = data.tail(window)
                price_changes = recent_data['close'].diff().dropna()
                
                if len(price_changes) > 0:
                    # Measure directional consistency
                    positive_changes = (price_changes > 0).sum()
                    negative_changes = (price_changes < 0).sum()
                    total_changes = len(price_changes)
                    
                    if total_changes > 0:
                        consistency = max(positive_changes, negative_changes) / total_changes
                        confidence_factors.append(consistency)
            
            # Combine factors
            if confidence_factors:
                final_confidence = base_confidence * np.mean(confidence_factors)
            else:
                final_confidence = base_confidence
            
            # Convert to percentage and clip
            confidence_pct = np.clip(final_confidence * 100, 0, 100)
            
            return float(confidence_pct)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 50.0  # Default moderate confidence
    
    def predict_short_reversal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict short-term reversal price
        
        Args:
            data (pd.DataFrame): Stock price data
            
        Returns:
            Dict[str, Any]: Short-term reversal prediction
        """
        return self.predict_reversal_price(data, 'short')
    
    '''
    def predict_long_reversal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict long-term reversal price
        
        Args:
            data (pd.DataFrame): Stock price data
            
        Returns:
            Dict[str, Any]: Long-term reversal prediction
        """
        return self.predict_reversal_price(data, 'long')
    '''
    
    def generate_ensemble_prediction(self, data: pd.DataFrame, prediction_type: str) -> Dict[str, Any]:
        """
        Generate ensemble predictions for improved accuracy
        
        Args:
            data (pd.DataFrame): Stock price data
            prediction_type (str): Type of prediction ('trend' or 'reversal')
            
        Returns:
            Dict[str, Any]: Ensemble prediction results
        """
        try:
            predictions = []
            confidences = []
            
            # Generate multiple predictions with slight data variations
            for i in range(self.ensemble_size):
                # Add small noise to input data for ensemble diversity
                noisy_data = data.copy()
                if len(data) > 10:
                    noise_factor = 0.001  # 0.1% noise
                    price_cols = ['open', 'high', 'low', 'close']
                    for col in price_cols:
                        if col in noisy_data.columns:
                            noise = np.random.normal(1, noise_factor, len(noisy_data))
                            noisy_data[col] = noisy_data[col] * noise
                
                # Generate prediction
                if prediction_type == 'trend':
                    short_pred = self.predict_trend(noisy_data, 'short')
                    long_pred = self.predict_trend(noisy_data, 'long')
                    pred_result = {
                        'short_trend': short_pred['trend'],
                        'long_trend': long_pred['trend'],
                        'short_confidence': short_pred['confidence'],
                        # 'long_confidence': long_pred['confidence']
                    }
                elif prediction_type == 'reversal':
                    short_rev = self.predict_short_reversal(noisy_data)
                    # long_rev = self.predict_long_reversal(noisy_data)
                    pred_result = {
                        'short_reversal_price': short_rev['price'],
                        # 'long_reversal_price': long_rev['price'],
                        'short_confidence': short_rev['confidence'],
                        # 'long_confidence': long_rev['confidence']
                    }
                else:
                    continue
                
                predictions.append(pred_result)
                confidences.append(np.mean([pred_result.get('short_confidence', 0.5),
                                          pred_result.get('long_confidence', 0.5)]))
            
            if not predictions:
                return {}
            
            # Aggregate predictions
            if prediction_type == 'trend':
                # Majority voting for trends
                short_trends = [p['short_trend'] for p in predictions]
                long_trends = [p['long_trend'] for p in predictions]
                
                ensemble_result = {
                    'short_trend': max(set(short_trends), key=short_trends.count),
                    'long_trend': max(set(long_trends), key=long_trends.count),
                    'short_confidence': np.mean([p['short_confidence'] for p in predictions]),
                    # 'long_confidence': np.mean([p['long_confidence'] for p in predictions]),
                    'ensemble_confidence': np.mean(confidences)
                }
                
            elif prediction_type == 'reversal':
                # Average for reversal prices
                ensemble_result = {
                    'short_reversal_price': np.mean([p['short_reversal_price'] for p in predictions]),
                    # 'long_reversal_price': np.mean([p['long_reversal_price'] for p in predictions]),
                    'short_confidence': np.mean([p['short_confidence'] for p in predictions]),
                    # 'long_confidence': np.mean([p['long_confidence'] for p in predictions]),
                    'ensemble_confidence': np.mean(confidences),
                    'price_std': {
                        'short': np.std([p['short_reversal_price'] for p in predictions]),
                        # 'long': np.std([p['long_reversal_price'] for p in predictions])
                    }
                }
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {}
    
    def predict_stock(self, data: pd.DataFrame, stock_code: str) -> Dict[str, Any]:
        """
        Generate complete prediction for a single stock
        
        Args:
            data (pd.DataFrame): Stock price data
            stock_code (str): Stock symbol
            
        Returns:
            Dict[str, Any]: Complete prediction results
        """
        try:
            # --- FIX: Load models within the child process ---
            # Check if models are loaded in this process. If not, load them.
            # This prevents pickling loaded Keras models from the parent process.
            if not self.models:
                logger.info(f"Loading models for worker process (PID: {os.getpid()})")
                self.load_models()
            # --- END FIX ---

            logger.info(f"Generating predictions for {stock_code}")
            
            # Current market data
            current_price = float(data['close'].iloc[-1]) if len(data) > 0 else 0.0
            current_date = data['date'].iloc[-1] if 'date' in data.columns and len(data) > 0 else datetime.now()
            
            # Basic trend predictions
            short_trend = self.predict_trend(data, 'short')
            long_trend = self.predict_trend(data, 'long')
            
            # Reversal price predictions
            short_reversal = self.predict_short_reversal(data)
            # long_reversal = self.predict_long_reversal(data)
            
            # Confidence ratings
            short_confidence = self.calculate_confidence(data, 'short')
            # long_confidence = self.calculate_confidence(data, 'long')
            
            # *** NEW: Calculate Bailout Points ***
            short_bailout_point = self._calculate_bailout_point(data, current_price, short_trend['trend'])
            # long_bailout_point = self._calculate_bailout_point(data, current_price, long_trend['trend'])
            
            prediction_result = {
                'stock_code': stock_code,
                'current_price': current_price,
                'analysis_date': current_date.isoformat() if hasattr(current_date, 'isoformat') else str(current_date),
                
                # Trend predictions
                'short_trend': short_trend['trend'],
                'long_trend': long_trend['trend'],
                
                # Reversal price predictions
                'short_reversal_price': short_reversal['price'],
                # 'long_reversal_price': long_reversal['price'],
                
                # *** NEW: Bailout points ***
                'short_bailout_point': short_bailout_point,
                # 'long_bailout_point': long_bailout_point,
                
                # Confidence ratings (0-100%)
                'short_confidence': short_confidence,
                # 'long_confidence': long_confidence,
                
                # Additional metadata
                'prediction_methods': {
                    'short_trend_method': short_trend.get('method', 'unknown'),
                    'long_trend_method': long_trend.get('method', 'unknown'),
                    'short_reversal_method': short_reversal.get('method', 'unknown')
                    #'long_reversal_method': long_reversal.get('method', 'unknown')
                },
                
                'data_quality': {
                    'data_points': len(data),
                    'sufficient_short': len(data) >= self.short_window,
                    'sufficient_long': len(data) >= self.long_window,
                    'sufficient_ml': len(data) >= self.lookback_period
                }
            }
            
            # Add ensemble predictions if enabled
            if self.ensemble_size > 1:
                try:
                    ensemble_trends = self.generate_ensemble_prediction(data, 'trend')
                    ensemble_reversals = self.generate_ensemble_prediction(data, 'reversal')
                    
                    if ensemble_trends:
                        prediction_result['ensemble_trends'] = ensemble_trends
                    if ensemble_reversals:
                        prediction_result['ensemble_reversals'] = ensemble_reversals
                        
                except Exception as e:
                    logger.warning(f"Ensemble prediction failed for {stock_code}: {e}")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error predicting stock {stock_code}: {e}")
            return {
                'stock_code': stock_code,
                'error': str(e),
                'timestamp': datetime.now(JST).isoformat()
            }
    
    def batch_predict(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Generate predictions for multiple stocks in parallel.
        """
        logger.info(f"Generating parallel batch predictions for {len(data_dict)} stocks")

        predictions = {}
        # Use max_workers=8 to match the e2-standard-8 VM's vCPUs
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            # Create a dictionary to map futures to stock codes
            future_to_stock = {
                executor.submit(self.predict_stock, stock_data, stock_code): stock_code
                for stock_code, stock_data in data_dict.items()
            }

            total_stocks = len(data_dict)
            completed_count = 0
            success_count = 0

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_stock):
                stock_code = future_to_stock[future]
                completed_count += 1
                try:
                    result = future.result()
                    predictions[stock_code] = result
                    if 'error' not in result:
                        success_count += 1
                    logger.info(f"Completed prediction for {stock_code} ({completed_count}/{total_stocks})")
                except Exception as e:
                    logger.error(f"Error in parallel prediction for {stock_code}: {e}")
                    predictions[stock_code] = {
                        'stock_code': stock_code,
                        'error': str(e),
                        'timestamp': datetime.now(JST).isoformat()
                    }

        logger.info(f"Parallel batch prediction completed: {success_count}/{len(data_dict)} successful")
        return predictions
    
    def get_prediction_summary(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics from batch predictions
        
        Args:
            predictions (Dict[str, Dict[str, Any]]): Prediction results
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        try:
            summary = {
                'total_stocks': len(predictions),
                'successful_predictions': 0,
                'failed_predictions': 0,
                'trend_distribution': {
                    'short': {'+': 0, '-': 0, '0': 0},
                    'long': {'+': 0, '-': 0, '0': 0}
                },
                'confidence_stats': {
                    'short': {'mean': 0, 'std': 0, 'min': 100, 'max': 0},
                    'long': {'mean': 0, 'std': 0, 'min': 100, 'max': 0}
                },
                'price_change_predictions': {
                    'short_avg_change': 0,
                    'long_avg_change': 0
                }
            }
            
            short_confidences = []
            long_confidences = []
            short_changes = []
            long_changes = []
            
            for stock_code, prediction in predictions.items():
                if 'error' in prediction:
                    summary['failed_predictions'] += 1
                    continue
                
                summary['successful_predictions'] += 1
                
                # Trend distribution
                short_trend = prediction.get('short_trend', '0')
                long_trend = prediction.get('long_trend', '0')
                
                if short_trend in summary['trend_distribution']['short']:
                    summary['trend_distribution']['short'][short_trend] += 1
                if long_trend in summary['trend_distribution']['long']:
                    summary['trend_distribution']['long'][long_trend] += 1
                
                # Confidence statistics
                short_conf = prediction.get('short_confidence', 50)
                # long_conf = prediction.get('long_confidence', 50)
                
                short_confidences.append(short_conf)
                # long_confidences.append(long_conf)
                
                # Price change predictions
                current_price = prediction.get('current_price', 0)
                short_reversal = prediction.get('short_reversal_price', current_price)
                # long_reversal = prediction.get('long_reversal_price', current_price)
                
                if current_price > 0:
                    short_change = (short_reversal - current_price) / current_price
                    # long_change = (long_reversal - current_price) / current_price
                    short_changes.append(short_change)
                    # long_changes.append(long_change)
            
            # Calculate confidence statistics
            if short_confidences:
                summary['confidence_stats']['short'] = {
                    'mean': float(np.mean(short_confidences)),
                    'std': float(np.std(short_confidences)),
                    'min': float(np.min(short_confidences)),
                    'max': float(np.max(short_confidences))
                }
            
            '''
            if long_confidences:
                summary['confidence_stats']['long'] = {
                    'mean': float(np.mean(long_confidences)),
                    'std': float(np.std(long_confidences)),
                    'min': float(np.min(long_confidences)),
                    'max': float(np.max(long_confidences))
                }
            '''
            
            # Calculate price change statistics
            if short_changes:
                summary['price_change_predictions']['short_avg_change'] = float(np.mean(short_changes))
            if long_changes:
                summary['price_change_predictions']['long_avg_change'] = float(np.mean(long_changes))
            
            summary['generation_time'] = datetime.now(JST).isoformat()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating prediction summary: {e}")
            return {'error': str(e)}

# Example usage and testing
if __name__ == "__main__":
    # For testing purposes
    import os
    from config import Config
    from data_manager import DataManager
    from trend_analyzer import TrendAnalyzer
    from model_trainer import ModelTrainer
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Please install: pip install tensorflow")
        print("Running with limited prediction capabilities...")
    
    # Initialize components
    config = Config("test_config.json")
    data_manager = DataManager(config)
    trend_analyzer = TrendAnalyzer(config)
    model_trainer = ModelTrainer(config)
    predictor = Predictor(config)
    
    # Set dependencies
    predictor.set_dependencies(trend_analyzer, data_manager)
    
    # Load sample data
    try:
        stock_list = data_manager.load_stock_list()
        sample_stocks = stock_list['code'].head(3).tolist()
        print(f"Testing with stocks: {sample_stocks}")
        
        # Load price data
        price_data = data_manager.load_price_data(sample_stocks)
        
        if price_data:
            # Process data through pipeline
            processed_data = trend_analyzer.calculate_moving_averages(price_data)
            trend_data = trend_analyzer.calculate_trend_features(processed_data)
            
            # Try to load trained models
            models_loaded = predictor.load_models(model_trainer)
            print(f"Models loaded: {models_loaded}")
            
            # Test single stock prediction
            test_stock = list(trend_data.keys())[0]
            test_data = trend_data[test_stock]
            
            print(f"\nTesting prediction for stock: {test_stock}")
            
            # Single stock prediction
            prediction = predictor.predict_stock(test_data, test_stock)
            
            print("Prediction Results:")
            print(f"  Current Price: ${prediction.get('current_price', 0):.2f}")
            print(f"  Short Trend: {prediction.get('short_trend', 'N/A')}")
            print(f"  Long Trend: {prediction.get('long_trend', 'N/A')}")
            print(f"  Short Reversal: ${prediction.get('short_reversal_price', 0):.2f}")
            print(f"  Long Reversal: ${prediction.get('long_reversal_price', 0):.2f}")
            print(f"  Short Bailout: ${prediction.get('short_bailout_point', 0):.2f}")
            print(f"  Long Bailout: ${prediction.get('long_bailout_point', 0):.2f}")
            print(f"  Short Confidence: {prediction.get('short_confidence', 0):.1f}%")
            print(f"  Long Confidence: {prediction.get('long_confidence', 0):.1f}%")
            
            if 'error' in prediction:
                print(f"  Error: {prediction['error']}")
            
            # Test batch prediction
            print(f"\nTesting batch prediction for {len(trend_data)} stocks...")
            batch_predictions = predictor.batch_predict(trend_data)
            
            # Generate summary
            summary = predictor.get_prediction_summary(batch_predictions)
            
            print("\nBatch Prediction Summary:")
            print(f"  Total stocks: {summary.get('total_stocks', 0)}")
            print(f"  Successful: {summary.get('successful_predictions', 0)}")
            print(f"  Failed: {summary.get('failed_predictions', 0)}")
            
            if 'trend_distribution' in summary:
                short_trends = summary['trend_distribution']['short']
                long_trends = summary['trend_distribution']['long']
                print(f"  Short trends: +{short_trends.get('+', 0)} -{short_trends.get('-', 0)} 0{short_trends.get('0', 0)}")
                print(f"  Long trends: +{long_trends.get('+', 0)} -{long_trends.get('-', 0)} 0{long_trends.get('0', 0)}")
            
            if 'confidence_stats' in summary:
                short_conf = summary['confidence_stats']['short']
                long_conf = summary['confidence_stats']['long']
                if short_conf.get('mean', 0) > 0:
                    print(f"  Avg Short Confidence: {short_conf['mean']:.1f}%")
                if long_conf.get('mean', 0) > 0:
                    print(f"  Avg Long Confidence: {long_conf['mean']:.1f}%")
            
            # Test individual prediction methods
            print(f"\nTesting individual prediction methods for {test_stock}:")
            
            # Test trend prediction
            short_trend_pred = predictor.predict_trend(test_data, 'short')
            long_trend_pred = predictor.predict_trend(test_data, 'long')
            
            print(f"  Short trend prediction: {short_trend_pred}")
            print(f"  Long trend prediction: {long_trend_pred}")
            
            # Test reversal prediction
            short_reversal_pred = predictor.predict_short_reversal(test_data)
            # long_reversal_pred = predictor.predict_long_reversal(test_data)
            
            print(f"  Short reversal prediction: {short_reversal_pred}")
            # print(f"  Long reversal prediction: {long_reversal_pred}")
            
            # Test confidence calculation
            short_confidence = predictor.calculate_confidence(test_data, 'short')
            # long_confidence = predictor.calculate_confidence(test_data, 'long')
            
            print(f"  Short confidence: {short_confidence:.1f}%")
            # print(f"  Long confidence: {long_confidence:.1f}%")
            
            print("\nTesting completed successfully!")
            
        else:
            print("No price data available for testing")
            
    except Exception as e:
        print(f"Testing failed with error: {e}")
        import traceback
        traceback.print_exc()