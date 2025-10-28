#!/usr/bin/env python3
"""
Neural Network Model Training for Stock Prediction System

This module handles training of neural network models for:
- Short-term trend prediction (5-day MA slope)
- Long-term trend prediction (25-day MA slope)
- Trend reversal price prediction (short & long)
- Confidence estimation models
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks, optimizers
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Model training will be limited.")

import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Neural Network Model Trainer for stock prediction system
    
    Handles training of multiple models:
    - Trend direction classifiers (short & long)
    - Reversal price regressors (short & long)
    - Confidence estimation models
    """
    
    def __init__(self, config):
        """
        Initialize ModelTrainer
        
        Args:
            config: Configuration object containing model settings
        """
        self.config = config
        self.models = {}
        self.model_history = {}
        self.label_encoders = {}
        self.feature_columns = {}
        self.scaler = StandardScaler()
        self.scaler_path = os.path.join(self.config.system.model_save_path, 'feature_scaler.gz')
        
        # Set TensorFlow settings
        if TENSORFLOW_AVAILABLE:
            tf.random.set_seed(42)
            # Configure GPU if available
            self._configure_gpu()
        
        logger.info("ModelTrainer initialized")
    
    def _configure_gpu(self):
        """Configure GPU settings for TensorFlow"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Configured {len(gpus)} GPU(s) for training")
            else:
                logger.info("No GPU available, using CPU for training")
        except Exception as e:
            logger.warning(f"GPU configuration failed: {e}")
    
    def prepare_training_data(self, data: Dict[str, pd.DataFrame],fit_scaler: bool = False) -> Dict[str, Any]:
        """
        Prepare data for neural network training
        
        Args:
            data (Dict[str, pd.DataFrame]): Processed stock data with features
            
        Returns:
            Dict[str, Any]: Prepared training datasets
        """
        logger.info("Preparing training data for neural network models")
        
        training_data = {
            'short_trend_classifier': {'X': [], 'y': []},
            'long_trend_classifier': {'X': [], 'y': []},
            'short_reversal_regressor': {'X': [], 'y': []},
            'long_reversal_regressor': {'X': [], 'y': []},
            'confidence_estimator': {'X': [], 'y': []}
        }
        
        # Feature columns for different models
        base_features = [
            'open', 'high', 'low', 'close', 'volume',
            'return_1d', 'volatility_5d', 'volatility_20d',
            'rsi', 'macd', 'macd_signal'
        ]
        
        trend_features = base_features + [
            'short_trend_slope', 'long_trend_slope',
            'short_trend_strength', 'long_trend_strength',
            'trend_agreement', 'ma_convergence'
        ]
        
        for stock_code, stock_data in data.items():
            try:
                # Prepare data for each stock
                stock_training_data = self._prepare_stock_data(stock_data, trend_features)
                
                # Append to training datasets
                for model_name, model_data in stock_training_data.items():
                    if model_name in training_data and model_data['X'] is not None:
                        training_data[model_name]['X'].append(model_data['X'])
                        training_data[model_name]['y'].append(model_data['y'])
                
            except Exception as e:
                logger.warning(f"Error preparing data for {stock_code}: {e}")
        
        # Convert to numpy arrays and create sequences
        prepared_data = {}
        for model_name, model_data in training_data.items():
            if model_data['X'] and model_data['y']:
                try:
                    X_combined = np.vstack(model_data['X'])
                    y_combined = np.hstack(model_data['y'])
                    
                    X_scaled = None
                    if fit_scaler:
                        X_scaled = self.scaler.fit_transform(X_combined)
                        logger.info(f"Scaler fitted on {model_name} data.")
                    else:
                        try:
                            X_scaled = self.scaler.transform(X_combined)
                        except Exception as e:
                            logger.error(f"Error transforming data with scaler: {e}. Scaler may not be fitted. Set fit_scaler=True on training data.")
                            X_scaled = X_combined

                    # Create sequences for time series prediction
                    X_seq, y_seq = self._create_sequences(
                        X_scaled, y_combined, 
                        sequence_length=self.config.model.lookback_period
                    )
                    
                    prepared_data[model_name] = {'X': X_seq, 'y': y_seq}
                    logger.info(f"Prepared {len(X_seq)} sequences for {model_name}")
                    
                except Exception as e:
                    logger.error(f"Error combining data for {model_name}: {e}")
        
        return prepared_data
    
    def _prepare_stock_data(self, data: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        Prepare training data for a single stock
        
        Args:
            data (pd.DataFrame): Stock data with features
            feature_columns (List[str]): List of feature column names
            
        Returns:
            Dict[str, Any]: Prepared data for each model type
        """
        stock_data = {}
        
        # Filter features that exist in the data
        available_features = [col for col in feature_columns if col in data.columns]
        
        if not available_features or len(data) < self.config.model.lookback_period + 10:
            return {model: {'X': None, 'y': None} for model in 
                   ['short_trend_classifier', 'long_trend_classifier', 
                    'short_reversal_regressor', 'long_reversal_regressor', 'confidence_estimator']}
        
        # Extract features
        X = data[available_features].ffill().fillna(0).values
        
        # Prepare targets for different models
        try:
            # Short trend classification targets
            if 'short_trend_direction' in data.columns:
                y_short_trend = data['short_trend_direction'].fillna(0).values
                stock_data['short_trend_classifier'] = {'X': X, 'y': y_short_trend}
            
            # Long trend classification targets
            if 'long_trend_direction' in data.columns:
                y_long_trend = data['long_trend_direction'].fillna(0).values
                stock_data['long_trend_classifier'] = {'X': X, 'y': y_long_trend}
            
            # Short reversal price regression targets
            if 'short_trend_slope' in data.columns:
                # Use future price changes as reversal targets
                future_returns = data['close'].pct_change(self.config.model.short_term_window).shift(-self.config.model.short_term_window)
                y_short_reversal = future_returns.fillna(0).values
                stock_data['short_reversal_regressor'] = {'X': X, 'y': y_short_reversal}
            
            # Long reversal price regression targets
            if 'long_trend_slope' in data.columns:
                # Use future price changes as reversal targets
                future_returns = data['close'].pct_change(self.config.model.long_term_window).shift(-self.config.model.long_term_window)
                y_long_reversal = future_returns.fillna(0).values
                stock_data['long_reversal_regressor'] = {'X': X, 'y': y_long_reversal}
            
            # Confidence estimation targets (based on trend strength and consistency)
            if 'short_trend_strength' in data.columns and 'short_trend_consistency' in data.columns:
                confidence_score = (data['short_trend_strength'].fillna(0) * 
                                  data['short_trend_consistency'].fillna(0.5))
                y_confidence = confidence_score.clip(0, 1).values
                stock_data['confidence_estimator'] = {'X': X, 'y': y_confidence}
            
        except Exception as e:
            logger.warning(f"Error preparing targets: {e}")
        
        return stock_data
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        
        Args:
            X (np.ndarray): Feature data
            y (np.ndarray): Target data
            sequence_length (int): Length of input sequences
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Sequenced X and y data
        """
        if len(X) <= sequence_length:
            return np.array([]), np.array([])
        
        X_seq = []
        y_seq = []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_trend_classifier(self, input_shape: Tuple, model_name: str) -> keras.Model:
        """
        Build neural network for trend classification
        
        Args:
            input_shape (Tuple): Input shape for the model
            model_name (str): Name of the model
            
        Returns:
            keras.Model: Compiled neural network model
        """
        model = models.Sequential(name=model_name)
        
        # LSTM layers for time series
        model.add(layers.LSTM(
            units=self.config.model.hidden_layers[0],
            return_sequences=True,
            input_shape=input_shape,
            dropout=self.config.model.dropout_rate
        ))
        
        model.add(layers.LSTM(
            units=self.config.model.hidden_layers[1],
            return_sequences=False,
            dropout=self.config.model.dropout_rate
        ))
        
        # Dense layers
        for units in self.config.model.hidden_layers[2:]:
            model.add(layers.Dense(
                units=units,
                activation=self.config.model.activation
            ))
            model.add(layers.Dropout(self.config.model.dropout_rate))
        
        # Output layer for classification (3 classes: -1, 0, 1)
        model.add(layers.Dense(3, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.model.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_reversal_regressor(self, input_shape: Tuple, model_name: str) -> keras.Model:
        """
        Build neural network for reversal price regression
        
        Args:
            input_shape (Tuple): Input shape for the model
            model_name (str): Name of the model
            
        Returns:
            keras.Model: Compiled neural network model
        """
        model = models.Sequential(name=model_name)
        
        # LSTM layers for time series
        model.add(layers.LSTM(
            units=self.config.model.hidden_layers[0],
            return_sequences=True,
            input_shape=input_shape,
            dropout=self.config.model.dropout_rate
        ))
        
        model.add(layers.LSTM(
            units=self.config.model.hidden_layers[1],
            return_sequences=False,
            dropout=self.config.model.dropout_rate
        ))
        
        # Dense layers
        for units in self.config.model.hidden_layers[2:]:
            model.add(layers.Dense(
                units=units,
                activation=self.config.model.activation
            ))
            model.add(layers.Dropout(self.config.model.dropout_rate))
        
        # Output layer for regression
        model.add(layers.Dense(1, activation=self.config.model.output_activation))
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.model.learning_rate),
            loss=self.config.model.loss_function,
            metrics=self.config.model.metrics
        )
        
        return model
    
    def build_confidence_estimator(self, input_shape: Tuple) -> keras.Model:
        """
        Build neural network for confidence estimation
        
        Args:
            input_shape (Tuple): Input shape for the model
            
        Returns:
            keras.Model: Compiled neural network model
        """
        model = models.Sequential(name='confidence_estimator')
        
        # LSTM layers for time series
        model.add(layers.LSTM(
            units=self.config.model.hidden_layers[0] // 2,
            return_sequences=True,
            input_shape=input_shape,
            dropout=self.config.model.dropout_rate
        ))
        
        model.add(layers.LSTM(
            units=self.config.model.hidden_layers[1] // 2,
            return_sequences=False,
            dropout=self.config.model.dropout_rate
        ))
        
        # Dense layers
        model.add(layers.Dense(
            units=self.config.model.hidden_layers[2] // 2,
            activation=self.config.model.activation
        ))
        model.add(layers.Dropout(self.config.model.dropout_rate))
        
        # Output layer for confidence (0-1)
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.model.learning_rate),
            loss='binary_crossentropy',
            metrics=['mae']
        )
        
        return model
    
    def train_short_term_model(self, data: Dict[str, pd.DataFrame], validation_data: Dict[str, pd.DataFrame] = None):
        """
        Train short-term trend prediction model
        
        Args:
            data (Dict[str, pd.DataFrame]): Training data
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available for model training")
            return
        
        logger.info("Preparing training data for short-term model")
        training_data = self.prepare_training_data(data, fit_scaler=True)
        X_train = training_data['short_trend_classifier']['X']
        y_train = training_data['short_trend_classifier']['y']
        
        # Prepare validation data
        validation_set = None
        if validation_data:
            logger.info("Preparing validation data for short-term model")
            val_prep = self.prepare_training_data(validation_data, fit_scaler=False)
            if 'short_trend_classifier' in val_prep and len(val_prep['short_trend_classifier']['X']) > 0:
                X_val = val_prep['short_trend_classifier']['X']
                y_val = val_prep['short_trend_classifier']['y']
                y_val_categorical = y_val + 1  # Convert labels
                validation_set = (X_val, y_val_categorical)
                logger.info(f"Using chronological validation set with {len(X_val)} samples.")
        
        if len(X_train) == 0:
            logger.error("No training sequences available for short-term model")
            return
        
        # Convert labels to categorical (from -1,0,1 to 0,1,2)
        y_train_categorical = y_train + 1
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_trend_classifier(input_shape, 'short_trend_classifier')
        
        # Set up callbacks
        callbacks_list = self._create_callbacks('short_trend_model')
        
        # Train model
        history = model.fit(
            X_train, y_train_categorical,
            epochs=self.config.model.epochs,
            batch_size=self.config.model.batch_size,
            validation_data=validation_set, 
            callbacks=callbacks_list,
            verbose=1
        )
        # Store model and history
        self.models['short_trend_classifier'] = model
        self.model_history['short_trend_classifier'] = history.history
        
        logger.info("Short-term trend model training completed")
    
    def train_long_term_model(self, data: Dict[str, pd.DataFrame], validation_data: Dict[str, pd.DataFrame] = None):
        """
        Train long-term trend prediction model
        
        Args:
            data (Dict[str, pd.DataFrame]): Training data
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available for model training")
            return
        
        logger.info("Training long-term trend model")
        
        # Prepare training data
        training_data = self.prepare_training_data(data, fit_scaler=True)
        
        if 'long_trend_classifier' not in training_data:
            logger.error("No data prepared for long-term trend model")
            return
        
        X_train = training_data['long_trend_classifier']['X']
        y_train = training_data['long_trend_classifier']['y']
        
        if len(X_train) == 0:
            logger.error("No training sequences available for long-term model")
            return
        
        # Convert labels to categorical (from -1,0,1 to 0,1,2)
        y_train_categorical = y_train + 1
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_trend_classifier(input_shape, 'long_trend_classifier')
        
        # Set up callbacks
        callbacks_list = self._create_callbacks('long_trend_model')

        validation_set = None
        if validation_data:
            logger.info("Preparing validation data for long-term model")
            val_prep = self.prepare_training_data(validation_data, fit_scaler=False)
            if 'long_trend_classifier' in val_prep and len(val_prep['long_trend_classifier']['X']) > 0:
                X_val = val_prep['long_trend_classifier']['X']
                y_val = val_prep['long_trend_classifier']['y']
                y_val_categorical = y_val + 1  # Convert labels
                validation_set = (X_val, y_val_categorical)
                logger.info(f"Using chronological validation set with {len(X_val)} samples.")
        
        # Train model
        history = model.fit(
            X_train, y_train_categorical,
            epochs=self.config.model.epochs,
            batch_size=self.config.model.batch_size,
            validation_data=validation_set,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Store model and history
        self.models['long_trend_classifier'] = model
        self.model_history['long_trend_classifier'] = history.history
        
        logger.info("Long-term trend model training completed")
    
    def train_reversal_models(self, data: Dict[str, pd.DataFrame], validation_data: Dict[str, pd.DataFrame] = None):
        """
        Train reversal price prediction models
        
        Args:
            data (Dict[str, pd.DataFrame]): Training data
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available for model training")
            return
        
        logger.info("Training reversal price prediction models")
        
        # Prepare training data
        training_data = self.prepare_training_data(data, fit_scaler=True)

        val_prep = {}
        if validation_data:
            logger.info("Preparing validation data for reversal models")
            val_prep = self.prepare_training_data(validation_data, fit_scaler=False)
        
        # Train short reversal model
        if 'short_reversal_regressor' in training_data:
            self._train_reversal_model(
                training_data['short_reversal_regressor'],
                'short_reversal_regressor',
                val_prep.get('short_reversal_regressor')
            )
        
        # Train long reversal model
        if 'long_reversal_regressor' in training_data:
            self._train_reversal_model(
                training_data['long_reversal_regressor'],
                'long_reversal_regressor',
                val_prep.get('long_reversal_regressor')
            )
        
        # Train confidence estimator
        if 'confidence_estimator' in training_data:
            self._train_confidence_model(training_data['confidence_estimator'], val_prep.get('confidence_estimator'))
        
        logger.info("Reversal models training completed")
    
    def _train_reversal_model(self, training_data: Dict[str, np.ndarray], model_name: str, validation_set_data: Dict[str, pd.DataFrame] = None):
        """
        Train a single reversal prediction model
        
        Args:
            training_data (Dict[str, np.ndarray]): Training data
            model_name (str): Name of the model
        """
        X_train = training_data['X']
        y_train = training_data['y']
        
        if len(X_train) == 0:
            logger.warning(f"No training data available for {model_name}")
            return
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_reversal_regressor(input_shape, model_name)
        
        # Set up callbacks
        callbacks_list = self._create_callbacks(model_name)

        #Prepare validation data
        validation_set = None
        if validation_set_data and len(validation_set_data['X']) > 0:
            validation_set = (validation_set_data['X'], validation_set_data['y'])
            logger.info(f"Using validation set with {len(validation_set[0])} samples for {model_name}.")
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=self.config.model.epochs,
            batch_size=self.config.model.batch_size,
            validation_data=validation_set,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Store model and history
        self.models[model_name] = model
        self.model_history[model_name] = history.history
        
        logger.info(f"{model_name} training completed")
    
    def _train_confidence_model(self, training_data: Dict[str, np.ndarray], validation_set_data: Dict[str, pd.DataFrame] = None):
        """
        Train confidence estimation model
        
        Args:
            training_data (Dict[str, np.ndarray]): Training data
        """
        X_train = training_data['X']
        y_train = training_data['y']
        
        if len(X_train) == 0:
            logger.warning("No training data available for confidence estimator")
            return
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.build_confidence_estimator(input_shape)
        
        # Set up callbacks
        callbacks_list = self._create_callbacks('confidence_estimator')

        #Prepare validation data
        validation_set = None
        if validation_set_data and len(validation_set_data['X']) > 0:
            validation_set = (validation_set_data['X'], validation_set_data['y'])
            logger.info(f"Using validation set with {len(validation_set[0])} samples for confidence_estimator.")
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=self.config.model.epochs // 2,  # Fewer epochs for confidence model
            batch_size=self.config.model.batch_size,
            validation_data=validation_set,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Store model and history
        self.models['confidence_estimator'] = model
        self.model_history['confidence_estimator'] = history.history
        
        logger.info("Confidence estimator training completed")
    
    def _create_callbacks(self, model_name: str) -> List[keras.callbacks.Callback]:
        """
        Create training callbacks
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            List[keras.callbacks.Callback]: List of callbacks
        """
        callbacks_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.model.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks_list.append(early_stopping)
        
        # Reduce learning rate
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config.model.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
        callbacks_list.append(reduce_lr)
        
        # Model checkpoint
        checkpoint_path = self.config.get_model_path(f"{model_name}_checkpoint")
        # --- FIX: Change .h5 to .keras to avoid legacy warning ---
        if checkpoint_path.endswith('.h5'):
            checkpoint_path = checkpoint_path.replace('.h5', '.keras')
        # --- End Fix ---
            
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        
        return callbacks_list
    
    def evaluate_models(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Evaluate all trained models on test data
        
        Args:
            test_data (Dict[str, pd.DataFrame]): Test data
            
        Returns:
            Dict[str, Any]: Evaluation metrics for all models
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available for model evaluation")
            return {}
        
        logger.info("Evaluating trained models")
        
        # Prepare test data
        test_datasets = self.prepare_training_data(test_data, fit_scaler=False)
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            if model_name not in test_datasets:
                continue
            
            try:
                X_test = test_datasets[model_name]['X']
                y_test = test_datasets[model_name]['y']
                
                if len(X_test) == 0:
                    continue
                
                # Make predictions
                if 'classifier' in model_name:
                    # Classification metrics
                    y_test_categorical = y_test + 1  # Convert to 0,1,2
                    predictions = model.predict(X_test, verbose=0)
                    y_pred = np.argmax(predictions, axis=1)
                    
                    accuracy = accuracy_score(y_test_categorical, y_pred)
                    evaluation_results[model_name] = {
                        'accuracy': float(accuracy),
                        'type': 'classification'
                    }
                    
                else:
                    # Regression metrics
                    y_pred = model.predict(X_test, verbose=0).flatten()
                    
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    evaluation_results[model_name] = {
                        'mse': float(mse),
                        'mae': float(mae),
                        'r2': float(r2),
                        'type': 'regression'
                    }
                
                logger.info(f"Evaluated {model_name}: {evaluation_results[model_name]}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        return evaluation_results
    
    def save_models(self):
        """Save all trained models and metadata"""
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available for model saving")
            return
        
        logger.info("Saving trained models")
        
        # Create models directory
        os.makedirs(self.config.system.model_save_path, exist_ok=True)
        
        for model_name, model in self.models.items():
            try:
                # Save model
                model_path = self.config.get_model_path(model_name)
                # --- FIX: Change .h5 to .keras ---
                if model_path.endswith('.h5'):
                    model_path = model_path.replace('.h5', '.keras')
                # --- End Fix ---
                model.save(model_path)
                
                # Save training history
                history_path = self.config.get_model_path(f"{model_name}_history")
                
                # --- FIX: Robust path for history JSON file ---
                # This splits "path/history.h5" into ("path/history", ".h5")
                base_history_path, _ = os.path.splitext(history_path)
                history_json_path = base_history_path + '.json'
                # --- End Fix ---

                with open(history_json_path, 'w') as f:
                    json.dump(self.model_history.get(model_name, {}), f, indent=2)
                
                logger.info(f"Saved {model_name} to {model_path}")
                
            except Exception as e:
                logger.error(f"Error saving {model_name}: {e}")

        try:
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Saved feature scaler to {self.scaler_path}")
        except Exception as e:
            logger.error(f"Error saving feature scaler: {e}")

        # Save metadata
        metadata = {
            'models': list(self.models.keys()),
            'training_date': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'tensorflow_version': tf.__version__ if TENSORFLOW_AVAILABLE else 'N/A'
        }
        
        metadata_path = os.path.join(self.config.system.model_save_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("All models saved successfully")
    
    def load_models(self) -> bool:
        """
        Load trained models from disk
        
        Returns:
            bool: True if models loaded successfully
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available for model loading")
            return False
        
        logger.info("Loading trained models")
        
        try:
            # Load metadata
            metadata_path = os.path.join(self.config.system.model_save_path, 'metadata.json')
            if not os.path.exists(metadata_path):
                logger.warning("No model metadata found")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if os.path.exists(self.scaler_path):
                try:
                    self.scaler = joblib.load(self.scaler_path)
                    logger.info(f"Loaded feature scaler from {self.scaler_path}")
                except Exception as e:
                    logger.error(f"Error loading feature scaler: {e}")
            else:
                logger.warning(f"Feature scaler file not found at {self.scaler_path}. You may need to retrain.")
                
            # Load models
            loaded_models = 0
            for model_name in metadata.get('models', []):
                model_path = self.config.get_model_path(model_name)
                if os.path.exists(model_path):
                    try:
                        self.models[model_name] = keras.models.load_model(model_path)
                        loaded_models += 1
                        logger.info(f"Loaded {model_name}")
                    except Exception as e:
                        logger.error(f"Error loading {model_name}: {e}")
            
            logger.info(f"Successfully loaded {loaded_models} models")
            return loaded_models > 0
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of all trained models
        
        Returns:
            Dict[str, Any]: Model summary information
        """
        summary = {
            'total_models': len(self.models),
            'model_details': {},
            'tensorflow_available': TENSORFLOW_AVAILABLE
        }
        
        for model_name, model in self.models.items():
            try:
                model_info = {
                    'parameters': model.count_params() if hasattr(model, 'count_params') else 0,
                    'layers': len(model.layers) if hasattr(model, 'layers') else 0,
                    'input_shape': str(model.input_shape) if hasattr(model, 'input_shape') else 'unknown',
                    'output_shape': str(model.output_shape) if hasattr(model, 'output_shape') else 'unknown'
                }
                summary['model_details'][model_name] = model_info
            except Exception as e:
                summary['model_details'][model_name] = {'error': str(e)}
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # For testing purposes
    from config import Config
    from data_manager import DataManager
    from trend_analyzer import TrendAnalyzer
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Please install: pip install tensorflow")
        exit(1)
    
    # Initialize components
    config = Config("test_config.json")
    data_manager = DataManager(config)
    trend_analyzer = TrendAnalyzer(config)
    model_trainer = ModelTrainer(config)
    
    # Load and prepare sample data
    stock_list = data_manager.load_stock_list()
    sample_stocks = stock_list['code'].head(5).tolist()
    price_data = data_manager.load_price_data(sample_stocks)
    
    if price_data:
        # Process data through the pipeline
        processed_data = trend_analyzer.calculate_moving_averages(price_data)
        trend_data = trend_analyzer.calculate_trend_features(processed_data)
        
        # Split data for training and testing
        train_data, val_data, test_data = data_manager.split_data(trend_data)
        
        print(f"Training data: {len(train_data)} stocks")
        print(f"Validation data: {len(val_data)} stocks")
        print(f"Test data: {len(test_data)} stocks")
        
        if train_data:
            # Train models
            print("Training short-term model...")
            model_trainer.train_short_term_model(train_data, val_data)
            
            print("Training long-term model...")
            model_trainer.train_long_term_model(train_data, val_data)
            
            print("Training reversal models...")
            model_trainer.train_reversal_models(train_data, val_data)
            
            # Evaluate models
            if test_data:
                print("Evaluating models...")
                evaluation_results = model_trainer.evaluate_models(test_data)
                print("Evaluation Results:")
                for model_name, metrics in evaluation_results.items():
                    print(f"  {model_name}: {metrics}")
            
            # Save models
            print("Saving models...")
            model_trainer.save_models()
            
            # Get model summary
            summary = model_trainer.get_model_summary()
            print(f"\nModel Summary: {summary['total_models']} models trained")
            
        else:
            print("No training data available")
    else:
        print("No sample data loaded for testing")