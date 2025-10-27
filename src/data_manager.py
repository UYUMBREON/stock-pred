#!/usr/bin/env python3
"""
Data Management for Stock Prediction System

This module handles all data-related operations including loading stock lists,
price data, preprocessing, validation, and data splitting for training/testing.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import json
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class DataManager:
    """
    Data Management class for stock prediction system
    
    Handles loading, preprocessing, validation, and splitting of stock data
    """
    
    def __init__(self, config):
        """
        Initialize DataManager
        
        Args:
            config: Configuration object containing data settings
        """
        self.config = config
        self.stock_list = None
        self.price_data = {}
        self.processed_data = {}
        self.scalers = {}
        
        # Create necessary directories
        self._create_directories()
        
        logger.info("DataManager initialized")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.config.data.price_data_path,
            os.path.dirname(self.config.data.stock_list_path),
            "cache/",
            "processed_data/"
        ]
        
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)
    
    def load_stock_list(self) -> pd.DataFrame:
        """
        Load the list of stocks from CSV file
        
        Returns:
            pd.DataFrame: Stock list with company information
        """
        try:
            if not os.path.exists(self.config.data.stock_list_path):
                raise FileNotFoundError(f"Stock list file not found: {self.config.data.stock_list_path}")
            
            self.stock_list = pd.read_csv(self.config.data.stock_list_path, encoding='utf-8')
            
            # Clean and standardize stock codes
            if 'Code' in self.stock_list.columns:
                self.stock_list['Code'] = self.stock_list['Code'].astype(str)
                self.stock_list['Code'] = self.stock_list['Code'].str.zfill(5)  # Pad with zeros
            
            # Add industry classification if missing
            if 'sector_name' not in self.stock_list.columns and 'industry_name' in self.stock_list.columns:
                self.stock_list['sector_name'] = self.stock_list['industry_name']
            
            logger.info(f"Loaded {len(self.stock_list)} stocks from stock list")
            return self.stock_list.copy()
            
        except Exception as e:
            logger.error(f"Error loading stock list: {e}")
            raise
    
    def load_price_data(self, stock_codes: Optional[List[str]] = None, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load price data for specified stocks
        
        Args:
            stock_codes (List[str], optional): List of stock codes to load
            start_date (str, optional): Start date for data loading
            end_date (str, optional): End date for data loading
            
        Returns:
            Dict[str, pd.DataFrame]: Price data for each stock
        """
        try:
            if stock_codes is None and self.stock_list is not None:
                stock_codes = self.stock_list['Code'].tolist()
            elif stock_codes is None:
                # Load all available price data files
                stock_codes = self._discover_available_stocks()
            
            self.price_data = {}
            loaded_count = 0
            
            for stock_code in stock_codes:
                try:
                    data = self._load_single_stock_data(stock_code, start_date, end_date)
                    if data is not None and len(data) >= self.config.data.min_data_points:
                        self.price_data[stock_code] = data
                        loaded_count += 1
                    else:
                        logger.warning(f"Insufficient data for stock {stock_code}")
                        
                except Exception as e:
                    logger.warning(f"Failed to load data for stock {stock_code}: {e}")
            
            logger.info(f"Loaded price data for {loaded_count} stocks")
            return self.price_data.copy()
            
        except Exception as e:
            logger.error(f"Error loading price data: {e}")
            raise
    
    def _discover_available_stocks(self) -> List[str]:
        """
        Discover available stock data files
        
        Returns:
            List[str]: List of available stock codes
        """
        pattern = os.path.join(self.config.data.price_data_path, "*.csv")
        files = glob.glob(pattern)
        
        stock_codes = []
        for file_path in files:
            filename = os.path.basename(file_path)
            stock_code = os.path.splitext(filename)[0]
            stock_codes.append(stock_code)
        
        return stock_codes
    
    def _load_single_stock_data(self, stock_code: str, 
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load price data for a single stock
        
        Args:
            stock_code (str): Stock code
            start_date (str, optional): Start date
            end_date (str, optional): End date
            
        Returns:
            pd.DataFrame: Price data for the stock
        """
        # Try different file naming conventions
        possible_paths = [
            os.path.join(self.config.data.price_data_path, f"{stock_code}.csv"),
            os.path.join(self.config.data.price_data_path, f"{stock_code}_price.csv"),
            os.path.join(self.config.data.price_data_path, "prices", f"{stock_code}.csv"),
        ]
        
        for file_path in possible_paths:
            if os.path.exists(file_path):
                try:
                    data = pd.read_csv(file_path, encoding='utf-8')
                    data = self._preprocess_price_data(data, stock_code)
                    
                    # Filter by date range if specified
                    if start_date or end_date:
                        data = self._filter_by_date(data, start_date, end_date)
                    
                    return data
                    
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
                    continue
        
        return None
    
    def _preprocess_price_data(self, data: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        Preprocess price data for a single stock
        
        Args:
            data (pd.DataFrame): Raw price data
            stock_code (str): Stock code
            
        Returns:
            pd.DataFrame: Preprocessed price data
        """
        # Standardize column names
        column_mapping = {
            'Date': 'date', 'DATE': 'date',
            'Open': 'open', 'OPEN': 'open',
            'High': 'high', 'HIGH': 'high',
            'Low': 'low', 'LOW': 'low',
            'Close': 'close', 'CLOSE': 'close',
            'Volume': 'volume', 'VOLUME': 'volume',
            'Adj Close': 'adj_close', 'ADJ_CLOSE': 'adj_close'
        }
        
        data = data.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['date', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns for {stock_code}: {missing_columns}")
        
        # Convert date column
        data['date'] = pd.to_datetime(data['date'], format=self.config.data.date_format, errors='coerce')
        
        # Remove rows with invalid dates
        data = data.dropna(subset=['date'])
        
        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)
        
        # Convert price columns to numeric
        price_columns = ['open', 'high', 'low', 'close']
        if 'volume' in data.columns:
            price_columns.append('volume')
        if 'adj_close' in data.columns:
            price_columns.append('adj_close')
        
        for col in price_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].astype(np.float32)
        
        # Remove rows with missing price data
        data = data.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Validate OHLC data
        data = self._validate_ohlc_data(data, stock_code)
        
        # Handle missing volume data
        if 'volume' not in data.columns:
            data['volume'] = 0
        else:
            data['volume'] = data['volume'].fillna(0)
        
        # Add stock code column
        data['stock_code'] = stock_code
        
        return data
    
    def _validate_ohlc_data(self, data: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        Validate OHLC data consistency
        
        Args:
            data (pd.DataFrame): Price data
            stock_code (str): Stock code
            
        Returns:
            pd.DataFrame: Validated price data
        """
        # Check for invalid OHLC relationships
        invalid_mask = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close']) |
            (data['open'] <= 0) |
            (data['high'] <= 0) |
            (data['low'] <= 0) |
            (data['close'] <= 0)
        )
        
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            logger.warning(f"Found {invalid_count} invalid OHLC rows for {stock_code}, removing them")
            data = data[~invalid_mask].reset_index(drop=True)
        
        return data
    
    def _filter_by_date(self, data: pd.DataFrame, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Filter data by date range
        
        Args:
            data (pd.DataFrame): Price data
            start_date (str, optional): Start date
            end_date (str, optional): End date
            
        Returns:
            pd.DataFrame: Filtered data
        """
        if start_date:
            start_date = pd.to_datetime(start_date)
            data = data[data['date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            data = data[data['date'] <= end_date]
        
        return data.reset_index(drop=True)
    
    def get_latest_data(self, stock_codes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Get the latest data for specified stocks
        
        Args:
            stock_codes (List[str], optional): List of stock codes
            
        Returns:
            Dict[str, pd.DataFrame]: Latest data for each stock
        """
        if stock_codes is None:
            stock_codes = list(self.price_data.keys())
        
        latest_data = {}
        
        for stock_code in stock_codes:
            if stock_code in self.price_data:
                latest_data[stock_code] = self.price_data[stock_code].copy()
            else:
                # Try to load fresh data
                data = self._load_single_stock_data(stock_code)
                if data is not None:
                    latest_data[stock_code] = data
                    self.price_data[stock_code] = data
        
        return latest_data
    
    def split_data(self, data: Optional[Dict[str, pd.DataFrame]] = None,
                   train_ratio: Optional[float] = None,
                   validation_ratio: Optional[float] = None,
                   random_state: int = 42) -> Tuple[Dict[str, pd.DataFrame], 
                                                   Dict[str, pd.DataFrame], 
                                                   Dict[str, pd.DataFrame]]:
        """
        Split data into training, validation, and testing sets
        
        Args:
            data (Dict[str, pd.DataFrame], optional): Data to split
            train_ratio (float, optional): Training data ratio
            validation_ratio (float, optional): Validation data ratio
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple: (train_data, validation_data, test_data)
        """
        if data is None:
            data = self.price_data
        
        if train_ratio is None:
            train_ratio = self.config.data.train_ratio
        if validation_ratio is None:
            validation_ratio = self.config.data.validation_ratio
        
        test_ratio = 1.0 - train_ratio - validation_ratio
        
        train_data = {}
        validation_data = {}
        test_data = {}
        
        for stock_code, stock_data in data.items():
            try:
                # Time-based split (more realistic for time series)
                if self.config.data.test_ratio > 0:
                    train_val_data, test_stock_data = self._time_based_split(
                        stock_data, 1.0 - test_ratio
                    )
                else:
                    train_val_data = stock_data
                    test_stock_data = pd.DataFrame()
                
                # Split training and validation
                if validation_ratio > 0 and len(train_val_data) > 0:
                    val_ratio = validation_ratio / (train_ratio + validation_ratio)
                    train_stock_data, val_stock_data = self._time_based_split(
                        train_val_data, 1.0 - val_ratio
                    )
                else:
                    train_stock_data = train_val_data
                    val_stock_data = pd.DataFrame()
                
                # Store splits if they have minimum required data
                if len(train_stock_data) >= self.config.data.min_data_points:
                    train_data[stock_code] = train_stock_data
                
                if len(val_stock_data) >= 10:  # Minimum validation size
                    validation_data[stock_code] = val_stock_data
                
                if len(test_stock_data) >= 10:  # Minimum test size
                    test_data[stock_code] = test_stock_data
                
            except Exception as e:
                logger.warning(f"Error splitting data for {stock_code}: {e}")
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(validation_data)}, Test: {len(test_data)}")
        
        return train_data, validation_data, test_data
    
    def _time_based_split(self, data: pd.DataFrame, split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data based on time (chronological split)
        
        Args:
            data (pd.DataFrame): Data to split
            split_ratio (float): Ratio for first split
            
        Returns:
            Tuple: (first_split, second_split)
        """
        split_idx = int(len(data) * split_ratio)
        return data.iloc[:split_idx].copy(), data.iloc[split_idx:].copy()
         
     
    def _scale_features(self, data: pd.DataFrame, stock_code: str, fit_scaler: bool) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            data (pd.DataFrame): Feature data
            stock_code (str): Stock code
            fit_scaler (bool): Whether to fit new scaler
            
        Returns:
            pd.DataFrame: Scaled data
        """
        # Identify numerical columns to scale
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude certain columns from scaling
        exclude_columns = ['stock_code', 'date']
        numerical_columns = [col for col in numerical_columns if col not in exclude_columns]
        
        if not numerical_columns:
            return data
        
        # Get or create scaler
        if stock_code not in self.scalers:
            self.scalers[stock_code] = StandardScaler()
        
        scaler = self.scalers[stock_code]
        
        # Scale the features
        scaled_data = data.copy()
        
        if fit_scaler:
            scaled_data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        else:
            scaled_data[numerical_columns] = scaler.transform(data[numerical_columns])
        
        return scaled_data
    
    def save_processed_data(self, data: Dict[str, pd.DataFrame], filename: str):
        """
        Save processed data to file
        
        Args:
            data (Dict[str, pd.DataFrame]): Data to save
            filename (str): Output filename
        """
        try:
            output_path = f"processed_data/{filename}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as pickle for efficient loading
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Processed data saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def load_processed_data(self, filename: str) -> Dict[str, pd.DataFrame]:
        """
        Load processed data from file
        
        Args:
            filename (str): Input filename
            
        Returns:
            Dict[str, pd.DataFrame]: Loaded data
        """
        try:
            input_path = f"processed_data/{filename}"
            
            import pickle
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Processed data loaded from {input_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of loaded data
        
        Returns:
            Dict[str, Any]: Summary statistics
        """
        summary = {
            'total_stocks': len(self.price_data),
            'date_range': {},
            'data_points': {},
            'missing_data': {}
        }
        
        if self.price_data:
            all_start_dates = []
            all_end_dates = []
            total_data_points = 0
            
            for stock_code, data in self.price_data.items():
                start_date = data['date'].min()
                end_date = data['date'].max()
                data_points = len(data)
                
                all_start_dates.append(start_date)
                all_end_dates.append(end_date)
                total_data_points += data_points
                
                summary['data_points'][stock_code] = data_points
            
            summary['date_range']['earliest_start'] = min(all_start_dates)
            summary['date_range']['latest_end'] = max(all_end_dates)
            summary['total_data_points'] = total_data_points
            summary['average_data_points'] = total_data_points / len(self.price_data)
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # For testing purposes
    from config import Config
    
    # Initialize with configuration
    config = Config("test_config.json")
    data_manager = DataManager(config)
    
    # Load stock list
    stock_list = data_manager.load_stock_list()
    print(f"Loaded {len(stock_list)} stocks")
    
    # Load price data (first 10 stocks for testing)
    test_stocks = stock_list['code'].head(10).tolist()
    price_data = data_manager.load_price_data(test_stocks)
    
    # Get data summary
    summary = data_manager.get_data_summary()
    print("Data Summary:", json.dumps(summary, indent=2, default=str))
    
    # Split data
    train_data, val_data, test_data = data_manager.split_data()
    print(f"Split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")