# Stock Price Trend Prediction System

A machine learning-based system for predicting stock price trends and trend reversal points using neural networks. The system analyzes daily stock price data for approximately 4000 Japanese stocks and provides both short-term and long-term trend predictions with confidence ratings.

## 🎯 Main Goal

Create a system to anticipate the coming stock price trends and the next trend reversal price, with the option to update its reasoning algorithm on the user's trigger.

## 📊 Features

- **Neural Network Training**: Train ML models on actual stock market data
- **Dual Trend Analysis**: Short-term (5-day) and long-term (25-day) trend predictions
- **Reversal Price Prediction**: Predict the price at which trends will reverse
- **Confidence Ratings**: Quantified confidence levels for all predictions
- **Model Updates**: User-triggered algorithm updates and retraining
- **Data Flexibility**: Multiple data splitting strategies for optimal training
- **Batch Processing**: Analyze thousands of stocks efficiently

## 🏗️ System Architecture

```
stock-prediction-system/
├── main.py                 # Main entry point and CLI
├── config.py               # Configuration management
├── data_manager.py         # Data loading and preprocessing
├── trend_analyzer.py       # Moving averages and trend calculations
├── model_trainer.py        # Neural network training
├── predictor.py            # Prediction and confidence calculation
├── config.json             # System configuration file
├── data/                   # Data directory
│   ├── stock_list.csv      # List of ~4000 stocks
│   └── price_data/         # Daily OHLC price data
├── models/                 # Trained ML models
├── output/                 # Prediction results
└── logs/                   # System logs
```

## 📈 Input Data

- **Daily Stock Price Data**: Open, High, Low, Close (OHLC) prices
- **Company Information**: Stock company field of business
- **Stock List**: Complete list of ~4000 Japanese stocks with industry classifications
- **Historical Data**: Files in the "data" directory

## 📉 Output Data

For each stock, the system provides:

| Output | Description |
|--------|-------------|
| **Short Trend** | +/- slope of 5-day moving average |
| **Long Trend** | +/- slope of 25-day moving average |
| **Short Reversal Price** | Predicted peak/trough price for 5-day trend reversal |
| **Long Reversal Price** | Predicted peak/trough price for 25-day trend reversal |
| **Short Confidence** | Confidence rating (0-100%) for short-term predictions |
| **Long Confidence** | Confidence rating (0-100%) for long-term predictions |

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-prediction-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up data directory**
   ```bash
   mkdir -p data/price_data models output logs
   ```

4. **Configure the system**
   ```bash
   cp config.json.example config.json
   # Edit config.json with your preferences
   ```

## 🚀 Usage

### Command Line Interface

####docker 権限
docker exec -u root <CONTAINER_ID> chown -R appuser:appuser /app/data /app/output /app/models /app/cache /app/logs /app/processed_data

docker exec -u root a7f520272d83 chown -R appuser:appuser /app/data /app/output /app/models /app/cache /app/logs /app/processed_data
a7f520272d83
#### 1. Train Models
Train neural networks on your stock data:
```bash
python main.py --mode train
```

#### 2. Generate Predictions
Predict trends for all stocks:
```bash
python main.py --mode predict --output predictions.json
```

Predict trends for specific stocks:
```bash
python main.py --mode predict --stocks 73140 73170 73180 --output selected_predictions.json
```

#### 3. Update Models
Retrain models with latest data:
```bash
python main.py --mode update
```

#### 4. Full Pipeline
Run complete training and prediction pipeline:
```bash
python main.py --mode full
```

### Python API Usage

```python
from main import StockPredictionSystem

# Initialize system
system = StockPredictionSystem("config.json")

# Load and prepare data
stock_list, price_data = system.load_data()
train_data, test_data = system.prepare_training_data(price_data)

# Train models
system.train_models(train_data)

# Generate predictions
predictions = system.make_predictions(['73140', '73170'])

# Update models with new data
metrics = system.update_models()
```

## 📋 Configuration

Edit `config.json` to customize system behavior:

```json
{
  "data": {
    "stock_list_path": "data/stock_list.csv",
    "price_data_path": "data/price_data/",
    "train_ratio": 0.8
  },
  "model": {
    "short_term_window": 5,
    "long_term_window": 25,
    "hidden_layers": [64, 32, 16],
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32
  },
  "prediction": {
    "confidence_threshold": 0.7,
    "update_frequency": "daily"
  }
}
```

## 📊 Output Format

Predictions are saved as JSON with the following structure:

```json
{
  "73140": {
    "short_trend": "+",
    "long_trend": "-",
    "short_reversal_price": 1234.56,
    "long_reversal_price": 1189.23,
    "short_confidence": 85.2,
    "long_confidence": 78.9,
    "timestamp": "2025-09-12T10:30:00"
  },
  "73170": {
    "short_trend": "-",
    "long_trend": "+",
    "short_reversal_price": 2341.78,
    "long_reversal_price": 2456.12,
    "short_confidence": 92.1,
    "long_confidence": 68.7,
    "timestamp": "2025-09-12T10:30:00"
  }
}
```

## 🧠 Technical Definitions

### Trend Definition
- **Short Trend**: Plus/minus of the slope of the 5-day running price average line
- **Long Trend**: Plus/minus of the slope of the 25-day running price average line

### Trend Reversal Price
- **Short Reversal Price**: Price at the peak/trough of the 5-day moving average curve when trend switches from upward to downward (or vice versa)
- **Long Reversal Price**: Price at the peak/trough of the 25-day moving average curve when trend switches from upward to downward (or vice versa)

### Credit Rating
Numerical confidence score (0-100%) indicating how confident the model is about each prediction, based on:
- Historical prediction accuracy
- Data quality and completeness
- Market volatility indicators
- Model uncertainty quantification

## 🔄 Model Updates

The system supports user-triggered model updates:

1. **Manual Update**: Run `python main.py --mode update`
2. **Scheduled Updates**: Configure automatic updates in `config.json`
3. **API Updates**: Call `system.update_models()` programmatically

## 📁 Data Requirements

### Stock List Format (CSV)
```csv
date,code,name,english_name,sector_id,sector_name,industry_code,industry_name,index,market_code,market_name
2025-05-01,73140,小田原機器,"ODAWARA AUTO-MACHINE MFG.CO.,LTD.",6,自動車・輸送機,3700,輸送用機器,-,0112,スタンダード
```

### Price Data Format (CSV per stock)
```csv
date,open,high,low,close,volume
2025-01-01,1234.0,1245.0,1230.0,1240.0,1000000
2025-01-02,1240.0,1250.0,1235.0,1248.0,1200000
```

## 🚨 Calculation Trigger

When the program is run, the output is renewed and recalculated using:
1. Latest trained model
2. Updated input data
3. Current market conditions
4. Fresh technical indicators

## 🛠️ Development

### Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes in appropriate module
3. Add tests and documentation
4. Submit pull request

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run performance tests
python -m pytest tests/performance/
```

### Logging
The system uses Python's logging module. Log files are stored in the `logs/` directory:
- `system.log`: General system operations
- `training.log`: Model training progress
- `predictions.log`: Prediction generation logs
- `errors.log`: Error and exception logs

## 📦 Dependencies

- `numpy`: Numerical computations
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning utilities
- `tensorflow/keras`: Neural network framework
- `matplotlib`: Data visualization
- `seaborn`: Statistical visualization
- `joblib`: Model serialization
- `configparser`: Configuration management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This system is for educational and research purposes. Stock market predictions are inherently uncertain, and this system should not be used as the sole basis for investment decisions. Always consult with financial professionals and conduct your own research before making investment choices.

## 📞 Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Check the documentation in the `docs/` folder
- Review existing issues and discussions

---

**Happy Predicting! 📈**