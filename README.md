# Cryptocurrency Price Prediction

## Project Overview
This project explores deep learning techniques for cryptocurrency price prediction through eleven experiments, starting from basic LSTM models and progressing to more sophisticated architectures. Several advanced approaches, including the SSA optimizer and transformer-like architectures, showed promise but were not fully explored due to time constraints and debugging challenges.

## Experiments

### Experiment 1: Initial LSTM Test
- **Dataset**: Bitcoin Historical Data (Kaggle)
- **Model**: Simple LSTM
- **Issues**: Data leakage in train-test split, hardcoded parameters, insufficient preprocessing
- **Outcome**: Limited plots and information

### Experiment 2: Basic Fixes
- Fixed data leakage issues from Experiment 1
- Improved code structure
- Same dataset and model architecture

### Experiment 3: Comprehensive Improvements
- **New Datasets**: ETH/USD and BTC/USD comprehensive data
- **Improvements**:
  - Better data preprocessing
  - Proper logging
  - Enhanced plotting
  - Train/validation/test split
  - Combined index implementation

### Experiment 4: Enhanced LSTM
Key additions:
1. Attention Mechanism
2. Multiple LSTM Layers
3. Skip Connections
4. Layer Normalization
5. Xavier Initialization
6. Dropout Implementation
7. Advanced Prediction Head

### Experiment 5: RobustScaler Implementation
- Replaced MinMaxScaler with RobustScaler
- Same LSTM architecture as Experiment 4

### Experiment 6: StandardScaler and Log Transform
- Implemented StandardScaler
- Added Log Transformation
- Same LSTM architecture

### Experiment 7: SSA Optimizer Test
- Replaced Adam with custom SSA optimizer
- Results: Slow convergence, poor performance
- Not pursued further
- Initial idea from MS-SSA-LSTM methodology presented in ["A Stock Price Prediction Model Based on Investor Sentiment and Optimized Deep Learning"](https://ieeexplore.ieee.org/document/10130578) (IEEE 2023)

### Experiment 8: Advanced Architecture
New components:
- Positional Encoding
- Multi-Head Self Attention
- Gated Linear Unit (GLU)
- Temporal Convolution Network
- Enhanced processing pipeline

### Experiment 9: RobustScaler Retest
- Applied RobustScaler to Experiment 8 architecture
- Results remained suboptimal

### Experiment 10: Multi-step Prediction Test
- Used LSTM from Experiment 8
- Attempted multi-step prediction
- Poor results

### Experiment 11: Refined Multi-step Prediction
- Reverted to Experiment 6 LSTM architecture
- Maintained StandardScaler
- Focus on multi-step prediction

## Data Sources
- Experiments 1-2: [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
- Experiments 3-11: 
  - [ETH/USD Data](https://www.kaggle.com/datasets/imranbukhari/comprehensive-ethusd-1m-data)
  - [BTC/USD Data](https://www.kaggle.com/datasets/imranbukhari/comprehensive-btcusd-1m-data)

## Results
