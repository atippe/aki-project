# Cryptocurrency price prediction

## Project overview
This project explores deep learning techniques for cryptocurrency price prediction through eleven experiments, starting from basic LSTM models and progressing to more complex architectures. Key findings show:
- Simpler models often performed better than complex ones
- Best results came from basic LSTM with StandardScaler and Log Transformation
- ETH/USD predictions were generally more accurate than BTC/USD
- Complex architectures and advanced optimizers (like SSA) often led to worse results
- Multi-step predictions worked but accuracy decreased with longer forecasts

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/MaximilianPowers/Thesis
   ```

2. Set up data:
   - Run the data downloader script first:
     ```bash
     python data/download_datasets.py
     ```
   - This will download:
     - For Experiments 1-2: Old Bitcoin dataset from Kaggle
     - For Experiments 3-11: New comprehensive BTC/USD and ETH/USD data

3. For each experiment:
   1. Process the data first:
      ```bash
      python experiment_X/dataset_processor.py
      ```
   2. Then run the main script:
      ```bash
      python experiment_X/main.py
      ```
   
Note: Replace 'X' with the experiment number (1-11). Always process the dataset before running the main script for each experiment.

## Experiments

### Experiment 1: Initial LSTM test
- First try at using LSTM for Bitcoin price prediction
- Very basic setup with many mistakes
- Main issue: data leakage due to wrong scaling order
- Good learning experience but not reliable results

### Experiment 2: Basic fixes
- Fixed the main data leakage problem from Experiment 1
- Cleaned up the code structure
- Still basic but now methodologically correct
- Same simple LSTM model as before

### Experiment 3: Comprehensive improvements
- Added more data (ETH/USD alongside BTC/USD)
- Better organization of code and data handling
- Added proper logging to track results
- Better plots to understand what's happening
- Proper split into train/validation/test sets

### Experiment 4: Enhanced LSTM
- Added several improvements to make the LSTM stronger:
  - Attention mechanism to focus on important parts
  - Multiple LSTM layers for deeper learning
  - Skip connections to help with training
  - Dropout to prevent overfitting
  - Better weight initialization
- More complex but potentially more powerful

### Experiment 5: RobustScaler implementation
- Same model as Experiment 4
- Changed to RobustScaler instead of MinMaxScaler
- Goal: Handle extreme price movements better
- Testing if different scaling helps with predictions

### Experiment 6: StandardScaler and Log Transform
- Tried another scaling approach with StandardScaler
- Added log transformation for price data
- Kept the same LSTM structure
- Looking for better ways to handle price variations

### Experiment 7: SSA Optimizer test
- Tried a new optimization method (SSA) instead of Adam
- Based on recent research paper
- Didn't work well: slow and poor results
- Decided not to continue with this approach

### Experiment 8: Advanced architecture
- Built a more modern version with new features:
  - Position encoding to handle time better
  - Self-attention to spot patterns
  - New types of layers (GLU, TCN)
  - Better data processing
- Most complex version so far

### Experiment 9: RobustScaler retest
- Used the complex model from Experiment 8
- Tried RobustScaler again to handle outliers
- Results weren't great
- Showed that more complex isn't always better

### Experiment 10: Multi-step Prediction Test
- First try at predicting multiple days ahead
- Used the complex model from Experiment 8
- Tried to predict 5 days into the future
- Results weren't satisfactory

### Experiment 11: Refined Multi-step Prediction
- Went back to simpler model from Experiment 6
- Kept the multi-day prediction (5 days ahead)
- Used StandardScaler and simpler LSTM
- Trying to find balance between complexity and accuracy

## Data Sources
- Experiments 1-2: [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
- Experiments 3-11: 
  - [ETH/USD Data](https://www.kaggle.com/datasets/imranbukhari/comprehensive-ethusd-1m-data)
  - [BTC/USD Data](https://www.kaggle.com/datasets/imranbukhari/comprehensive-btcusd-1m-data)

## Overall conclusion from experiments 1-11

- Simple architectures often outperformed complex ones, with the best results achieved in Experiment 6 using StandardScaler with Log Transformation (R² = 0.99 for BTC/USD)
- ETH/USD predictions consistently showed better performance than BTC/USD across most experiments, likely due to its lower price volatility
- RobustScaler showed mixed results, performing well with ETH but struggling with BTC's wider price ranges
- Multi-step predictions demonstrated degrading accuracy with longer time horizons, though maintaining acceptable performance
- Over-engineered solutions (like Experiment 8's complex architecture) often led to decreased performance
- The Sparrow Search Algorithm (SSA) optimizer implementation proved ineffective and computationally expensive

## Key lessons learned

- Easy to make fatal mistakes like incorrect data scaling timing
- Proper train-test splits are crucial before any data transformation
- Small oversights can lead to misleading results

- Proper logging is essential for tracking and comparing experiments
- Clear documentation and visualization of results are crucial
- Consistent evaluation metrics are needed across experiments

