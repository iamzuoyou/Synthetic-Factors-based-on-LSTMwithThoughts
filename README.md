# Synthetic-Factors-based-on-LSTMwithThoughts

## Quantitative Stock Selection: Synthetic Factors Based on Deep Learning

## Introduction
Our project explores the use of deep learning to create synthetic factors for quantitative stock selection. Traditional methods and time-series models are compared with our approach, which leverages synthetic factors to enhance stock selection strategies.

## Data Input

### Data Sources and Processing
- Data from the Shanghai Main Board and STAR Market from January 2020 to December 2024.
- Negative P/E ratios are replaced with zero.
- Stocks listed after January 1, 2020, are excluded.
- Missing values are appropriately filled.
- The final dataset includes 1,120 stocks.
- Data is sourced from the Tushare database, including A-share adjustment factors, daily basic indicators, daily, weekly, and monthly market data, and technical factors.

### Features
- 101 technical factors, including ASI, BBI, MACD, KDJ, momentum and directional indicators, and emotional indicators like BRAR.
- Adjustment factors are forward-adjusted to account for magnitude issues.

### Training Labels
- Future one-month returns of individual stocks.
- Standardized across individual stocks each month.

### Training Features
- Each sample: a 2D array of time steps * number of features.
- Lookback period: 22 days.
- Number of features: 101.
- Final training samples: a 3D array of sample number * time steps * number of features.

## Training Process

### LSTM with Thoughts Module
- Introduces a "thought vector" generated from hidden states at each time step.
- Combines all thought vectors into a comprehensive representation.
- Integrates this representation with the final hidden state for prediction.

```
# Core Code of Thoughts Module
class LSTMWithThoughts(tf.keras.Model):
    def __init__(self, hidden_dim, thought_dim, num_thoughts, dropout_rate, output_dim=1):
        super(LSTMWithThoughts, self).__init__()
        self.hidden_dim = hidden_dim
        self.thought_dim = thought_dim
        self.num_thoughts = num_thoughts
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        
        # Define layers
        self.batchnorm = BatchNormalization()
        self.encoder = LSTM(hidden_dim, return_sequences=False)
        self.thought_generator = Dense(num_thoughts * thought_dim)
        self.mlp_combine_thoughts = tf.keras.Sequential([
            Dense(thought_dim, activation='relu'),
            Dense(thought_dim)
        ])
        self.dropout = Dropout(dropout_rate)
        self.output_layer = Dense(output_dim)}
```    
### Training Steps
1. Load data and create samples.
2. Use multiprocessing to accelerate sample creation.
3. Split data into training and validation sets chronologically.
4. Set random seeds for reproducibility.
5. Shuffle samples for training and validation.
6. Save model parameters after each epoch.
7. Use custom IC (Information Coefficient) as the loss function.

### Hyperparameters
- Hidden units: 64
- Thought dimension: 32
- Number of thoughts: 5
- Dropout rate: 0.2
- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 256
- Epochs: 10

### Rebalancing
- Training uses data from the past three years.
- Training set: three years to six months before rebalancing.
- Validation set: six months to one month before rebalancing.
- Monthly sampling to avoid excessive overlap.

## Preliminary Evaluation

### Backtesting Metrics
- Cumulative Returns: Based on predicted returns, select the top 10 stocks each month and calculate actual cumulative returns.
- IC (Information Coefficient): Pearson correlation between predicted and actual returns.
- Rank IC: Spearman rank correlation between predicted and actual returns.
- ICIR (Information Coefficient Information Ratio): Mean IC divided by its standard deviation.

### Backtesting Results
- Synthetic factors are tested individually.
- IC above 5% is effective; above 10% is excellent.
- Rank IC above 5% is effective; above 10% is excellent.
- ICIR above 0.5 indicates good predictive power; above 1 indicates strong and stable predictive power.

## Result Analysis

### Single Factor Layered Test
- Predict returns for all stocks and select the top 10.
- Use Monte Carlo simulation to generate random weight combinations following an exponential distribution.

### Mean-Variance Optimization
- Based on historical monthly returns from the past year, plot the efficient frontier.
- Identify the minimum risk portfolio (purple) and the maximum Sharpe ratio portfolio (blue).

### Out-of-Sample Results (2024)
- Quarterly results for 2024 show consistent performance.

### Summary of Investment Strategy Performance (2023-2024)
- The model shows good generalization and robustness.
- No overfitting observed; performance remains strong in out-of-sample tests.

## Limitations and Improvements

### Limitations
- Data quality needs verification.
- Potential issues with raw data processing.

### Improvements
- Test and filter single factors to remove highly correlated and ineffective ones.
- Optimize data preprocessing, especially missing value handling.
- Consider transaction costs in portfolio weighting.
- Enhance model robustness by averaging predictions from multiple random seeds.

## Conclusion
The model demonstrates strong potential for real-world application, with excellent out-of-sample performance in 2024. The approach shows promise for future quantitative stock selection strategies.

Thank you for reading!

### Great thanks to our Main Contributers!!!
- Guang Yang: https://github.com/sjtuyg
- Zuoyou Jiang: https://github.com/iamzuoyou
- Zhenliang Xiong: https://github.com/ericxiong0331
- Qianxue Shi: https://github.com/Lizzzz7
