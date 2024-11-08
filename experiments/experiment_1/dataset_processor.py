import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib
import sys
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


class BasicDataProcessor:
    def __init__(self, raw_data_path):
        self.raw_data_path = Path(raw_data_path)
        self.scaler = MinMaxScaler()
        self.processed_dir = Path(project_root / 'experiments/experiment_1/processed_data')

    def process(self, start_date='2022-01-01', resample_rule='1h'):
        """
        Process the data with options to limit date range and resample

        Args:
            start_date (str): Start date for data (e.g., '2022-01-01')
            resample_rule (str): Resample frequency ('1h' for hourly, '4h' for 4 hours)
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        print("Loading data...")
        df = pd.read_csv(self.raw_data_path)

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Timestamp', inplace=True)

        print(f"Filtering data from {start_date}...")
        df = df[df.index >= start_date]

        print(f"Resampling data to {resample_rule} intervals...")
        df = df.resample(resample_rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })

        # Remove any NaN values
        df = df.dropna()

        print("Scaling data...")
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        scaled_data = self.scaler.fit_transform(df[features])
        df_scaled = pd.DataFrame(scaled_data, columns=features, index=df.index)

        # train-test split (80-20), info: this splits after scaling which causes data leakage
        train_size = int(len(df_scaled) * 0.8)
        train_data = df_scaled[:train_size]
        test_data = df_scaled[train_size:]

        print("Saving processed data...")
        self.save_processed_data(train_data, test_data, df)

        return {
            'train_data': train_data,
            'test_data': test_data,
            'scaler': self.scaler,
            'original_data': df
        }

    def save_processed_data(self, train_data, test_data, original_data):
        train_data.to_csv(self.processed_dir / 'train_data.csv')
        test_data.to_csv(self.processed_dir / 'test_data.csv')
        original_data.to_csv(self.processed_dir / 'original_data.csv')
        joblib.dump(self.scaler, self.processed_dir / 'scaler.pkl')
        print(f"Saved processed data to {self.processed_dir}")


def main():
    processor = BasicDataProcessor(project_root / 'data/raw/bitcoin-historical-data/btcusd_1-min_data.csv')

    # Process only recent data (2022+) and resample to 4-hour intervals
    processed_data = processor.process(
        start_date='2022-01-01',  # Last ~2 years of data
        resample_rule='4h'  # 4-hour intervals
    )

    print("\nData Shape Summary:")
    print(f"Training data shape: {processed_data['train_data'].shape}")
    print(f"Testing data shape: {processed_data['test_data'].shape}")

    print("\nDate Range:")
    print(f"Start: {processed_data['original_data'].index.min()}")
    print(f"End: {processed_data['original_data'].index.max()}")

    print("\nFirst few rows of processed data:")
    print(processed_data['train_data'].head())

    print("\nFirst few rows of original data:")
    print(processed_data['original_data'].head())


if __name__ == "__main__":
    main()
