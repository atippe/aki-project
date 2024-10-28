import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib


class BasicDataProcessor:
    def __init__(self, raw_data_path):
        self.raw_data_path = Path(raw_data_path)
        self.scaler = MinMaxScaler()
        self.processed_dir = Path('processed/bitcoin-historical-data')

    def process(self):
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.raw_data_path)

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Timestamp', inplace=True)

        df = df.dropna()  # Remove any NaN values

        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        scaled_data = self.scaler.fit_transform(df[features])
        df_scaled = pd.DataFrame(scaled_data, columns=features, index=df.index)

        # train-test split (80-20)
        train_size = int(len(df_scaled) * 0.8)
        train_data = df_scaled[:train_size]
        test_data = df_scaled[train_size:]

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
    processor = BasicDataProcessor('raw/bitcoin-historical-data/btcusd_1-min_data.csv')
    processed_data = processor.process()

    print("\nData Shape Summary:")
    print(f"Training data shape: {processed_data['train_data'].shape}")
    print(f"Testing data shape: {processed_data['test_data'].shape}")

    print("\nFirst few rows of processed data:")
    print(processed_data['train_data'].head())

    print("\nFirst few rows of original data:")
    print(processed_data['original_data'].head())


if __name__ == "__main__":
    main()
