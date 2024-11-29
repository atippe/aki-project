import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import pickle
import sys
import torch
import logging
import time

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


class BasicDataProcessor:
    def __init__(self, raw_data_path, subfolder=None):
        self.raw_data_path = Path(raw_data_path)
        self.scaler = RobustScaler()
        self.processed_dir = Path(Path(__file__).parent / 'processed_data')
        if subfolder:
            self.processed_dir = self.processed_dir / subfolder
        self.setup_logging()


    def setup_logging(self):
        """Setup logging configuration"""
        logging.getLogger().handlers.clear()
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.processed_dir / 'processing.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def validate_features(self, df, features, target_feature):
        """
        Validate that all required features exist in the dataset

        Args:
            df: DataFrame to validate
            features: List of features to check
            target_feature: Target feature to check
        """
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            self.logger.error(f"Missing required features: {missing_features}")
            self.logger.error(f"Available features: {df.columns.tolist()}")
            raise ValueError(f"Dataset missing required features: {missing_features}")
        if target_feature not in df.columns:
            self.logger.error(f"Target feature '{target_feature}' not found in dataset")
            self.logger.error(f"Available features: {df.columns.tolist()}")
            raise ValueError(f"Dataset missing target feature: {target_feature}")

        self.logger.info("\nFeature validation:")
        self.logger.info(f"Required features: {features}")
        self.logger.info(f"Target feature: {target_feature}")
        self.logger.info("All required features found in dataset")

        self.logger.info("\nFeature statistics:")
        for feature in features:
            stats = df[feature].describe()
            self.logger.info(f"\n{feature}:")
            self.logger.info(f"  Min: {stats['min']:.2f}")
            self.logger.info(f"  Max: {stats['max']:.2f}")
            self.logger.info(f"  Mean: {stats['mean']:.2f}")
            self.logger.info(f"  Std: {stats['std']:.2f}")

    def create_sequences(self, data, seq_length=60, target_feature='Close'):
        """
        Create sequences for time series prediction

        Args:
            data (DataFrame): Scaled data with OHLCV columns
            seq_length (int): Length of the sequence
            target_feature (str): Feature to use as target for prediction

        Returns:
            tuple: (sequences array, targets array)
        """
        sequences = []
        targets = []

        # Convert DataFrame to numpy array for faster processing
        data_array = data.values
        target_idx = data.columns.get_loc(target_feature)

        self.logger.info(f"\nSequence creation details:")
        self.logger.info(f"Target feature: {target_feature} (index: {target_idx})")
        self.logger.info(f"Available features: {list(data.columns)}")
        self.logger.info(f"Sequence length: {seq_length}")

        for i in range(len(data_array) - seq_length):
            sequence = data_array[i:(i + seq_length)]
            target = data_array[i + seq_length][target_idx]
            sequences.append(sequence)
            targets.append(target)

        sequences = np.array(sequences)
        targets = np.array(targets)

        self.logger.info(f"Created {len(sequences)} sequences")
        self.logger.info(f"Target value range: min={targets.min():.4f}, max={targets.max():.4f}")

        return sequences, targets

    def process(self, start_date='2022-01-01', resample_rule='1h', agg_dict=None, seq_length=60, features=None, target_feature=None, split_ratios=(0.7, 0.15, 0.15)):
        """
        Process the data and create PyTorch tensors
        Args:
            start_date (str): Start date for data (e.g., '2022-01-01')
            resample_rule (str): Resample frequency ('1h' for hourly, '4h' for 4 hours)
            agg_dict (dict): Custom aggregation rules for resampling, e.g., {'Open': 'first', 'Custom': 'mean'}
            seq_length (int): Length of the sequence
            features (list): List of features to use, defaults to OHLCV
            target_feature (str): Target feature to predict
            split_ratios (tuple): (train, val, test) ratios that sum to 1.0
        """
        start_time = time.time()

        if sum(split_ratios) != 1.0:
            raise ValueError("Split ratios must sum to 1.0")

        self.logger.info(f"\n{'=' * 50}")
        self.logger.info("TIME SERIES DATA PROCESSING")
        self.logger.info(f"{'=' * 50}")

        self.logger.info("\nLoading data from:")
        self.logger.info(f"File: {self.raw_data_path.name}")
        df = pd.read_csv(self.raw_data_path)

        if features is None:
            self.logger.info("\nNo features specified, using default OHLCV")
            #features = ['Open', 'High', 'Low', 'Close', 'Volume'] # For old dataset
            features = ['Volume', 'Open', 'High', 'Low', 'Close']

        if target_feature is None:
            self.logger.info("\nNo target feature specified, using default 'Close'")
            target_feature = 'Close'

        self.validate_features(df, features, target_feature)

        # For old dataset with unix timestamp
        #df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        #df.set_index('Timestamp', inplace=True)

        df['Open time'] = pd.to_datetime(df['Open time'])
        df.set_index('Open time', inplace=True)

        initial_size = len(df)
        self.logger.info(f"\nInitial data size: {initial_size} entries")

        self.logger.info(f"\nFiltering data from {start_date}...")
        df = df[df.index >= start_date]
        self.logger.info(f"Remaining entries after filtering: {len(df)}")

        self.logger.info(f"\nResampling data to {resample_rule} intervals...")

        if agg_dict is None:
            self.logger.info("\nNo aggregation rules specified, using default for OHLCV")
            """agg_dict = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }"""
            agg_dict = {
                'Volume': 'sum',
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }

        self.logger.info(f"Using aggregation methods: {agg_dict}")
        df = df.resample(resample_rule).agg(agg_dict)

        df = df.dropna()
        self.logger.info(f"Final entries after resampling: {len(df)}")

        # Split data into train, validation, and test sets
        train_size = int(len(df) * split_ratios[0])
        val_size = int(len(df) * split_ratios[1])

        train_data = df[:train_size]
        val_data = df[train_size:train_size + val_size]
        test_data = df[train_size + val_size:]

        self.logger.info("\nData split:")
        self.logger.info(f"Training set: {len(train_data)} entries ({split_ratios[0] * 100}%)")
        self.logger.info(f"Validation set: {len(val_data)} entries ({split_ratios[1] * 100}%)")
        self.logger.info(f"Test set: {len(test_data)} entries ({split_ratios[2] * 100}%)")

        self.logger.info("\nScaling data...")
        self.scaler.fit(train_data[features])

        train_scaled = pd.DataFrame(
            self.scaler.transform(train_data[features]),
            columns=features,
            index=train_data.index
        )
        val_scaled = pd.DataFrame(
            self.scaler.transform(val_data[features]),
            columns=features,
            index=val_data.index
        )
        test_scaled = pd.DataFrame(
            self.scaler.transform(test_data[features]),
            columns=features,
            index=test_data.index
        )

        self.logger.info(f"\nCreating sequences (length={seq_length})...")

        self.logger.info("\nCreating training sequences...")
        X_train, y_train = self.create_sequences(train_scaled, seq_length)

        self.logger.info("\nCreating validation sequences...")
        X_val, y_val = self.create_sequences(val_scaled, seq_length)

        self.logger.info("\nCreating testing sequences...")
        X_test, y_test = self.create_sequences(test_scaled, seq_length)

        self.logger.info("\nSequence shapes:")
        self.logger.info(f"X_train: {X_train.shape}")
        self.logger.info(f"y_train: {y_train.shape}")
        self.logger.info(f"X_val: {X_val.shape}")
        self.logger.info(f"y_val: {y_val.shape}")
        self.logger.info(f"X_test: {X_test.shape}")
        self.logger.info(f"y_test: {y_test.shape}")

        self.logger.info("\nConverting to PyTorch tensors...")
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)

        self.logger.info("\nSaving processed data...")

        self.save_processed_data(
            X_train_tensor, y_train_tensor,
            X_val_tensor, y_val_tensor,
            X_test_tensor, y_test_tensor,
            df, seq_length, features
        )

        processing_time = time.time() - start_time
        self.logger.info(f"\nProcessing completed in {processing_time:.2f} seconds")
        self.logger.info(f"{'=' * 50}")

        return {
            'X_train': X_train_tensor,
            'y_train': y_train_tensor,
            'X_val': X_val_tensor,
            'y_val': y_val_tensor,
            'X_test': X_test_tensor,
            'y_test': y_test_tensor,
            'scaler': self.scaler,
            'original_data': df,
            'seq_length': seq_length
        }

    def save_processed_data(self, X_train, y_train, X_val, y_val, X_test, y_test, original_data, seq_length, features):
        """Save each component separately"""
        # Create subdirectories
        tensors_dir = self.processed_dir / 'tensors'
        tensors_dir.mkdir(parents=True, exist_ok=True)

        # Save tensors
        torch.save(X_train, tensors_dir / 'X_train.pt')
        torch.save(y_train, tensors_dir / 'y_train.pt')
        torch.save(X_val, tensors_dir / 'X_val.pt')
        torch.save(y_val, tensors_dir / 'y_val.pt')
        torch.save(X_test, tensors_dir / 'X_test.pt')
        torch.save(y_test, tensors_dir / 'y_test.pt')

        # Save scaler
        with open(self.processed_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save original data
        original_data.to_pickle(self.processed_dir / 'original_data.pkl')

        # Save metadata
        metadata = {
            'seq_length': seq_length,
            'feature_indices': {name: idx for idx, name in enumerate(features)},
            'data_shape': {
                'X_train': X_train.shape,
                'y_train': y_train.shape,
                'X_val': X_val.shape,
                'y_val': y_val.shape,
                'X_test': X_test.shape,
                'y_test': y_test.shape
            },
            'date_range': {
                'start': original_data.index.min().strftime('%Y-%m-%d'),
                'end': original_data.index.max().strftime('%Y-%m-%d')
            }
        }
        with open(self.processed_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

    @staticmethod
    def load_processed_data(processed_dir):
        """Load all processed data components"""
        processed_dir = Path(processed_dir)
        tensors_dir = processed_dir / 'tensors'

        # Load tensors
        X_train = torch.load(tensors_dir / 'X_train.pt', weights_only=True)
        y_train = torch.load(tensors_dir / 'y_train.pt', weights_only=True)
        X_val = torch.load(tensors_dir / 'X_val.pt', weights_only=True)
        y_val = torch.load(tensors_dir / 'y_val.pt', weights_only=True)
        X_test = torch.load(tensors_dir / 'X_test.pt', weights_only=True)
        y_test = torch.load(tensors_dir / 'y_test.pt', weights_only=True)

        # Load scaler
        with open(processed_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Load original data
        original_data = pd.read_pickle(processed_dir / 'original_data.pkl')

        # Load metadata
        with open(processed_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler,
            'original_data': original_data,
            'metadata': metadata
        }


def main():
    """crypto_currency = 'btc'
    btc_processor = BasicDataProcessor(project_root / 'data/raw/comprehensive-btcusd-1m-data/BTCUSD_1m_Combined_Index.csv', subfolder=crypto_currency)

    btc_processed_data = btc_processor.process(
        start_date='2011-08-18',
        resample_rule='4h',
        seq_length=60
    )"""

    crypto_currency = 'eth'
    eth_processor = BasicDataProcessor(project_root / 'data/raw/comprehensive-ethusd-1m-data/ETHUSD_1m_Combined_Index.csv', subfolder=crypto_currency)

    eth_processed_data = eth_processor.process(
        start_date='2016-09-29',
        resample_rule='24h',
        seq_length=60
    )

if __name__ == "__main__":
    main()
