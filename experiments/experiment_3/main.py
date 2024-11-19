import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from custom_lstm import PricePredictionModel
import matplotlib.pyplot as plt
import pickle
import logging

from experiments.experiment_3.dataset_processor import BasicDataProcessor


def setup_logging(results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / 'training.log'

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scaler, logger, close_idx, feature_count):
    logger.info("Starting training process")
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Training parameters: epochs={num_epochs}, device={device}")
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Loss function: {criterion}")

    train_losses = []
    val_losses = []

    # Log initial dataset sizes
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_loss_scaled = 0
        num_batches = 0

        # Log epoch start
        logger.info(f"\nStarting Epoch [{epoch + 1}/{num_epochs}]")

        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Log batch shapes occasionally
            if batch_idx == 0:
                logger.debug(f"Batch shapes - X: {batch_X.shape}, y: {batch_y.shape}")

            optimizer.zero_grad()
            outputs = model(batch_X)
            scaled_loss = criterion(outputs, batch_y.unsqueeze(1))
            scaled_loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_train_loss_scaled += scaled_loss.item()

                # Price scale loss calculation
                scaled_pred = outputs.cpu().numpy()
                scaled_target = batch_y.cpu().numpy()

                pred_features = np.zeros((scaled_pred.shape[0], feature_count))
                pred_features[:, close_idx] = scaled_pred.flatten()
                target_features = np.zeros((scaled_target.shape[0], feature_count))
                target_features[:, close_idx] = scaled_target

                price_pred = scaler.inverse_transform(pred_features)[:, close_idx]
                price_target = scaler.inverse_transform(target_features)[:, close_idx]
                price_scale_loss = np.mean((price_pred - price_target) ** 2)

                # Log first batch details in first epoch
                if epoch == 0 and batch_idx == 0:
                    logger.info("\nFirst batch details:")
                    logger.info(f"Scaled predictions (first 3):\n{scaled_pred[:3]}")
                    logger.info(f"Scaled targets (first 3):\n{scaled_target[:3]}")
                    logger.info(f"Price predictions (first 3):\n{price_pred[:3]}")
                    logger.info(f"Price targets (first 3):\n{price_target[:3]}")
                    logger.info(f"Initial normalized MSE: {scaled_loss.item():.6f}")
                    logger.info(f"Initial price-scale MSE: {price_scale_loss:.2f}")

            total_train_loss += price_scale_loss
            num_batches += 1

        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_loss_scaled = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)

                # Normalized loss
                scaled_loss = criterion(outputs, batch_y.unsqueeze(1))
                total_val_loss_scaled += scaled_loss.item()

                # Price scale loss
                scaled_pred = outputs.cpu().numpy()
                scaled_target = batch_y.cpu().numpy()

                pred_features = np.zeros((scaled_pred.shape[0], feature_count))
                pred_features[:, close_idx] = scaled_pred.flatten()

                target_features = np.zeros((scaled_target.shape[0], feature_count))
                target_features[:, close_idx] = scaled_target

                price_pred = scaler.inverse_transform(pred_features)[:, close_idx]
                price_target = scaler.inverse_transform(target_features)[:, close_idx]

                price_scale_loss = np.mean((price_pred - price_target) ** 2)
                total_val_loss += price_scale_loss
                num_val_batches += 1

        # Calculate and log metrics
        avg_train_loss = total_train_loss / num_batches
        avg_val_loss = total_val_loss / num_val_batches
        avg_train_loss_scaled = total_train_loss_scaled / num_batches
        avg_val_loss_scaled = total_val_loss_scaled / num_val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Log epoch results
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}] Summary:')
        logger.info(
            f'  Normalized - Train MSE: {avg_train_loss_scaled:.6f}, RMSE: {np.sqrt(avg_train_loss_scaled):.6f}')
        logger.info(f'  Normalized - Val MSE: {avg_val_loss_scaled:.6f}, RMSE: {np.sqrt(avg_val_loss_scaled):.6f}')
        logger.info(f'  Price Scale - Train MSE: {avg_train_loss:.2f}, RMSE: ${np.sqrt(avg_train_loss):.2f}')
        logger.info(f'  Price Scale - Val MSE: {avg_val_loss:.2f}, RMSE: ${np.sqrt(avg_val_loss):.2f}')

    logger.info("Training completed")
    return train_losses, val_losses


def main():
    # Parameters
    hidden_size = 100
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.0005
    device = 'cpu'

    # Setup directories and logging
    data_dir = Path(__file__).parent / 'processed_data'
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    logger = setup_logging(results_dir)

    # Log training configuration
    logger.info("=== Starting new training run ===")
    logger.info(f"Parameters:")
    logger.info(f"  Hidden size: {hidden_size}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Device: {device}")

    # Load data
    logger.info("Loading pre-processed data...")
    processed_data = BasicDataProcessor.load_processed_data(data_dir)

    # Log dataset information
    feature_count = len(processed_data['metadata']['feature_indices'])
    close_idx = processed_data['metadata']['feature_indices']['Close']
    logger.info(f"Sequence length: {processed_data['metadata']['seq_length']}")
    logger.info(f"Feature count: {feature_count}")
    logger.info(f"Close price index: {close_idx}")
    logger.info(f"Training set shape: {processed_data['X_train'].shape}")
    logger.info(f"Validation set shape: {processed_data['X_val'].shape}")
    logger.info(f"Test set shape: {processed_data['X_test'].shape}")

    # Extract metadata
    metadata = processed_data['metadata']
    sequence_length = metadata['seq_length']

    # Create data loaders directly from saved tensors
    train_dataset = torch.utils.data.TensorDataset(
        processed_data['X_train'],
        processed_data['y_train']
    )
    val_dataset = torch.utils.data.TensorDataset(
        processed_data['X_val'],
        processed_data['y_val']
    )
    test_dataset = torch.utils.data.TensorDataset(
        processed_data['X_test'],
        processed_data['y_test']
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    # Create model
    input_size = processed_data['X_train'].shape[2]
    model = PricePredictionModel(input_size=input_size, hidden_size=hidden_size).to(device)
    logger.info(f"Model input size: {input_size}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    # Training
    logger.info("Starting training process...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs, device, processed_data['scaler'], logger, close_idx, feature_count
    )


    # Save results
    logger.info("Saving model and results...")
    save_path = results_dir / 'lstm_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metadata': metadata
    }, save_path)

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Original Scale)')
    plt.legend()
    plt.savefig(results_dir / 'training_history.png')
    plt.close()


if __name__ == "__main__":
    main()
