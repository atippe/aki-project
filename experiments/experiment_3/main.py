import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from custom_lstm import PricePredictionModel
import matplotlib.pyplot as plt
import pickle

from experiments.experiment_3.dataset_processor import BasicDataProcessor


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scaler):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_loss_scaled = 0
        num_batches = 0

        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)

            # Normalized/Scaled loss (0-1 range)
            scaled_loss = criterion(outputs, batch_y.unsqueeze(1))
            scaled_loss.backward()
            optimizer.step()

            # For monitoring, calculate both losses
            with torch.no_grad():
                # Track normalized loss
                total_train_loss_scaled += scaled_loss.item()

                # Calculate price-scale loss
                scaled_pred = outputs.cpu().numpy()
                scaled_target = batch_y.cpu().numpy()

                pred_features = np.zeros((scaled_pred.shape[0], 5))
                pred_features[:, 1] = scaled_pred[:, 0]

                target_features = np.zeros((scaled_target.shape[0], 5))
                target_features[:, 3] = scaled_target  # Index 3 for Close price

                price_pred = scaler.inverse_transform(pred_features)[:, 1]
                price_target = scaler.inverse_transform(target_features)[:, 1]

                price_scale_loss = np.mean((price_pred - price_target) ** 2)

                if epoch == 0 and batch_idx == 0:
                    print("\nDebug information:")
                    print(f"Scaled predictions (first 3):\n{scaled_pred[:3]}")
                    print(f"Scaled targets (first 3):\n{scaled_target[:3]}")
                    print(f"Normalized MSE: {scaled_loss.item():.6f}")
                    print(f"Normalized RMSE: {np.sqrt(scaled_loss.item()):.6f}")
                    print(f"Price predictions (first 3):\n{price_pred[:3]}")
                    print(f"Price targets (first 3):\n{price_target[:3]}")
                    print(f"Price-scale MSE: {price_scale_loss:.2f}")
                    print(f"Price-scale RMSE: {np.sqrt(price_scale_loss):.2f}\n")

            total_train_loss += price_scale_loss
            num_batches += 1

        # Validation
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

                pred_features = np.zeros((scaled_pred.shape[0], 5))
                pred_features[:, 1] = scaled_pred[:, 0]

                target_features = np.zeros((scaled_target.shape[0], 5))
                target_features[:, 3] = scaled_target  # Index 3 for Close price

                price_pred = scaler.inverse_transform(pred_features)[:, 1]
                price_target = scaler.inverse_transform(target_features)[:, 1]

                price_scale_loss = np.mean((price_pred - price_target) ** 2)
                total_val_loss += price_scale_loss
                num_val_batches += 1

        # Calculate averages
        avg_train_loss = total_train_loss / num_batches
        avg_val_loss = total_val_loss / num_val_batches
        avg_train_loss_scaled = total_train_loss_scaled / num_batches
        avg_val_loss_scaled = total_val_loss_scaled / num_val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}]:')
        print(f'  Normalized - Train MSE: {avg_train_loss_scaled:.6f}, RMSE: {np.sqrt(avg_train_loss_scaled):.6f}')
        print(f'  Normalized - Val MSE: {avg_val_loss_scaled:.6f}, RMSE: {np.sqrt(avg_val_loss_scaled):.6f}')
        print(f'  Price Scale - Train MSE: {avg_train_loss:.2f}, RMSE: ${np.sqrt(avg_train_loss):.2f}')
        print(f'  Price Scale - Val MSE: {avg_val_loss:.2f}, RMSE: ${np.sqrt(avg_val_loss):.2f}')

    return train_losses, val_losses


def main():
    # Parameters
    hidden_size = 100
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.0005
    device = 'cpu'

    # Setup directories
    data_dir = Path(__file__).parent / 'processed_data'
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    # Load pre-processed data
    print("Loading pre-processed data...")
    processed_data = BasicDataProcessor.load_processed_data(data_dir)

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

    # Initialize model and training components
    input_size = processed_data['X_train'].shape[2]  # Get input size from data
    model = PricePredictionModel(
        input_size=input_size,
        hidden_size=hidden_size
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    # Training
    print("Training model...")
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs,
        device,
        processed_data['scaler']
    )

    # Save results
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metadata': metadata
    }, results_dir / 'lstm_model.pth')

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
