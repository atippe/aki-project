import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from custom_lstm import PricePredictionModel
import matplotlib.pyplot as plt
import sys
import joblib

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


def prepare_data(data, sequence_length):
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length:i + sequence_length + 1, 3]  # Index 3 is Close price
        sequences.append(seq)
        targets.append(target)

    sequences = np.array(sequences)
    targets = np.array(targets)

    return torch.FloatTensor(sequences), torch.FloatTensor(targets)


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
            scaled_loss = criterion(outputs, batcproh_y)
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
                target_features[:, 1] = scaled_target[:, 0]

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
                scaled_loss = criterion(outputs, batch_y)
                total_val_loss_scaled += scaled_loss.item()

                # Price scale loss
                scaled_pred = outputs.cpu().numpy()
                scaled_target = batch_y.cpu().numpy()

                pred_features = np.zeros((scaled_pred.shape[0], 5))
                pred_features[:, 1] = scaled_pred[:, 0]

                target_features = np.zeros((scaled_target.shape[0], 5))
                target_features[:, 1] = scaled_target[:, 0]

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
    sequence_length = 60
    hidden_size = 100
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.0005
    device = 'cpu'

    data_dir = project_root / 'data/processed/bitcoin-historical-data'
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    train_data = pd.read_csv(data_dir / 'train_data.csv', index_col=0)
    test_data = pd.read_csv(data_dir / 'test_data.csv', index_col=0)
    scaler = joblib.load(data_dir / 'scaler.pkl')

    print("\nData Statistics:")
    print("Training data:")
    print(train_data.describe())
    print("\nTest data:")
    print(test_data.describe())

    train_data.index = pd.to_datetime(train_data.index)
    test_data.index = pd.to_datetime(test_data.index)

    print(f"Data timeframe: {train_data.index.min()} to {train_data.index.max()}")
    print(f"Sampling frequency: {train_data.index[1] - train_data.index[0]}")

    X_train, y_train = prepare_data(train_data.values, sequence_length)
    X_test, y_test = prepare_data(test_data.values, sequence_length)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    model = PricePredictionModel(input_size=5, hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5)  # L2 regularization

    print("Training model...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        criterion, optimizer, num_epochs, device, scaler
    )

    # Save model and scaler
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, results_dir / 'lstm_model.pth')

    joblib.dump(scaler, results_dir / 'scaler.pkl')

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
