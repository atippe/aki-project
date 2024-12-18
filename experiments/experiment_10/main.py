import time

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from custom_lstm import MultiStepPricePredictionModel
import matplotlib.pyplot as plt
import pickle
import logging
import seaborn as sns


from dataset_processor import BasicDataProcessor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, scaler, logger, close_idx,
                feature_count):
    logger.info("Starting training process")
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Training parameters: epochs={num_epochs}, device={device}")
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler}")
    logger.info(f"Loss function: {criterion}")

    train_losses = []
    val_losses = []
    attention_patterns = []

    # Log initial dataset sizes
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_train_loss_scaled = 0
        num_batches = 0
        epoch_attention_weights = []
        # Log epoch start
        logger.info(f"\nStarting Epoch [{epoch + 1}/{num_epochs}]")

        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            if device == 'cuda':
                logger.info(f"CUDA Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB allocated")

            optimizer.zero_grad()
            outputs, attention_weights = model(batch_X)
            # Calculate loss for all steps
            scaled_loss = 0
            for step in range(outputs.size(1)):
                step_loss = criterion(outputs[:, step, :], batch_y[:, step].unsqueeze(1))
                scaled_loss += step_loss
            scaled_loss = scaled_loss / outputs.size(1)  # Average loss across steps

            scaled_loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_idx % 100 == 0:
                epoch_attention_weights.append(attention_weights.detach().cpu().numpy())

            with torch.no_grad():
                total_train_loss_scaled += scaled_loss.item()

                # Price scale loss calculation
                price_scale_loss = 0
                for step in range(outputs.size(1)):
                    scaled_pred = outputs[:, step, :].cpu().numpy()
                    scaled_target = batch_y[:, step].cpu().numpy()

                    pred_features = np.zeros((scaled_pred.shape[0], feature_count))
                    pred_features[:, close_idx] = scaled_pred.flatten()
                    target_features = np.zeros((scaled_target.shape[0], feature_count))
                    target_features[:, close_idx] = scaled_target

                    price_pred = scaler.inverse_transform(pred_features)[:, close_idx]
                    price_target = scaler.inverse_transform(target_features)[:, close_idx]

                    price_pred = np.expm1(price_pred)
                    price_target = np.expm1(price_target)
                    price_scale_loss += np.mean((price_pred - price_target) ** 2)

                price_scale_loss = price_scale_loss / outputs.size(1)
                total_train_loss += price_scale_loss
                num_batches += 1

                # Log first batch details in first epoch
                if epoch == 0 and batch_idx == 0:
                    logger.info("\nFirst batch details:")
                    logger.info(f"Scaled predictions (first 3):\n{scaled_pred[:3]}")
                    logger.info(f"Scaled targets (first 3):\n{scaled_target[:3]}")
                    logger.info(f"Price predictions (first 3):\n{price_pred[:3]}")
                    logger.info(f"Price targets (first 3):\n{price_target[:3]}")
                    logger.info(f"Initial normalized MSE: {scaled_loss.item():.6f}")
                    logger.info(f"Initial price-scale MSE: {price_scale_loss:.2f}")

        attention_patterns.append(epoch_attention_weights)

        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_loss_scaled = 0
        num_val_batches = 0
        val_attention_weights = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs, attention_weights = model(batch_X)

                scaled_loss = 0
                price_scale_loss = 0

                for step in range(outputs.size(1)):
                    # Normalized loss for each step
                    step_loss = criterion(outputs[:, step, :], batch_y[:, step].unsqueeze(1))
                    scaled_loss += step_loss

                    # Price scale loss calculation for each step
                    scaled_pred = outputs[:, step, :].cpu().numpy()
                    scaled_target = batch_y[:, step].cpu().numpy()

                    pred_features = np.zeros((scaled_pred.shape[0], feature_count))
                    pred_features[:, close_idx] = scaled_pred.flatten()
                    target_features = np.zeros((scaled_target.shape[0], feature_count))
                    target_features[:, close_idx] = scaled_target

                    price_pred = scaler.inverse_transform(pred_features)[:, close_idx]
                    price_target = scaler.inverse_transform(target_features)[:, close_idx]

                    # Because of log transformation, we need to apply expm1 to reverse it
                    price_pred = np.expm1(price_pred)
                    price_target = np.expm1(price_target)
                    price_scale_loss += np.mean((price_pred - price_target) ** 2)

                # Average losses across steps
                scaled_loss = scaled_loss / outputs.size(1)
                price_scale_loss = price_scale_loss / outputs.size(1)

                total_val_loss_scaled += scaled_loss.item()
                total_val_loss += price_scale_loss
                num_val_batches += 1

                if batch_idx % 100 == 0:
                    val_attention_weights.append(attention_weights.cpu().numpy())

        # Calculate and log metrics
        avg_train_loss = total_train_loss / num_batches
        avg_val_loss = total_val_loss / num_val_batches
        avg_train_loss_scaled = total_train_loss_scaled / num_batches
        avg_val_loss_scaled = total_val_loss_scaled / num_val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 5 == 0:
            logger.info(f"\nAttention Pattern Sample (Epoch {epoch + 1}):")
            if epoch_attention_weights:
                avg_attention = np.mean(epoch_attention_weights[0], axis=0)
                logger.info(f"Average attention weights: {avg_attention[:5]}")

        # Log epoch results
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}] Summary:')
        logger.info(
            f'  Normalized - Train MSE: {avg_train_loss_scaled:.6f}, RMSE: {np.sqrt(avg_train_loss_scaled):.6f}')
        logger.info(f'  Normalized - Val MSE: {avg_val_loss_scaled:.6f}, RMSE: {np.sqrt(avg_val_loss_scaled):.6f}')
        logger.info(f'  Price Scale - Train MSE: {avg_train_loss:.2f}, RMSE: ${np.sqrt(avg_train_loss):.2f}')
        logger.info(f'  Price Scale - Val MSE: {avg_val_loss:.2f}, RMSE: ${np.sqrt(avg_val_loss):.2f}')

    return train_losses, val_losses, attention_patterns


def generate_predictions(model, test_loader, device, processed_data, feature_count, close_idx):
    model.eval()
    y_pred_list = [[] for _ in range(processed_data['metadata']['prediction_steps'])]
    y_test_list = [[] for _ in range(processed_data['metadata']['prediction_steps'])]
    attention_weights_list = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs, attention_weights = model(X)

            attention_weights_list.append(attention_weights.cpu().numpy())

            # Handle each prediction step separately
            for step in range(outputs.size(1)):
                scaled_pred = outputs[:, step, :].cpu().numpy()
                scaled_target = y[:, step].cpu().numpy()

                pred_features = np.zeros((scaled_pred.shape[0], feature_count))
                pred_features[:, close_idx] = scaled_pred.flatten()

                target_features = np.zeros((scaled_target.shape[0], feature_count))
                target_features[:, close_idx] = scaled_target

                price_pred = processed_data['scaler'].inverse_transform(pred_features)[:, close_idx]
                price_target = processed_data['scaler'].inverse_transform(target_features)[:, close_idx]

                # Because of log transformation, we need to apply expm1 to reverse it
                price_pred = np.expm1(price_pred)
                price_target = np.expm1(price_target)

                y_pred_list[step].extend(price_pred)
                y_test_list[step].extend(price_target)

    # Convert to DataFrame for better organization of multi-step predictions
    predictions = pd.DataFrame({
        f'step_{i + 1}': pd.Series(np.array(y_pred_list[i]))
        for i in range(processed_data['metadata']['prediction_steps'])
    })

    actuals = pd.DataFrame({
        f'step_{i + 1}': pd.Series(np.array(y_test_list[i]))
        for i in range(processed_data['metadata']['prediction_steps'])
    })

    attention_weights = np.concatenate(attention_weights_list, axis=0)

    return predictions, actuals, attention_weights


def plot_training_history(train_losses, val_losses):
    fig = plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Original Scale)')
    plt.legend()
    return fig


def plot_prediction_analysis(y_test, y_pred):
    # Create subplots for each prediction step
    steps = y_pred.shape[1]
    figs_scatter = []
    figs_residuals = []
    residuals_all = []

    for step in range(steps):
        # Actual vs Predicted Plot for each step
        fig1 = plt.figure(figsize=(10, 6))
        plt.scatter(y_test[f'step_{step + 1}'], y_pred[f'step_{step + 1}'], alpha=0.5)
        plt.plot([y_test[f'step_{step + 1}'].min(), y_test[f'step_{step + 1}'].max()],
                 [y_test[f'step_{step + 1}'].min(), y_test[f'step_{step + 1}'].max()], 'r--', lw=2)
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title(f'Price: Actual vs Predicted - Step {step + 1} (Test Set)')
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.grid(True)
        figs_scatter.append(fig1)

        # Residuals Plot for each step
        residuals = y_test[f'step_{step + 1}'] - y_pred[f'step_{step + 1}']
        residuals_all.append(residuals)

        fig2 = plt.figure(figsize=(10, 6))
        plt.scatter(y_pred[f'step_{step + 1}'], residuals, alpha=0.5)
        plt.xlabel('Predicted Price ($)')
        plt.ylabel('Residuals ($)')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'Residuals Analysis - Step {step + 1} (Test Set)')
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.grid(True)
        figs_residuals.append(fig2)

    return residuals_all, figs_scatter, figs_residuals


def calculate_metrics(y_test, y_pred):
    # Basic metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
    wmape = np.sum(np.abs(y_test - y_pred)) / np.sum(y_test) * 100  # Weighted MAPE

    # Directional accuracy (for price movements)
    y_test_diff = np.diff(y_test)
    y_pred_diff = np.diff(y_pred)
    directional_accuracy = np.mean(
        ((y_test_diff >= 0) & (y_pred_diff >= 0)) |
        ((y_test_diff < 0) & (y_pred_diff < 0))
    ) * 100

    # Maximum drawdown in predictions
    max_error = np.max(np.abs(y_test - y_pred))
    mean_pct_error = np.mean((y_test - y_pred) / y_test) * 100

    return {
        'Mean Squared Error ($)': mse,
        'Root Mean Squared Error ($)': rmse,
        'Mean Absolute Error ($)': mae,
        'Mean Absolute Percentage Error (%)': mape,
        'Weighted MAPE (%)': wmape,
        'Directional Accuracy (%)': directional_accuracy,
        'Maximum Absolute Error ($)': max_error,
        'Mean Percentage Error (%)': mean_pct_error,
        'R² Score': r2
    }


def create_styled_table(data, title, figsize=(10, 3), custom_widths=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Set column widths based on content
    if custom_widths:
        colWidths = custom_widths
    else:
        colWidths = [0.7 / len(data.columns)] * len(data.columns)

    # Create table
    table = ax.table(
        cellText=data.values,
        colLabels=data.columns,
        cellLoc='center',
        loc='center',
        colWidths=colWidths
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color scheme
    header_color = '#3d5a80'
    cell_color = '#f0f8ff'

    # Style header
    for j, cell in enumerate(table._cells[(0, j)] for j in range(len(data.columns))):
        cell.set_facecolor(header_color)
        cell.set_text_props(color='white', weight='bold')
        cell.set_height(0.15)

    # Style cells
    for i in range(len(data)):
        for j in range(len(data.columns)):
            cell = table._cells[(i + 1, j)]
            cell.set_facecolor(cell_color)
            cell.set_height(0.15)

    plt.subplots_adjust(top=0.85)
    plt.title(title, pad=30, fontsize=12, fontweight='bold', y=1.2)
    plt.tight_layout()
    #plt.savefig(results_dir / filename, bbox_inches='tight', dpi=300, facecolor='white')

    return fig


def save_metrics_and_samples(y_test, y_pred, metrics, processed_data):
    # Metrics table
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': [f"{v:,.2f}" for v in metrics.values()]
    })
    fig1 = create_styled_table(metrics_df, 'Model Performance Metrics (Test Set)', figsize=(6, 3))

    # Sample predictions table
    sample_size = 10
    indices = np.random.randint(0, len(y_test), sample_size)

    # Get dates from original data
    test_dates = processed_data['original_data'].index[-len(y_test):]  # Get dates for test set
    selected_dates = test_dates[indices]  # Get dates for selected samples

    comparison_df = pd.DataFrame({
        'Date': selected_dates.strftime('%Y-%m-%d %H:%M'),
        'Actual Price': [f"${x:,.2f}" for x in y_test.iloc[indices]],
        'Predicted Price': [f"${x:,.2f}" for x in y_pred[indices]],
        'Absolute Error': [f"${x:,.2f}" for x in abs(y_test.iloc[indices] - y_pred[indices])],
        'Percentage Error': [f"{x:.2f}%" for x in
                             abs((y_test.iloc[indices] - y_pred[indices]) / y_test.iloc[indices] * 100)]
    })

    fig2 = create_styled_table(
        comparison_df,
        'Sample Predictions Comparison',
        figsize=(12, 4),
        custom_widths=[0.2, 0.2, 0.2, 0.2, 0.2]
    )

    return fig1, fig2


def plot_error_distribution(residuals):
    fig = plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors (Test Set)')
    plt.tight_layout()

    return fig


def plot_time_series(y_test, y_pred, processed_data):
    test_dates = processed_data['original_data'].index[-len(y_test):]
    fig = plt.figure(figsize=(15, 6))

    # Plot actual values
    plt.plot(test_dates, y_test['step_1'], label='Actual Price', color='blue', alpha=0.7)

    # Plot predictions for each step with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, y_pred.shape[1]))
    for step in range(y_pred.shape[1]):
        plt.plot(test_dates, y_pred[f'step_{step + 1}'],
                 label=f'Predicted (t+{step + 1})',
                 color=colors[step], alpha=0.7)

    plt.title('Multi-step Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()

    return fig

def plot_attention_heatmap(attention_weights):
    plt.figure(figsize=(12, 8))

    attention_weights = attention_weights.squeeze(-1)
    sequence_length = attention_weights.shape[1]

    avg_attention = np.mean(attention_weights, axis=0)

    heatmap = sns.heatmap(avg_attention.reshape(-1, 1).T, cmap='viridis')
    plt.title('Average Attention Weights Across Sequence')
    plt.xlabel(f'Time Step (sequence length: {sequence_length})')
    plt.ylabel('Attention Weight')

    plt.colorbar(heatmap.collections[0], label='Attention Weight')

    return plt.gcf()


def plot_time_series_with_attention(y_test, y_pred, attention_weights, processed_data):
    """Plot time series with attention weights overlay."""
    original_data = processed_data['original_data']
    sequence_length = processed_data['metadata']['seq_length']
    test_dates = original_data.index[-len(y_test):]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])

    # Price plot
    ax1.plot(test_dates, y_test.values, label='Actual Price', color='blue', alpha=0.7)
    ax1.plot(test_dates, y_pred, label='Predicted Price', color='red', alpha=0.7)
    ax1.set_title('Price Predictions with Attention Weights')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)

    # Attention weights plot
    attention_weights = attention_weights.squeeze(-1)
    avg_attention = np.mean(attention_weights, axis=0)
    im = ax2.imshow(avg_attention.reshape(1, -1), aspect='auto', cmap='viridis')
    ax2.set_title('Attention Weights')
    ax2.set_xlabel('Input Sequence Position')
    ax2.set_ylabel('Attention Weight')

    plt.colorbar(im, ax=ax2, label='Attention Weight')
    plt.tight_layout()
    return fig


def evaluate_and_visualize(model, test_loader, device, processed_data, feature_count, close_idx, train_losses,
                           val_losses, results_dir, logger):
    # Generate predictions
    logger.info("Generating predictions...")
    y_pred, y_test, attention_weights = generate_predictions(model, test_loader, device, processed_data, feature_count,
                                                             close_idx)

    # Create visualizations
    logger.info("Creating visualizations...")
    plot_1 = plot_training_history(train_losses, val_losses)
    plot_1.savefig(results_dir / 'training_history.png', bbox_inches='tight', dpi=300)
    plt.close()
    logger.info("Training history plot saved")

    # Calculate metrics for each step and combine into one table
    all_metrics = {}
    metrics_combined = pd.DataFrame()

    for step in range(y_pred.shape[1]):
        step_metrics = calculate_metrics(y_test[f'step_{step + 1}'], y_pred[f'step_{step + 1}'])
        all_metrics[f'Step {step + 1}'] = step_metrics

        # Create DataFrame for this step's metrics
        step_df = pd.DataFrame({
            'Metric': list(step_metrics.keys()),
            f'Step {step + 1}': [f"{v:,.2f}" for v in step_metrics.values()]
        })

        if metrics_combined.empty:
            metrics_combined = step_df
        else:
            metrics_combined = metrics_combined.merge(step_df, on='Metric')

    # Create combined metrics table
    plot_4 = create_styled_table(metrics_combined, 'Model Performance Metrics Across Steps', figsize=(12, 6))
    plot_4.savefig(results_dir / 'metrics_table.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

    # Create combined time series plot
    plot_7 = plot_time_series(y_test, y_pred, processed_data)
    plot_7.savefig(results_dir / 'time_series.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Create combined error distribution plot
    fig = plt.figure(figsize=(12, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, y_pred.shape[1]))
    for step in range(y_pred.shape[1]):
        residuals = y_test[f'step_{step + 1}'] - y_pred[f'step_{step + 1}']
        plt.hist(residuals, bins=50, alpha=0.5, label=f'Step {step + 1}', color=colors[step])
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors Across Steps (Test Set)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / 'error_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Attention visualization
    logger.info("Creating attention visualizations...")
    plot_8 = plot_attention_heatmap(attention_weights)
    plot_9 = plot_time_series_with_attention(y_test, y_pred, attention_weights, processed_data)
    plot_8.savefig(results_dir / 'attention_heatmap.png', bbox_inches='tight', dpi=300)
    plot_9.savefig(results_dir / 'time_series_with_attention.png', bbox_inches='tight', dpi=300)
    plt.close('all')
    logger.info("Attention visualization plots saved")

    return all_metrics


def save_model_specifications(params, model, criterion, optimizer, processed_data, results_dir):
    specs = {
        'Model Architecture': {
            'Type': 'Advanced LSTM with Attention',
            'Hidden Size': params['hidden_size'],
            'Number of Layers': params['num_layers'],
            'Dropout Rate': params['dropout'],
            'Input Size': model.lstm.input_size,
            'Output Size': model.prediction_heads[0][-1].out_features,
            'Prediction Steps': len(model.prediction_heads),
            'Model Class': model.__class__.__name__
        },
        'Training Parameters': {
            'Epochs': params['num_epochs'],
            'Batch Size': params['batch_size'],
            'Learning Rate': params['learning_rate'],
            'Device': str(params['device']),
            'Loss Function': criterion.__class__.__name__,
            'Optimizer': optimizer.__class__.__name__,
            'Weight Decay': params['weight_decay'],
            'Beta Values': params['betas'],
            'Epsilon': params['eps']
        },
        'Data Configuration': {
            'Sequence Length': processed_data['metadata']['seq_length'],
            'Feature Count': len(processed_data['metadata']['feature_indices']),
            'Training Samples': processed_data['X_train'].shape,
            'Validation Samples': processed_data['X_val'].shape,
            'Test Samples': processed_data['X_test'].shape,
            'Close Price Index': processed_data['metadata']['feature_indices']['Close'],
            'Prediction Steps': processed_data['metadata']['prediction_steps']
        }
    }

    # Save as text file
    with open(results_dir / 'model_specifications.txt', 'w') as f:
        f.write("=== Model Specifications ===\n\n")
        for category, parameters in specs.items():
            f.write(f"{category}:\n")
            for param, value in parameters.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")

def main():
    # Parameters
    params = {
        'hidden_size': 256,
        'num_layers': 3,
        'dropout': 0.3,
        'num_epochs': 50,
        'batch_size': 64,
        'learning_rate': 0.001,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'weight_decay': 1e-5,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'criterion': 'MSE'  # or 'Huber' or 'MAE'
    }

    crypto_currency = 'eth'
    # Setup directories and logging
    data_dir = Path(__file__).parent / 'processed_data' / crypto_currency
    results_dir = Path(__file__).parent / 'results' / crypto_currency
    results_dir.mkdir(exist_ok=True)

    logger = setup_logging(results_dir)

    start_time = time.time()

    # Log training configuration
    logger.info("=== Starting new training run ===")
    logger.info(f"Crypto Currency: {crypto_currency}")
    logger.info("\nModel Parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")

    # Load data
    logger.info("Loading pre-processed data...")
    processed_data = BasicDataProcessor.load_processed_data(data_dir)

    # Log dataset information
    feature_count = len(processed_data['metadata']['feature_indices'])
    close_idx = processed_data['metadata']['feature_indices']['Close']
    scaler = processed_data['scaler']

    logger.info("\n=== Dataset Configuration ===")
    logger.info(f"Sequence length: {processed_data['metadata']['seq_length']}")
    logger.info(f"Feature count: {feature_count}")
    logger.info(f"Feature indices: {processed_data['metadata']['feature_indices']}")
    logger.info(f"Close price index: {close_idx}")

    logger.info("\n=== Data Shapes ===")
    logger.info(f"Training set shape: {processed_data['X_train'].shape}")
    logger.info(f"Validation set shape: {processed_data['X_val'].shape}")
    logger.info(f"Test set shape: {processed_data['X_test'].shape}")

    logger.info("\n=== Date Range ===")
    logger.info(f"Start date: {processed_data['metadata']['date_range']['start']}")
    logger.info(f"End date: {processed_data['metadata']['date_range']['end']}")

    # Add feature alignment verification
    logger.info("\n=== Feature Alignment Verification ===")
    dummy_data = np.zeros((1, feature_count))
    dummy_data[:, close_idx] = 1.0
    logger.info(f"Feature alignment test - Close position: {close_idx}")
    logger.info(f"Inverse transform test shape: {scaler.inverse_transform(dummy_data).shape}")

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
        batch_size=params['batch_size'],
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=params['batch_size']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=params['batch_size']
    )

    # Create model
    input_size = processed_data['X_train'].shape[2]
    prediction_steps = processed_data['metadata']['prediction_steps']
    logger.info(f"Prediction steps: {prediction_steps}")
    model = MultiStepPricePredictionModel(
        input_size=processed_data['X_train'].shape[2],
        hidden_size=params['hidden_size'],
        output_steps=prediction_steps,
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(params['device'])
    logger.info(f"Model input size: {input_size}")

    if params['criterion'] == 'MSE':
        criterion = nn.MSELoss()
    elif params['criterion'] == 'MAE':
        criterion = nn.L1Loss()
    elif params['criterion'] == 'Huber':
        criterion = nn.HuberLoss()
    else:
        raise ValueError(f"Unknown criterion type: {params['criterion']}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
        betas=params['betas'],
        eps=params['eps']
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=params['learning_rate'] * 3,
        epochs=params['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=10000
    )

    # Training
    logger.info("Starting training process...")
    train_losses, val_losses, attention_patterns = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        params['num_epochs'], params['device'], scaler, logger, close_idx, feature_count
    )

    logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")

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

    # Evaluation
    logger.info("Starting evaluation process...")
    metrics = evaluate_and_visualize(
        model, test_loader, params['device'], processed_data, feature_count, close_idx, train_losses, val_losses, results_dir,
        logger
    )

    # Log metrics
    logger.info("Evaluation metrics (for test set):")
    for step, step_metrics in metrics.items():
        logger.info(f"\n{step}:")
        for k, v in step_metrics.items():
            logger.info(f"  {k}: {v}")

    # Save model specifications
    save_model_specifications(params, model, criterion, optimizer, processed_data, results_dir)

if __name__ == "__main__":
    main()
