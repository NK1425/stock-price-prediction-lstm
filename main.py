#!/usr/bin/env python3
"""
Stock Price Prediction with LSTM and Attention - Main Entry Point

This script provides a command-line interface for training, evaluating,
and making predictions with the LSTM-Attention model.

Usage:
    python main.py --ticker AAPL --train
    python main.py --ticker AAPL --predict
    python main.py --ticker AAPL --evaluate
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import StockDataLoader
from feature_engineering import FeatureEngineer
from sequence_builder import SequenceBuilder, create_train_val_test_sequences, create_single_prediction_sequence
from model import build_attention_model_with_weights, get_callbacks, AttentionLayer
from evaluate import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)


def get_feature_columns():
    """Get the list of feature columns to use."""
    return [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Returns', 'Log_Returns',
        'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50',
        'RSI_14', 'RSI_7',
        'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Percent',
        'ATR', 'ATR_Percent', 'Volatility_20d',
        'Momentum_10', 'Momentum_20',
        'OBV', 'OBV_Trend', 'VWAP', 'Price_VWAP_Ratio',
        'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI',
        'ROC_10', 'MFI'
    ]


def prepare_data(ticker, start_date, end_date, sequence_length, forecast_horizon):
    """Prepare data for training or prediction."""
    logger.info(f"Preparing data for {ticker}...")

    # Load data
    loader = StockDataLoader(cache_dir='./data')
    data = loader.download_stock_data(ticker, start_date, end_date)
    logger.info(f"Downloaded {len(data)} trading days")

    # Generate features
    engineer = FeatureEngineer()
    data_with_features = engineer.add_all_features(data)
    data_clean = engineer.handle_missing_values(data_with_features, method='drop')
    logger.info(f"Generated {len(engineer.feature_columns)} features")

    # Filter feature columns
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in data_clean.columns]

    # Create sequence builder
    builder = SequenceBuilder(
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        feature_columns=feature_cols,
        target_column='Close'
    )

    return data_clean, builder, feature_cols


def train(args):
    """Train the model."""
    import tensorflow as tf
    tf.random.set_seed(SEED)

    logger.info("=" * 60)
    logger.info(f"Training model for {args.ticker}")
    logger.info("=" * 60)

    # Prepare data
    data_clean, builder, feature_cols = prepare_data(
        args.ticker, args.start_date, args.end_date,
        args.sequence_length, args.forecast_horizon
    )

    # Create sequences
    (X_train, y_train), (X_val, y_val), (X_test, y_test), test_dates = \
        create_train_val_test_sequences(data_clean, builder, 0.7, 0.15, 0.15)

    logger.info(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    # Build model
    n_features = X_train.shape[2]
    model, attention_model = build_attention_model_with_weights(
        input_shape=(args.sequence_length, n_features),
        output_steps=args.forecast_horizon,
        lstm_units=[128, 64],
        attention_units=64,
        dropout_rate=0.2,
        learning_rate=0.001
    )

    model.summary()

    # Training callbacks
    os.makedirs('./models', exist_ok=True)
    callbacks = get_callbacks(
        model_path=f'./models/{args.ticker}_best.keras',
        patience=args.patience
    )

    # Train
    logger.info("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    predictions, attention = attention_model.predict(X_test, verbose=0)
    pred_unscaled = builder.inverse_transform_predictions(predictions)
    actual_unscaled = builder.inverse_transform_predictions(y_test)

    # Calculate metrics
    evaluator = ModelEvaluator(results_dir='./results', ticker=args.ticker)
    metrics = evaluator.calculate_metrics(pred_unscaled, actual_unscaled)

    logger.info("\n=== Test Set Metrics ===")
    logger.info(f"RMSE: ${metrics['rmse']:.2f}")
    logger.info(f"MAE: ${metrics['mae']:.2f}")
    logger.info(f"MAPE: {metrics['mape']:.2f}%")

    # Generate visualizations
    evaluator.generate_report(
        predictions=pred_unscaled,
        actuals=actual_unscaled,
        attention_weights=attention,
        dates=test_dates,
        history={k: [float(v) for v in vals] for k, vals in history.history.items()},
        save=True
    )

    # Save final model
    model.save(f'./models/{args.ticker}_final.keras')

    # Save parameters
    params = {
        'ticker': args.ticker,
        'sequence_length': args.sequence_length,
        'forecast_horizon': args.forecast_horizon,
        'feature_columns': feature_cols,
        'metrics': metrics,
        'trained_at': datetime.now().isoformat()
    }
    with open(f'./models/{args.ticker}_params.json', 'w') as f:
        json.dump(params, f, indent=2)

    logger.info(f"\nModel saved to ./models/{args.ticker}_final.keras")
    logger.info(f"Visualizations saved to ./results/figures/")
    logger.info("Training complete!")


def predict(args):
    """Make predictions using a trained model."""
    import tensorflow as tf

    logger.info("=" * 60)
    logger.info(f"Making predictions for {args.ticker}")
    logger.info("=" * 60)

    # Load model parameters
    params_path = f'./models/{args.ticker}_params.json'
    if not os.path.exists(params_path):
        logger.error(f"Model parameters not found: {params_path}")
        logger.error("Please train the model first using --train")
        return

    with open(params_path, 'r') as f:
        params = json.load(f)

    # Load model
    model_path = f'./models/{args.ticker}_final.keras'
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    logger.info("Model loaded successfully")

    # Get latest data
    loader = StockDataLoader(cache_dir='./data')
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    data = loader.download_stock_data(args.ticker, start_date, end_date, use_cache=False)

    # Generate features
    engineer = FeatureEngineer()
    data_with_features = engineer.add_all_features(data)
    data_clean = engineer.handle_missing_values(data_with_features, method='drop')

    # Create sequence builder
    feature_cols = params['feature_columns']
    feature_cols = [c for c in feature_cols if c in data_clean.columns]

    builder = SequenceBuilder(
        sequence_length=params['sequence_length'],
        forecast_horizon=params['forecast_horizon'],
        feature_columns=feature_cols,
        target_column='Close'
    )

    # Fit scaler on available data
    builder.fit(data_clean)

    # Create prediction sequence
    latest_sequence = create_single_prediction_sequence(data_clean, builder)

    # Predict
    predictions = model.predict(latest_sequence, verbose=0)
    future_prices = builder.inverse_transform_predictions(predictions)[0]

    last_price = data_clean['Close'].iloc[-1]
    last_date = data_clean.index[-1]

    logger.info(f"\nLast trading day: {last_date.strftime('%Y-%m-%d')}")
    logger.info(f"Last closing price: ${last_price:.2f}")
    logger.info(f"\n=== {params['forecast_horizon']}-Day Price Prediction ===")

    for i, price in enumerate(future_prices):
        change = (price - last_price) / last_price * 100
        logger.info(f"Day {i+1}: ${price:.2f} ({change:+.2f}%)")

    # Summary
    avg_prediction = np.mean(future_prices)
    total_change = (future_prices[-1] - last_price) / last_price * 100

    logger.info(f"\nSummary:")
    logger.info(f"  Average predicted price: ${avg_prediction:.2f}")
    logger.info(f"  7-day expected change: {total_change:+.2f}%")

    if total_change > 2:
        logger.info("  Trend: Bullish")
    elif total_change < -2:
        logger.info("  Trend: Bearish")
    else:
        logger.info("  Trend: Neutral")


def evaluate(args):
    """Evaluate a trained model."""
    import tensorflow as tf

    logger.info("=" * 60)
    logger.info(f"Evaluating model for {args.ticker}")
    logger.info("=" * 60)

    # Load model parameters
    params_path = f'./models/{args.ticker}_params.json'
    if not os.path.exists(params_path):
        logger.error(f"Model parameters not found: {params_path}")
        return

    with open(params_path, 'r') as f:
        params = json.load(f)

    # Load model
    model_path = f'./models/{args.ticker}_final.keras'
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'AttentionLayer': AttentionLayer}
    )

    # Prepare test data
    data_clean, builder, feature_cols = prepare_data(
        args.ticker, args.start_date, args.end_date,
        params['sequence_length'], params['forecast_horizon']
    )

    # Create sequences
    (_, _), (_, _), (X_test, y_test), test_dates = \
        create_train_val_test_sequences(data_clean, builder, 0.7, 0.15, 0.15)

    # Predict
    predictions = model.predict(X_test, verbose=0)
    pred_unscaled = builder.inverse_transform_predictions(predictions)
    actual_unscaled = builder.inverse_transform_predictions(y_test)

    # Evaluate
    evaluator = ModelEvaluator(results_dir='./results', ticker=args.ticker)

    logger.info("\n=== Overall Metrics ===")
    metrics = evaluator.calculate_metrics(pred_unscaled, actual_unscaled)
    for name, value in metrics.items():
        logger.info(f"  {name.upper()}: {value:.4f}")

    logger.info("\n=== Metrics by Horizon ===")
    for h in range(params['forecast_horizon']):
        h_metrics = evaluator.calculate_metrics(pred_unscaled, actual_unscaled, horizon=h)
        logger.info(f"  Day {h+1}: RMSE=${h_metrics['rmse']:.2f}, "
                   f"MAE=${h_metrics['mae']:.2f}, MAPE={h_metrics['mape']:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Stock Price Prediction with LSTM and Attention',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a model:
    python main.py --ticker AAPL --train --epochs 100

  Make predictions:
    python main.py --ticker AAPL --predict

  Evaluate model:
    python main.py --ticker AAPL --evaluate
        """
    )

    # Main arguments
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Stock ticker symbol (default: AAPL)')

    # Actions
    parser.add_argument('--train', action='store_true',
                       help='Train a new model')
    parser.add_argument('--predict', action='store_true',
                       help='Make predictions with trained model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained model')

    # Data parameters
    parser.add_argument('--start-date', type=str, default='2019-01-01',
                       help='Start date for training data (default: 2019-01-01)')
    parser.add_argument('--end-date', type=str, default='2024-01-01',
                       help='End date for training data (default: 2024-01-01)')

    # Model parameters
    parser.add_argument('--sequence-length', type=int, default=60,
                       help='LSTM sequence length (default: 60)')
    parser.add_argument('--forecast-horizon', type=int, default=7,
                       help='Days to predict ahead (default: 7)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')

    args = parser.parse_args()

    # Validate arguments
    if not any([args.train, args.predict, args.evaluate]):
        parser.print_help()
        print("\nError: Please specify an action: --train, --predict, or --evaluate")
        sys.exit(1)

    # Execute action
    if args.train:
        train(args)
    elif args.predict:
        predict(args)
    elif args.evaluate:
        evaluate(args)


if __name__ == '__main__':
    main()
