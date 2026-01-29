"""
LSTM Model with Attention Mechanism for Stock Price Prediction.

This module implements a multi-layer LSTM architecture with an attention
mechanism for improved time-series forecasting. The attention layer allows
the model to focus on the most relevant time steps.
"""

import logging
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


class AttentionLayer(layers.Layer):
    """
    Custom Attention Layer for sequence models.

    This layer implements a soft attention mechanism that learns to weight
    the importance of each time step in the input sequence.

    The attention mechanism computes:
        attention_weights = softmax(W * tanh(V * hidden_states + b))
        context_vector = sum(attention_weights * hidden_states)

    Attributes:
        units (int): Number of attention units.
        return_attention (bool): Whether to return attention weights.
    """

    def __init__(
        self,
        units: int = 64,
        return_attention: bool = False,
        **kwargs
    ):
        """
        Initialize the Attention Layer.

        Args:
            units: Number of hidden units in the attention mechanism.
            return_attention: If True, returns both context and attention weights.
        """
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.return_attention = return_attention

    def build(self, input_shape):
        """Build the layer weights."""
        # input_shape: (batch_size, time_steps, features)
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        """
        Forward pass of the attention layer.

        Args:
            inputs: Input tensor of shape (batch_size, time_steps, features).
            mask: Optional mask tensor.

        Returns:
            Context vector or tuple of (context, attention_weights).
        """
        # Score calculation: (batch, time_steps, units)
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)

        # Attention weights: (batch, time_steps, 1)
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=1),
            axis=1
        )

        # Context vector: weighted sum of inputs
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)

        if self.return_attention:
            return context_vector, tf.squeeze(attention_weights, axis=-1)
        return context_vector

    def get_config(self):
        """Get layer configuration."""
        config = super(AttentionLayer, self).get_config()
        config.update({
            'units': self.units,
            'return_attention': self.return_attention
        })
        return config


class MultiHeadAttentionLayer(layers.Layer):
    """
    Multi-Head Attention Layer for enhanced attention capabilities.

    Uses multiple attention heads to capture different aspects of the
    temporal dependencies in the data.
    """

    def __init__(
        self,
        num_heads: int = 4,
        key_dim: int = 32,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Initialize Multi-Head Attention.

        Args:
            num_heads: Number of attention heads.
            key_dim: Dimension of each attention head.
            dropout: Dropout rate.
        """
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout
        )
        self.layernorm = layers.LayerNormalization()
        self.dropout_layer = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        """Forward pass with residual connection."""
        attention_output = self.mha(inputs, inputs, training=training)
        attention_output = self.dropout_layer(attention_output, training=training)
        return self.layernorm(inputs + attention_output)

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout
        })
        return config


def build_lstm_attention_model(
    input_shape: Tuple[int, int],
    output_steps: int = 7,
    lstm_units: List[int] = [128, 64, 32],
    attention_units: int = 64,
    dense_units: List[int] = [64, 32],
    dropout_rate: float = 0.2,
    recurrent_dropout: float = 0.2,
    l2_reg: float = 0.001,
    use_bidirectional: bool = False,
    use_multi_head_attention: bool = False,
    num_attention_heads: int = 4,
    learning_rate: float = 0.001
) -> Model:
    """
    Build an LSTM model with attention mechanism.

    This function creates a multi-layer LSTM architecture with an attention
    layer for improved time-series forecasting.

    Args:
        input_shape: Shape of input data (sequence_length, n_features).
        output_steps: Number of time steps to predict (forecast horizon).
        lstm_units: List of units for each LSTM layer.
        attention_units: Number of units in the attention layer.
        dense_units: List of units for dense layers after attention.
        dropout_rate: Dropout rate for regularization.
        recurrent_dropout: Dropout rate for recurrent connections.
        l2_reg: L2 regularization factor.
        use_bidirectional: Whether to use bidirectional LSTMs.
        use_multi_head_attention: Whether to use multi-head attention.
        num_attention_heads: Number of attention heads if using multi-head.
        learning_rate: Learning rate for optimizer.

    Returns:
        Compiled Keras Model.
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input')
    x = inputs

    # Add batch normalization to input
    x = layers.BatchNormalization(name='input_bn')(x)

    # LSTM layers
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1) or True  # Always return sequences for attention

        lstm_layer = layers.LSTM(
            units=units,
            return_sequences=return_sequences,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
            name=f'lstm_{i+1}'
        )

        if use_bidirectional:
            x = layers.Bidirectional(lstm_layer, name=f'bilstm_{i+1}')(x)
        else:
            x = lstm_layer(x)

        # Add layer normalization after each LSTM
        x = layers.LayerNormalization(name=f'ln_{i+1}')(x)

    # Attention mechanism
    if use_multi_head_attention:
        x = MultiHeadAttentionLayer(
            num_heads=num_attention_heads,
            key_dim=attention_units // num_attention_heads,
            dropout=dropout_rate,
            name='multi_head_attention'
        )(x)
        # Global average pooling after multi-head attention
        attention_output = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        attention_weights = None
    else:
        # Custom attention layer
        attention_layer = AttentionLayer(
            units=attention_units,
            return_attention=True,
            name='attention'
        )
        attention_output, attention_weights = attention_layer(x)

    x = attention_output

    # Dense layers
    for i, units in enumerate(dense_units):
        x = layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'dense_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'dense_bn_{i+1}')(x)
        x = layers.Dropout(dropout_rate, name=f'dense_dropout_{i+1}')(x)

    # Output layer for multi-step forecasting
    outputs = layers.Dense(
        output_steps,
        activation='linear',
        name='output'
    )(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='lstm_attention_model')

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mape']
    )

    logger.info(f"Built LSTM-Attention model with {model.count_params():,} parameters")

    return model


def build_attention_model_with_weights(
    input_shape: Tuple[int, int],
    output_steps: int = 7,
    lstm_units: List[int] = [128, 64],
    attention_units: int = 64,
    dropout_rate: float = 0.2,
    l2_reg: float = 0.001,
    learning_rate: float = 0.001
) -> Tuple[Model, Model]:
    """
    Build LSTM model that also outputs attention weights for visualization.

    Args:
        input_shape: Shape of input data (sequence_length, n_features).
        output_steps: Number of time steps to predict.
        lstm_units: Units for LSTM layers.
        attention_units: Units in attention layer.
        dropout_rate: Dropout rate.
        l2_reg: L2 regularization.
        learning_rate: Learning rate.

    Returns:
        Tuple of (main_model, attention_model) where attention_model
        outputs both predictions and attention weights.
    """
    inputs = layers.Input(shape=input_shape, name='input')
    x = inputs

    x = layers.BatchNormalization()(x)

    # LSTM layers
    for i, units in enumerate(lstm_units):
        x = layers.LSTM(
            units=units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'lstm_{i+1}'
        )(x)
        x = layers.LayerNormalization()(x)

    # Attention layer that returns weights
    attention_layer = AttentionLayer(
        units=attention_units,
        return_attention=True,
        name='attention'
    )
    context, attention_weights = attention_layer(x)

    # Dense layers
    x = layers.Dense(64, activation='relu')(context)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(32, activation='relu')(x)

    # Output
    predictions = layers.Dense(output_steps, name='predictions')(x)

    # Main model for training
    main_model = Model(inputs=inputs, outputs=predictions, name='main_model')
    main_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    # Attention model for visualization
    attention_model = Model(
        inputs=inputs,
        outputs=[predictions, attention_weights],
        name='attention_model'
    )

    return main_model, attention_model


def get_callbacks(
    model_path: str = 'models/best_model.keras',
    patience: int = 15,
    min_delta: float = 0.0001,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-7,
    tensorboard_log_dir: Optional[str] = None
) -> List[keras.callbacks.Callback]:
    """
    Get training callbacks for the model.

    Args:
        model_path: Path to save the best model.
        patience: Epochs to wait before early stopping.
        min_delta: Minimum change to qualify as improvement.
        reduce_lr_patience: Epochs to wait before reducing LR.
        reduce_lr_factor: Factor to reduce LR by.
        min_lr: Minimum learning rate.
        tensorboard_log_dir: Directory for TensorBoard logs.

    Returns:
        List of Keras callbacks.
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    if tensorboard_log_dir:
        callbacks.append(
            TensorBoard(
                log_dir=tensorboard_log_dir,
                histogram_freq=1,
                write_graph=True
            )
        )

    return callbacks


class LSTMAttentionPredictor:
    """
    High-level wrapper for LSTM-Attention stock price prediction.

    This class provides a simple interface for building, training,
    and using LSTM-Attention models.

    Example:
        >>> predictor = LSTMAttentionPredictor(sequence_length=60, forecast_horizon=7)
        >>> predictor.build_model(n_features=20)
        >>> history = predictor.fit(X_train, y_train, X_val, y_val)
        >>> predictions = predictor.predict(X_test)
    """

    def __init__(
        self,
        sequence_length: int = 60,
        forecast_horizon: int = 7,
        model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the predictor.

        Args:
            sequence_length: Number of time steps in input sequence.
            forecast_horizon: Number of steps to predict.
            model_config: Configuration for the model architecture.
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model_config = model_config or {}
        self.model = None
        self.attention_model = None
        self.history = None

    def build_model(
        self,
        n_features: int,
        with_attention_output: bool = True
    ) -> Model:
        """
        Build the LSTM-Attention model.

        Args:
            n_features: Number of input features.
            with_attention_output: Whether to create attention output model.

        Returns:
            The built model.
        """
        input_shape = (self.sequence_length, n_features)

        if with_attention_output:
            self.model, self.attention_model = build_attention_model_with_weights(
                input_shape=input_shape,
                output_steps=self.forecast_horizon,
                **self.model_config
            )
        else:
            self.model = build_lstm_attention_model(
                input_shape=input_shape,
                output_steps=self.forecast_horizon,
                **self.model_config
            )

        return self.model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        callbacks: Optional[List] = None,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            epochs: Maximum number of epochs.
            batch_size: Batch size.
            callbacks: List of callbacks.
            verbose: Verbosity level.

        Returns:
            Training history.
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        if callbacks is None:
            callbacks = get_callbacks()

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features of shape (samples, sequence_length, n_features).

        Returns:
            Predictions of shape (samples, forecast_horizon).
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        return self.model.predict(X, verbose=0)

    def predict_with_attention(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions and return attention weights.

        Args:
            X: Input features.

        Returns:
            Tuple of (predictions, attention_weights).
        """
        if self.attention_model is None:
            raise ValueError("Attention model not available.")
        return self.attention_model.predict(X, verbose=0)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            X: Test features.
            y: Test targets.

        Returns:
            Dictionary of metric names to values.
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")

        results = self.model.evaluate(X, y, verbose=0)
        metric_names = ['loss'] + [m.name for m in self.model.metrics]

        return dict(zip(metric_names, results))

    def save(self, path: str):
        """Save the model to disk."""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load a model from disk."""
        self.model = keras.models.load_model(
            path,
            custom_objects={
                'AttentionLayer': AttentionLayer,
                'MultiHeadAttentionLayer': MultiHeadAttentionLayer
            }
        )
        logger.info(f"Model loaded from {path}")

    def summary(self):
        """Print model summary."""
        if self.model is not None:
            self.model.summary()


if __name__ == "__main__":
    # Example usage
    print("Building LSTM-Attention Model...")

    # Example input shape
    sequence_length = 60
    n_features = 20
    forecast_horizon = 7

    # Build model
    model = build_lstm_attention_model(
        input_shape=(sequence_length, n_features),
        output_steps=forecast_horizon,
        lstm_units=[128, 64, 32],
        attention_units=64,
        dense_units=[64, 32],
        dropout_rate=0.2
    )

    model.summary()

    # Test with random data
    X_test = np.random.randn(10, sequence_length, n_features)
    predictions = model.predict(X_test, verbose=0)
    print(f"\nTest input shape: {X_test.shape}")
    print(f"Predictions shape: {predictions.shape}")

    # Build model with attention weights output
    print("\n\nBuilding model with attention weights output...")
    main_model, attention_model = build_attention_model_with_weights(
        input_shape=(sequence_length, n_features),
        output_steps=forecast_horizon
    )

    preds, attention = attention_model.predict(X_test, verbose=0)
    print(f"Predictions shape: {preds.shape}")
    print(f"Attention weights shape: {attention.shape}")
