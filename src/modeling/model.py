#!/usr/bin/env python3

import logging
import tensorflow as tf
from typing import List, Dict, Any
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense,
    LSTM, Dropout, BatchNormalization, Concatenate,
    Reshape, Bidirectional, Add, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling2D,
    SeparableConv2D, SpatialDropout1D, GaussianNoise
)

logger = logging.getLogger(__name__)

class ResidualBlock(tf.keras.layers.Layer):
    """Residual block with batch normalization and dropout."""
    def __init__(self, filters, kernel_size=3, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters, kernel_size, padding='same')
        self.bn2 = BatchNormalization()
        self.dropout = Dropout(dropout_rate)
        self.add = Add()
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.dropout(x, training=training)
        return self.add([x, inputs])

class TemporalAttention(tf.keras.layers.Layer):
    """Temporal attention mechanism for LSTM outputs."""
    def __init__(self, units):
        super(TemporalAttention, self).__init__()
        self.attention = Dense(units)
        self.context = Dense(1)
        
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        attention_weights = self.attention(inputs)
        attention_weights = tf.nn.tanh(attention_weights)
        attention_weights = self.context(attention_weights)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        return tf.reduce_sum(inputs * attention_weights, axis=1)

def create_ndvi_cnn(input_shape=(100, 100, 1)):
    """Creates the CNN branch for processing NDVI data.
    
    Args:
        input_shape: Shape of input NDVI images (height, width, channels)
        
    Returns:
        Model: CNN model for NDVI processing
    """
    inputs = layers.Input(shape=input_shape)
    
    # Add noise for regularization
    x = layers.GaussianNoise(0.1)(inputs)
    
    # Initial conv layer with separable convolutions
    x = layers.SeparableConv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Residual blocks with increasing filters
    for filters in [64, 128, 256]:
        x = ResidualBlock(filters)(x)
        x = layers.MaxPooling2D(2)(x)
    
    # Global context attention
    attention = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(256, activation='relu')(attention)
    attention = layers.Reshape((1, 1, 256))(attention)
    attention = layers.Conv2D(1, 1, activation='sigmoid')(attention)
    x = layers.Multiply()([x, attention])
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    return Model(inputs, x, name='ndvi_cnn')

def create_futures_lstm(sequence_length=8, n_features=1):
    """Creates the LSTM branch for processing futures data.
    
    Args:
        sequence_length: Number of time steps in input sequence
        n_features: Number of features per time step
        
    Returns:
        Model: LSTM model for futures processing
    """
    inputs = layers.Input(shape=(sequence_length, n_features))
    
    # Add noise for regularization
    x = layers.GaussianNoise(0.1)(inputs)
    
    # First LSTM layer with residual connection
    lstm1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    lstm1 = layers.LayerNormalization()(lstm1)
    lstm1 = layers.SpatialDropout1D(0.3)(lstm1)
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=64
    )(lstm1, lstm1)
    attention_output = layers.LayerNormalization()(attention_output + lstm1)
    
    # Second LSTM layer
    lstm2 = layers.Bidirectional(layers.LSTM(32))(attention_output)
    lstm2 = layers.LayerNormalization()(lstm2)
    
    # Temporal attention
    temporal_attention = TemporalAttention(32)(attention_output)
    temporal_attention = layers.LayerNormalization()(temporal_attention)
    
    # Combine LSTM and attention outputs
    x = layers.Concatenate()([lstm2, temporal_attention])
    
    return Model(inputs, x, name='futures_lstm')

def create_hybrid_model(config):
    """Create a hybrid CNN-LSTM model for NDVI and futures data."""
    
    # Input layers
    ndvi_input = layers.Input(shape=(config['sequence_length'], config['ndvi_features']))
    futures_input = layers.Input(shape=(config['sequence_length'], config['futures_features']))
    
    # CNN branch for NDVI data
    x1 = layers.Conv1D(filters=config['cnn_filters'], kernel_size=3, activation='relu')(ndvi_input)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)
    x1 = layers.Conv1D(filters=config['cnn_filters']*2, kernel_size=3, activation='relu')(x1)
    x1 = layers.MaxPooling1D(pool_size=2)(x1)
    
    # LSTM branch for futures data
    x2 = layers.LSTM(config['lstm_units'], return_sequences=True)(futures_input)
    x2 = layers.LSTM(config['lstm_units'])(x2)
    
    # Combine branches
    x1 = layers.Flatten()(x1)
    combined = layers.Concatenate()([x1, x2])
    
    # Dense layers
    x = layers.Dense(config['dense_units'], activation='relu')(combined)
    x = layers.Dropout(config['dropout_rate'])(x)
    x = layers.Dense(config['dense_units']//2, activation='relu')(x)
    x = layers.Dropout(config['dropout_rate'])(x)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=[ndvi_input, futures_input], outputs=output)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_futures_model(config):
    """Create a futures-only LSTM model."""
    
    # Input layer
    futures_input = layers.Input(shape=(config['sequence_length'], config['futures_features']))
    
    # LSTM layers
    x = layers.LSTM(config['lstm_units'], return_sequences=True)(futures_input)
    x = layers.LSTM(config['lstm_units'])(x)
    
    # Dense layers
    x = layers.Dense(config['dense_units'], activation='relu')(x)
    x = layers.Dropout(config['dropout_rate'])(x)
    x = layers.Dense(config['dense_units']//2, activation='relu')(x)
    x = layers.Dropout(config['dropout_rate'])(x)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=futures_input, outputs=output)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_model_summary(model: tf.keras.Model) -> str:
    """
    Get model summary as string.
    
    Args:
        model: Keras model
        
    Returns:
        Model summary string
    """
    try:
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
    except Exception as e:
        logger.error(f"Error getting model summary: {str(e)}")
        raise

# Example usage:
if __name__ == "__main__":
    # Create and compile hybrid model
    hybrid_model = create_hybrid_model()
    print("\nHybrid Model Summary:")
    hybrid_model.summary()
    
    # Create and compile futures-only model
    futures_model = create_futures_model()
    print("\nFutures-Only Model Summary:")
    futures_model.summary() 
