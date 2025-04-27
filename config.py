# config.py
"""
Configuration file for the face mask detection model.
Contains all hyperparameters and constants used throughout the project.
"""

# Training hyperparameters
BATCH_SIZE = 64  
EPOCHS = 20  
LEARNING_RATE = 0.001  
WEIGHT_DECAY = 1e-5  

# Data processing parameters
RESIZE_HEIGHT = 224
RESIZE_WIDTH = 224
INPUT_CHANNELS = 3
NUM_WORKERS = 4  
VALIDATION_SPLIT = 0.2  

# Model parameters
NUM_CLASSES = 2  
CLASS_NAMES = ['with_mask', 'without_mask']  # Added explicit class names

# Device configuration
USE_CUDA = True 