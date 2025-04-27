"""
Main script to train the face mask detection model and make predictions.
"""

import os
import argparse
import torch
import random
import numpy as np
from model import create_model
from dataset import create_data_loaders
from train import train_model, evaluate_model, plot_training_history
from predict import predict_batch, visualize_predictions
from config import CLASS_NAMES

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='Face Mask Detection')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help='Operation mode: train or predict')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for prediction or resuming training')
    parser.add_argument('--predict_dir', type=str, default=None,
                        help='Directory containing images for prediction')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
    print(f"Using device: {device}")
    
    try:
        if args.mode == 'train':
            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Create data loaders
            train_loader, val_loader, class_names = create_data_loaders(args.data_dir)
            print(f"Classes: {class_names}")
            
            # Create model
            model = create_model()
            model = model.to(device)
            
        # Load checkpoint if provided
        if args.checkpoint and os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            # Check if checkpoint is a state_dict directly or has model_state_dict key
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Try direct loading - the checkpoint itself is the state dict
                    try:
                        model.load_state_dict(checkpoint)
                        print(f"Loaded model state directly from checkpoint")
                    except Exception as e:
                        print(f"Failed to load model: {e}")
                        return
            print(f"Loaded model checkpoint from {args.checkpoint}")
            
            # Use class names from config, or try to get them from data directory if available
            class_names = CLASS_NAMES
            try:
                _, _, data_class_names = create_data_loaders(args.data_dir)
                class_names = data_class_names
            except Exception as e:
                print(f"Could not load class names from data directory, using defaults: {class_names}")
            
            print(f"Classes: {class_names}")
            
            # Get image paths for prediction
            if not args.predict_dir or not os.path.exists(args.predict_dir):
                print("Error: Prediction directory required")
                return
            
            image_paths = [
                os.path.join(args.predict_dir, f)
                for f in os.listdir(args.predict_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            if not image_paths:
                print("No images found in the prediction directory")
                return
            
            print(f"Making predictions on {len(image_paths)} images...")
            predictions = predict_batch(model, image_paths, device, class_names)
            
            # Visualize predictions
            output_dir = os.path.join(args.output_dir, 'predictions')
            visualize_predictions(image_paths, predictions, class_names, output_dir)
            print(f"Saved prediction visualizations to {output_dir}")
            
            # Print prediction results
            for pred in predictions:
                print(f"Image: {os.path.basename(pred['image_path'])}, "
                      f"Predicted: {pred['predicted_class']}, "
                      f"Confidence: {pred['confidence']:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()