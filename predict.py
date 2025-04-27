# predict.py
"""
Prediction functionality for the face mask detection classifier.
Contains functions for making predictions on new images.
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from config import RESIZE_HEIGHT, RESIZE_WIDTH


def preprocess_image(image_path):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Preprocessed image tensor or None if processing fails
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
            
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        
        # Apply the same transformation as for validation
        transform = transforms.Compose([
            transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image)
        return image_tensor
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def predict_image(model, image_path, device, class_names):
    """
    Make a prediction for a single image.
    
    Args:
        model: Trained model
        image_path: Path to the image file
        device: Device to run inference on
        class_names: List of class names
    
    Returns:
        Predicted class index, class name, and confidence
    """
    # Preprocess the image
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        return None, None, None
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get prediction details
    predicted_idx = predicted.item()
    predicted_class = class_names[predicted_idx] if predicted_idx < len(class_names) else "Unknown"
    confidence_score = confidence.item()
    
    return predicted_idx, predicted_class, confidence_score


def predict_batch(model, image_paths, device, class_names):
    """
    Make predictions for a batch of images.
    
    Args:
        model: Trained model
        image_paths: List of paths to image files
        device: Device to run inference on
        class_names: List of class names
    
    Returns:
        List of prediction results (class index, name, confidence)
    """
    # Preprocess all images
    image_tensors = []
    valid_paths = []
    
    for path in image_paths:
        tensor = preprocess_image(path)
        if tensor is not None:
            image_tensors.append(tensor)
            valid_paths.append(path)
    
    if not image_tensors:
        print("No valid images to process")
        return []
    
    # Stack tensors into a batch
    batch = torch.stack(image_tensors).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        outputs = model(batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
    
    # Collect results
    results = []
    for i, (path, pred_idx, conf) in enumerate(zip(valid_paths, predictions, confidences)):
        pred_class = class_names[pred_idx.item()] if pred_idx.item() < len(class_names) else "Unknown"
        results.append({
            'image_path': path,
            'predicted_index': pred_idx.item(),
            'predicted_class': pred_class,
            'confidence': conf.item()
        })
    
    return results


def visualize_predictions(image_paths, predictions, class_names, output_dir=None):
    """
    Visualize prediction results with images and labels.
    
    Args:
        image_paths: List of image paths
        predictions: List of prediction results
        class_names: List of class names
        output_dir: Directory to save visualizations (optional)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if not predictions:
        print("No predictions to visualize")
        return
        
    n_images = len(image_paths)
    cols = min(5, n_images)
    rows = (n_images + cols - 1) // cols
    
    plt.figure(figsize=(15, 3*rows))
    
    for i, pred in enumerate(predictions):
        path = pred['image_path']
        
        try:
            # Load image
            img = Image.open(path).convert('RGB')
            
            # Create subplot
            plt.subplot(rows, cols, i+1)
            plt.imshow(img)
            
            # Add prediction label
            pred_class = pred['predicted_class']
            confidence = pred['confidence']
            title = f"{pred_class} ({confidence:.2f})"
            plt.title(title, color='green' if confidence > 0.7 else 'red')
            plt.axis('off')
        except Exception as e:
            print(f"Error visualizing image {path}: {e}")
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'predictions.png'))
        plt.close()
    else:
        plt.show()
