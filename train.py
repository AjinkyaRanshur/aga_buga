# train.py
"""
Training functionality for the face mask detection classifier.
Contains functions for training, validating, and evaluating the model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
from config import LEARNING_RATE, WEIGHT_DECAY, EPOCHS, CLASS_NAMES


def train_model(model, train_loader, val_loader, device, checkpoint_dir='checkpoints'):
    """
    Train the model on the provided data loaders.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cuda/cpu)
        checkpoint_dir: Directory to save model checkpoints
    
    Returns:
        Trained model and training history
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # LR scheduler to reduce learning rate when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # Initialize best validation metrics
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Training loop
    print(f"Starting training for {EPOCHS} epochs")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Use tqdm for progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]") as t:
            for i, (images, labels) in enumerate(t):
                # Move data to device
                images, labels = images.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                t.set_postfix(loss=train_loss/(i+1), acc=train_correct/train_total)
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"Time: {time_elapsed:.2f}s, "
              f"LR: {optimizer.param_groups[0]['lr']:.8f}")
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            print(f"Validation loss decreased from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, os.path.join(checkpoint_dir, 'best_model_loss.pth'))
        
        if val_accuracy > best_val_accuracy:
            print(f"Validation accuracy increased from {best_val_accuracy:.4f} to {val_accuracy:.4f}. Saving model...")
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, os.path.join(checkpoint_dir, 'best_model_accuracy.pth'))
    
    # Save final model
    torch.save({
        'epoch': EPOCHS-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
    }, os.path.join(checkpoint_dir, 'final_weights.pth'))
    
    # Plot training history
    plot_training_history(history, os.path.join(checkpoint_dir, '../plots'))
    
    return model, history


def validate(model, val_loader, criterion, device):
    """
    Validate the model on the validation set.
    
    Args:
        model: Model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (cuda/cpu)
    
    Returns:
        Validation loss and accuracy
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Update statistics
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    
    return val_loss, val_accuracy


def evaluate_model(model, val_loader, device, output_dir=None):
    """
    Evaluate the model on the validation set with detailed metrics.
    
    Args:
        model: Model to evaluate
        val_loader: DataLoader for validation data
        device: Device to evaluate on (cuda/cpu)
        output_dir: Directory to save evaluation results (optional)
    
    Returns:
        Dictionary of evaluation metrics
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Add batch to lists
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert lists to arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Store metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }
    
    # Print metrics
    print(f"Evaluation Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:")
    print(conf_matrix)
    
    # Generate confusion matrix plot
    if output_dir:
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        class_names = CLASS_NAMES
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add values to cells
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
    
    return metrics


def plot_training_history(history, output_dir=None):
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary containing training history
        output_dir: Directory to save plots (optional)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
        plt.close()
    else:
        plt.show()
    
    # Plot learning rate
    plt.figure(figsize=(10, 4))
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
        plt.close()
    else:
        plt.show()