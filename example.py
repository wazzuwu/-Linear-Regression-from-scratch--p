#!/usr/bin/env python3
"""
Example usage of the linear regression implementation.
This script demonstrates how to use the regression functions with different datasets.
"""

import numpy as np
from regression import regression, loss, update

def demo_basic_usage():
    """Demonstrates basic usage with the default dataset."""
    print("=== Basic Linear Regression Demo ===\n")
    
    # Sample data (house size vs price example)
    x = np.array([2, 3, 5, 6, 8])  # House size (in 100 sq ft)
    y = np.array([65, 70, 75, 85, 90])  # Price (in $1000)
    
    print(f"Training data:")
    print(f"House sizes: {x} (×100 sq ft)")
    print(f"Prices: {y} (×$1000)")
    
    # Initial parameters
    m, b = 0.5, 20
    lr = 0.01
    epochs = 1000
    
    print(f"\nTraining with learning rate {lr} for {epochs} epochs...")
    print(f"Initial parameters: slope={m}, intercept={b}")
    
    # Training loop
    loss_history = []
    for i in range(epochs):
        # Make prediction
        y_pred = regression(x, m, b)
        
        # Calculate loss
        loss_value = loss(y, y_pred)
        loss_history.append(loss_value)
        
        # Print progress
        if i % 200 == 0:
            print(f"Epoch {i:3d}: Loss={loss_value:8.2f}, slope={m:.3f}, intercept={b:.3f}")
        
        # Update parameters
        m, b = update(x, y, y_pred, m, b, lr)
    
    print(f"\nTraining completed!")
    print(f"Final parameters: slope={m:.3f}, intercept={b:.3f}")
    print(f"Final loss: {loss_history[-1]:.2f}")
    
    # Make predictions on new data
    print(f"\nMaking predictions:")
    test_sizes = np.array([4, 7, 10])
    test_predictions = regression(test_sizes, m, b)
    
    for size, pred in zip(test_sizes, test_predictions):
        print(f"  House size: {size}×100 sq ft → Predicted price: ${pred:.1f}k")

def demo_custom_dataset():
    """Demonstrates usage with a custom dataset."""
    print("\n\n=== Custom Dataset Demo ===\n")
    
    # Create a synthetic dataset with known relationship: y = 3x + 5 + noise
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y = 3 * x + 5 + np.random.normal(0, 1, 20)  # Adding some noise
    
    print(f"Custom dataset: y ≈ 3x + 5 (with noise)")
    print(f"Data points: {len(x)} samples")
    
    # Train the model
    m, b = 0.0, 0.0  # Start from zero
    lr = 0.01
    epochs = 500
    
    for i in range(epochs):
        y_pred = regression(x, m, b)
        loss_value = loss(y, y_pred)
        
        if i % 100 == 0:
            print(f"Epoch {i:3d}: Loss={loss_value:6.2f}, slope={m:.3f}, intercept={b:.3f}")
        
        m, b = update(x, y, y_pred, m, b, lr)
    
    print(f"\nTrue relationship: y = 3x + 5")
    print(f"Learned relationship: y = {m:.2f}x + {b:.2f}")
    print(f"Parameter recovery: slope error = {abs(3-m):.3f}, intercept error = {abs(5-b):.3f}")

if __name__ == "__main__":
    demo_basic_usage()
    demo_custom_dataset()
    
    print("\n" + "="*60)
    print("Experiment with different learning rates, datasets, and parameters!")
    print("Try modifying the values in this script to see how it affects training.")