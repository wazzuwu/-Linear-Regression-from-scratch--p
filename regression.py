import numpy as np

def regression(x, m, b):
    """
    Linear regression prediction function.
    
    Args:
        x: Input features
        m: Slope parameter
        b: Intercept parameter
    
    Returns:
        Predicted values
    """
    return m * x + b

def loss(y, y_pred):
    """
    Calculate Mean Squared Error loss.
    
    Args:
        y: True values
        y_pred: Predicted values
    
    Returns:
        Mean squared error
    """
    return np.mean((y - y_pred) ** 2)

def update(x, y, y_pred, m, b, lr=0.01):
    """
    Update parameters using gradient descent.
    
    Args:
        x: Input features
        y: True values
        y_pred: Predicted values
        m: Current slope parameter
        b: Current intercept parameter
        lr: Learning rate
    
    Returns:
        Updated slope and intercept parameters
    """
    dm = -2 * np.mean(x * (y - y_pred))
    db = -2 * np.mean(y - y_pred)
    new_m = m - lr * dm
    new_b = b - lr * db
    return new_m, new_b

def main():
    """Main training function - runs when script is executed directly."""
    # Initial parameters
    m = 0.5
    b = 20
    
    # Sample data
    x = np.array([2, 3, 5, 6, 8])
    y = np.array([65, 70, 75, 85, 90])
    
    # Training
    loss_history = []
    for i in range(1000):
        # Make prediction
        y_pred = regression(x, m, b)
        # Calculate loss
        loss_value = loss(y, y_pred)
        loss_history.append(loss_value)
        
        if i % 100 == 0:
            print(f"Iteration {i}: Loss={loss_value}, m={m}, b={b}")
        
        # Update parameters
        m, b = update(x, y, y_pred, m, b)
    
    print("\n---Training complete---")
    print(f"final loss:{loss_history[-1]:.2f}")

if __name__ == "__main__":
    main()
