# Linear Regression Project

This repository contains a minimal implementation of linear regression in Python. It is designed to help you understand the basics of regression analysis and how to apply it to simple datasets.

 Regression analysis is a statistical method used to examine the relationship between one or more independent variables and a dependent variable, enabling the prediction of outcomes and the understanding of how changes in the independent variables affect the dependent variable.

## Files
- `regression.py`: Main script for linear regression.
## How to Use
1. Clone or download this repository.
2. Run `regression.py` to see linear regression in action.
3. Explore and modify the code to experiment with different datasets or parameters.

## What is Linear Regression?
Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It finds the best-fitting straight line (the regression line) through the data points.

## Loss and Update Functions
- **loss(y, y_pred)**: Calculates the mean squared error (MSE) between the true values (`y`) and the predicted values (`y_pred`). This measures how well the model fits the data.
- **update(x, y, y_pred, m, b, lr=0.01)**: Performs one step of gradient descent to update the parameters `m` (slope) and `b` (intercept). It computes the gradients and adjusts the parameters to minimize the loss.

```python
# Loss function
def loss(y, y_pred):
	return np.mean((y - y_pred) ** 2)

# Update function
def update(x, y, y_pred, m, b, lr=0.01):
	dm = -2 * np.mean(x * (y - y_pred))
	db = -2 * np.mean(y - y_pred)
	new_m = m - lr * dm
	new_b = b - lr * db
	return new_m, new_b
```

## Example
Suppose you have data points representing the relationship between hours studied and exam scores. Linear regression can help you predict exam scores based on hours studied.

## Fun Fact
Did you know? The concept of linear regression dates back to the early 19th century and was first described by Adrien-Marie Legendre and Carl Friedrich Gauss. The term "regression" was coined by Francis Galton when studying the relationship between parents' and children's heights!
