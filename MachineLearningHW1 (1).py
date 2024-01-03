import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Load the dataset
data = pd.read_csv('HW1_house_data(1).csv')

# Preprocessing Step: Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)


# Step 2(a): Report statistics about the dataset
print(data.describe())

# Step 2(b): Display the head of the dataset
print(data.head())

# Step 3: Extract the input feature into X and the output into y
X = data['area'].values.reshape(-1, 1)
y = data[['price']].values

# Step 4: Draw a scatter plot showing the feature (X) against the output (Y
plt.figure(figsize=(8, 8))
plt.scatter(X, y, marker='x')
plt.title('Original Training Dataset')
plt.xlabel('X (House Area)')
plt.ylabel('Y (Cost)')
plt.grid(True)
plt.show()

# Step 5(a): Normalize the input features
X_normalized = (X - X.mean()) / X.std()

# Step 5(b): Plot the data after normalization
plt.figure(figsize=(8, 6))
plt.scatter(X_normalized, y, marker='x')
plt.title('Normalized Training Dataset')
plt.xlabel('Normalized X (House Area)')
plt.ylabel('Y (Cost)')
plt.grid(True)
plt.show()

# Step 6(a): Code the batch gradient descent algorithm
def batch_gradient_descent(X, y, eta, n_iterations):
    m = len(y)
    theta0 = 0
    theta1 = 0
    
    for i in range(n_iterations):
        predictions = theta0 + theta1 * X
        error = predictions - y
        theta0 = theta0 - (eta / m) * error.sum()
        theta1 = theta1 - (eta / m) * (error * X).sum()
    
    return theta0, theta1

# Step 6(b): Run gradient descent
eta = 0.01 # learning rate
n_iterations = 1000
theta0, theta1 = batch_gradient_descent(X_normalized, y, eta, n_iterations)

# Step 6(c): Report the values of theta0 and theta1 found
print("Batch Gradient Descent:")
print(f'Theta0: {[theta0]}')
print(f'Theta1: {[theta1]}' )

# Step 7: Normalize the area before predicting the price of a house with an area of 2000
area_to_predict = (2000 - X.mean()) / X.std()
predicted_price = theta0 + theta1 * area_to_predict
print(f'Predicted price of a house with area 2000: {[predicted_price]}')

# Step 8: Plot the predicted versus actual values
plt.figure(figsize=(8, 6))
plt.scatter(X_normalized, y, color='blue', marker='o', label='Actual Data')
plt.plot(X_normalized, theta0 + theta1 * X_normalized, color='red', label='Regression Line')
plt.title('Predicted vs. Actual House Prices')
plt.xlabel('Area (sqft)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Step 9(a): Learn a new regression model using LinearRegression from sklearn
lin_reg  = LinearRegression()
lin_reg .fit(X_normalized, y)

# Step 9(b): Print the LinearRegression parameters
print("Linear Regression:")
print(f'Theta0: {lin_reg.intercept_}')
print(f'Theta1: {lin_reg.coef_[0]}')

# Step 10(a): Use sklearn model to predict the price of a house with an area of 2000
predicted_price_sklearn = lin_reg.predict(np.array([[area_to_predict]]))

# Step 10(b): Display the outputt
print(f'Predicted price of a house with area 2000: {predicted_price_sklearn[0]}')

# Step 11(a): Implement linear regression using the normal equation method
X_normalized = np.insert(X_normalized, 0, 1, axis=1)  # Add a column of ones for bias term
theta = np.linalg.inv(X_normalized.T.dot(X_normalized)).dot(X_normalized.T).dot(y)

# Step 11(b): Print the resulting coefficients and predicted price for a house of area = 2000
print("Normal Equation:")
print(f'Theta0: {theta[0]}')
print(f'Theta1: {theta[1]}')
predicted_price_normalEquation = theta[0] + theta[1] * area_to_predict
print(f'Predicted price of a house with area 2000: {predicted_price_normalEquation}')
