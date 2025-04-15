# Load the california housing dataset
import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

# extract the features and target variable
x = df.drop('km', axis=1).values
y = df['km'].values

a, b = np.polyfit(x.flatten(), y, 1)
print(b, a)

# feature scaling
from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()
X_norm = scaler_x.fit_transform(x)

# add a column of ones to the feature matrix for the bias term
X = np.c_[np.ones(X_norm.shape[0]), X_norm]
# print(X)

# scaler_y = StandardScaler()
# Y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten() 


alpha = 0.1
epochs = int(100)

theta = np.random.randn(X.shape[1])


theta[0] = np.mean(y)
theta[1] = 0

def gradient_descent(X, Y, theta, alpha, epochs):
  m = len(y)
  errors = []
  for _ in range(epochs):

    # Calculate the predictions
    predictions = X.dot(theta)

    # Calculate MSE
    error = (1/m) * np.sum((predictions - Y) ** 2)
    errors.append(error)

    # Calculate the grdient of the cost function
    gradient = (2/m) * X.T.dot(X.dot(theta) - Y)

    theta = theta - alpha * gradient

  return theta, errors

theta, errors = gradient_descent(X, y, theta, alpha, epochs)

print(theta)

sigma_x , mu_x = scaler_x.scale_[0], scaler_x.mean_[0]
# sigma_y, mu_y = scaler_y.scale_[0], scaler_x.mean_[0]

theta[1] = theta[1] / sigma_x
theta[0] = y.mean() - theta[1]  * x.mean()

'''
Explanation:

    Normalization:

        StandardScaler is used to normalize both the input X and the output y. This scales each feature of X to have mean 0 and standard deviation 1, and scales y similarly.

    Linear Regression:

        We fit a simple linear regression model (LinearRegression from scikit-learn) to the normalized data.

    Transformation of Coefficients:

        After fitting the model, we get the coefficients from the normalized data. These are in the normalized scale, so we reverse the normalization for the coefficients and intercept:

            Coefficients: Each coefficient is scaled back by dividing by the standard deviation of the corresponding input feature in the original data (scaler_X.scale_).

            Intercept: The intercept is adjusted based on both the mean of the original y and the means of the original X.
'''

# Step 3: Convert the coefficients back to the original scale
# For each coefficient, multiply by the standard deviation of the original input feature
# theta_original = theta_norm / scaler_X.scale_

# Adjust the intercept
# intercept_original = scaler_y.mean_ - np.dot(scaler_X.mean_, theta_original)

print(theta)

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# Fit a line to the data (find the slope 'a' and intercept 'b' using np.polyfit)

# Generate the line values using the formula Y = ax + b
a , b = theta[1], theta[0]
Y_line = a * x + b

# Create scatter plot
plt.scatter(x, y, color='blue', label='Data points')

# Plot the line
plt.plot(x, Y_line, color='red', label=f'Line: $y = {a:.2f}x + {b:.2f}$')

# Add title and labels
plt.title("Scatter Plot with Line $ax + b$")
plt.xlabel("X")
plt.ylabel("Y")

# Add a legend
plt.legend()

# Display the plot
plt.savefig("testing.png")
plt.clf()

from matplotlib import pyplot as plt

# plot the error curve so we can see the descent
plt.plot(errors)

print(errors[-1])

plt.savefig('testing_lost.png')