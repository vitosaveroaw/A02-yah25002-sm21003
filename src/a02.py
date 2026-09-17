from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Define features (X) and target (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the training and testing sets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Initialize MLPRegressor with early stopping and custom hyperparameters
mlp_regressor = MLPRegressor(
    hidden_layer_sizes=(10, 5), # Custom hyperparameter: two hidden layers with 10 and 5 neurons
    activation='relu',
    solver='adam',
    early_stopping=True,          # Enable early stopping
    max_iter=1000,                # Maximum number of iterations for the solver to converge
    batch_size=100,                # Custom hyperparameter: Use mini-batches of size 100
    random_state=42
)

# Train the model
mlp_regressor.fit(X_train, y_train)
print("MLPRegressor model trained successfully with early stopping.")