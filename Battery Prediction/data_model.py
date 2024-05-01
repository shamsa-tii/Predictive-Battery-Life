import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function for graph design
def graph_settings(xlabel, ylabel, title="", legend_location='best', walls=[0, 0, 1, 1]):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.spines["top"].set_visible(walls[0])
    ax.spines["right"].set_visible(walls[1])
    ax.spines["left"].set_visible(walls[2])
    ax.spines["bottom"].set_visible(walls[3])
    ax.minorticks_on()
    ax.set_xlabel(xlabel, fontsize=20, labelpad=4, fontfamily='monospace')
    ax.set_ylabel(ylabel, fontsize=20, labelpad=4, fontfamily='monospace')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax.set_title(title, fontsize=16, fontfamily='monospace')
    ax.xaxis.set_tick_params(length=5, width=1)
    ax.yaxis.set_tick_params(length=5, width=1)

# Function to fit the data and plot results
def fit_data(model_name, degree=2, file_path='/home/shamsahamad/Desktop/Sam flights/average.csv'):
    # Load the dataset
    data = pd.read_csv(file_path)
    X = data.drop('remaining', axis=1).values.reshape(-1, data.shape[1] - 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = data['remaining'].values.reshape(-1, 1)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dictionary to hold the models
    models = {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression": make_pipeline(PolynomialFeatures(degree), LinearRegression()),
        "SVR": make_pipeline(StandardScaler(), SVR()),
        "Random Forest Regression": RandomForestRegressor(),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor())
    }

    # Initialize and train the model
    model = models[model_name]
    model.fit(X_train, y_train.ravel())

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse} for model : {model_name}')

    # Plot settings and actual vs. predicted values
    graph_settings('Time', 'Remaining Battery', f'{model_name} Model')
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_test)

    plt.scatter(X_pca, y_test, color='blue', label='Actual')

    # Check if the model is RandomForestRegressor and plot the mean of the trees' predictions
    if isinstance(model, RandomForestRegressor):
        mean_pred = np.mean([tree.predict(X_test) for tree in model.estimators_], axis=0)
        plt.scatter(X_pca, mean_pred, color='red', linewidth=2, label='Predicted - Mean of Trees')
    else:
        plt.scatter(X_pca, y_pred, color='red', linewidth=2, label='Predicted')

    # Display the legend and show the plot
    plt.legend()
    plt.show()

# Main script execution
if __name__ == "__main__":
    # List of model names to iterate over
    model_names = ["KNN", "Random Forest Regression", "SVR", "Polynomial Regression", "Linear Regression"]

    # Loop through each model and fit the data
    for name in model_names:
        fit_data(name, degree=3)  # Adjust the degree for polynomial regression as needed
