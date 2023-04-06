import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load diabetes dataset and split into train and test datasets
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("rf_diabetes_hyperparameter_search")

# Create a new MLflow run
with mlflow.start_run(run_name="rf_diabetes_example") as run:
    # Set run description
    desc = "Random Forest Regressor for diabetes dataset"
    mlflow.set_tag("description", desc)
    # Enable auto-logging
    mlflow.sklearn.autolog()
    # Define hyperparameters to search
    n_estimators = [10, 50, 100]
    max_depth = [2, 5, 10]
    # Perform hyperparameter search
    best_score = None
    best_params = {}
    best_model_uri = None
    for n in n_estimators:
        for d in max_depth:
            # Create nested run
            with mlflow.start_run(nested=True):
                # Train RandomForestRegressor with hyperparameters
                rf = RandomForestRegressor(n_estimators=n, max_depth=d)
                rf.fit(X_train, y_train)
                # Evaluate model and log metrics
                y_pred = rf.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mlflow.log_metric("mse", mse)
                mlflow.log_param("n_estimators", n)
                mlflow.log_param("max_depth", d)
                # Check if this is the best model so far
                if best_score is None or mse < best_score:
                    best_score = mse
                    best_params = {"n_estimators": n, "max_depth": d}
                    best_model_uri = mlflow.get_artifact_uri("model")

    # Register the best model with the Model Registry
    if best_model_uri is not None:
        mlflow.register_model(best_model_uri, "diabetes_rf_model")
        # check the Model Registry API to see the registered model
        mlflow.search_registered_models()


model_name = "diabetes_rf_model"
model_stage = "Production"
port_number = 1234

# Load the registered model
loaded_model = mlflow.pyfunc.load_model(f'models:/{model_name}/{model_stage}')
#print(loaded_model)
# Create MLflow Project and specify dependencies
# ...
# Run MLflow Project using mlflow run command
# ...

