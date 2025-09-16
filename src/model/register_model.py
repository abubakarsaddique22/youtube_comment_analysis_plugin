import json
import mlflow
import logging
import os
from mlflow.tracking import MlflowClient

# Set up MLflow tracking URI (your AWS MLflow server)
mlflow.set_tracking_uri('http://13.61.25.27:5000/')

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry and move to Staging."""
    try:
        client = MlflowClient()

        # Full model URI from the run
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug(f"Registering model from URI: {model_uri}")

        # Ensure registered model exists (idempotent)
        try:
            client.get_registered_model(model_name)
            logger.debug(f"Registered model '{model_name}' already exists.")
        except mlflow.exceptions.RestException:
            client.create_registered_model(model_name)
            logger.debug(f"Created new registered model '{model_name}'.")

        # Create a new model version
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=model_info['run_id']
        )
        logger.debug(f"Model version {model_version.version} created for '{model_name}'.")

        # Transition model to "Staging"
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=True  # ✅ old versions move to Archived
        )
        logger.debug(f"Model {model_name} version {model_version.version} transitioned to Staging.")

    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = os.path.join("reports", "experiment_info.json")  # ✅ updated to reports path
        model_info = load_model_info(model_info_path)
        
        model_name = "yt_chrome_plugin_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
