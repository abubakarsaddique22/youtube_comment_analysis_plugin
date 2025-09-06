# src/model/model_building.py

import os
import pickle
import logging
import lightgbm as lgb
import yaml

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters: %s', e)
        raise


def load_features(features_path: str, labels_path: str):
    """Load precomputed features and labels from pickle files."""
    try:
        with open(features_path, 'rb') as f:
            X = pickle.load(f)
        with open(labels_path, 'rb') as f:
            y = pickle.load(f)
        logger.debug('Features and labels loaded successfully.')
        return X, y
    except Exception as e:
        logger.error('Error loading features/labels: %s', e)
        raise


def train_lgbm(X, y, params: dict):
    """Train a LightGBM model."""
    try:
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(set(y)),
            metric='multi_logloss',
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators'],
            is_unbalance=True,
            class_weight='balanced',
            reg_alpha= 0.1,
            reg_lambda=0.1
        )
        model.fit(X, y)
        logger.debug('LightGBM model training completed.')
        return model
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def save_model(model, model_path: str):
    """Save the trained model to a pickle file."""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug('Model saved to %s', model_path)
    except Exception as e:
        logger.error('Error saving model: %s', e)
        raise


def main():
    """Main model-building pipeline."""
    try:
        print("hooooo")
        root_dir = get_root_directory()
        print(root_dir)
        # Load parameters
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        model_params = params['model_building']

        # Load precomputed features and labels
        features_path = os.path.join(root_dir, 'data/processed/X_train_tfidf.pkl')
        labels_path = os.path.join(root_dir, 'data/processed/y_train.pkl')
        X_train, y_train = load_features(features_path, labels_path)
        
        # Train the LightGBM model
        model = train_lgbm(X_train, y_train, model_params)

        # Save the trained model
        model_path = os.path.join(root_dir, 'models/lgbm_model.pkl')
        save_model(model, model_path)

        logger.debug('Model building pipeline completed successfully.')

    except Exception as e:
        logger.error('Model building pipeline failed: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
