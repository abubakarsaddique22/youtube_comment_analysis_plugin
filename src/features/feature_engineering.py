# src/model/feature_engineering.py

import os
import pickle
import yaml
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('feature_engineering_errors.log')
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
    """
    Load parameters from a YAML file.

    Args:
        params_path (str): Path to the YAML file.

    Returns:
        dict: Dictionary of parameters.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f'Parameters loaded from {params_path}')
        return params
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data and fill NaN values.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug(f'Data loaded from {file_path}')
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """
    Apply TF-IDF transformation to text data.

    Args:
        train_data (pd.DataFrame): Training data containing 'clean_comment'.
        max_features (int): Maximum number of TF-IDF features.
        ngram_range (tuple): N-gram range for TF-IDF.

    Returns:
        tuple: TF-IDF features matrix and labels.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        X_train_tfidf = vectorizer.fit_transform(X_train)

        logger.debug(f'TF-IDF transformation completed. Shape: {X_train_tfidf.shape}')

        # Save the vectorizer for later use
        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.debug('TF-IDF vectorizer saved successfully.')

        return X_train_tfidf, y_train
    except Exception as e:
        logger.error(f"Error applying TF-IDF: {e}")
        raise


def save_features(X, y, save_dir: str) -> None:
    """
    Save TF-IDF features and labels to pickle files.

    Args:
        X: TF-IDF feature matrix.
        y: Labels array.
        save_dir (str): Directory path to save files.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, 'X_train_tfidf.pkl'), 'wb') as f:
            pickle.dump(X, f)
        with open(os.path.join(save_dir, 'y_train.pkl'), 'wb') as f:
            pickle.dump(y, f)

        logger.debug(f'Features and labels saved to {save_dir}')
    except Exception as e:
        logger.error(f"Error saving features: {e}")
        raise


def main():
    """
    Main feature engineering pipeline:
        - Load parameters
        - Load processed training data
        - Apply TF-IDF
        - Save features and labels
    """
    try:
        root_dir = get_root_directory()

        # Load parameters
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        max_features = params['feature_engineering']['max_features']
        ngram_range = tuple(params['feature_engineering']['ngram_range'])

        # Load processed training data
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        # Apply TF-IDF
        X_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

        # Save features and labels
        save_dir = os.path.join(root_dir, 'data/processed')
        save_features(X_train_tfidf, y_train, save_dir)

        logger.debug('Feature engineering completed successfully.')

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
