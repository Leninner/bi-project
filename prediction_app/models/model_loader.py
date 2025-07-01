"""
Model Loader Module
Responsibility: Load and manage trained models for prediction
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys

# Add parent directory to path to import from ml_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml_pipeline'))


class ModelLoader:
    """Load and manage trained models for poverty prediction"""
    
    def __init__(self, models_dir: str = "../models"):
        """
        Initialize ModelLoader
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = models_dir
        self.models = {}
        self.model_info = {}
        
        # Available model types (standard names)
        self.model_types = {
            'linear': 'logistic_regression',
            'neural': 'poverty_classifier'
        }
        
        # Load available models
        self._load_available_models()
    
    def _load_available_models(self):
        """Load all available models from the models directory"""
        if not os.path.exists(self.models_dir):
            print(f"Models directory not found: {self.models_dir}")
            return
        
        # Look for model files
        for filename in os.listdir(self.models_dir):
            filepath = os.path.join(self.models_dir, filename)
            
            # Skip preprocessing components and info files
            if filename in ['preprocessing_components.pkl', 'poverty_classifier_info.pkl']:
                continue
                
            if filename.endswith('.h5') or filename.endswith('.hdf5'):
                # Neural network model
                model_name = filename.replace('.h5', '').replace('.hdf5', '')
                try:
                    model = tf.keras.models.load_model(filepath)
                    self.models[model_name] = model
                    self.model_info[model_name] = {
                        'type': 'neural',
                        'filepath': filepath,
                        'loaded': True
                    }
                    print(f"Loaded neural model: {model_name}")
                except Exception as e:
                    print(f"Error loading neural model {model_name}: {e}")
            
            elif filename.endswith('.pkl'):
                # Scikit-learn model
                model_name = filename.replace('.pkl', '')
                try:
                    with open(filepath, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Handle both direct model objects and model data dictionaries
                    if isinstance(model_data, dict) and 'model' in model_data:
                        # Model saved with metadata (feature_importance, best_params, etc.)
                        model = model_data['model']
                        model_type = self._get_model_type(model)
                        self.model_info[model_name] = {
                            'type': model_type,
                            'filepath': filepath,
                            'loaded': True,
                            'has_metadata': True,
                            'feature_importance': model_data.get('feature_importance'),
                            'best_params': model_data.get('best_params')
                        }
                    else:
                        # Direct model object
                        model = model_data
                        model_type = self._get_model_type(model)
                        self.model_info[model_name] = {
                            'type': model_type,
                            'filepath': filepath,
                            'loaded': True,
                            'has_metadata': False
                        }
                    
                    self.models[model_name] = model
                    print(f"Loaded {model_type} model: {model_name}")
                    
                except Exception as e:
                    print(f"Error loading model {model_name}: {e}")
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Get information about available models"""
        return self.model_info.copy()
    
    def _get_model_type(self, model) -> str:
        """
        Determine the type of a scikit-learn model
        
        Args:
            model: The model object
            
        Returns:
            Model type string ('linear' or 'logistic')
        """
        model_class = type(model).__name__
        
        if 'LogisticRegression' in model_class:
            return 'logistic'
        elif 'LinearRegression' in model_class:
            return 'linear'
        else:
            # Default to linear for other scikit-learn models
            return 'linear'
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load a specific model by name
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model or None if not found
        """
        if model_name in self.models:
            return self.models[model_name]
        
        # Try to load from file if not in memory
        if model_name in self.model_info:
            filepath = self.model_info[model_name]['filepath']
            model_type = self.model_info[model_name]['type']
            
            try:
                if model_type == 'neural':
                    model = tf.keras.models.load_model(filepath)
                else:
                    with open(filepath, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    # Handle both direct model objects and model data dictionaries
                    if isinstance(model_data, dict) and 'model' in model_data:
                        model = model_data['model']
                    else:
                        model = model_data
                
                self.models[model_name] = model
                return model
                
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                return None
        
        return None
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a specific model
        
        Args:
            model_name: Name of the model to use
            X: Feature matrix (numpy array or pandas DataFrame)
            
        Returns:
            Predictions array
        """
        model = self.load_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found or could not be loaded")
        
        try:
            if self.model_info[model_name]['type'] == 'neural':
                # Neural network prediction
                predictions = model.predict(X)
                # For binary classification, return class predictions
                if predictions.shape[1] == 2:
                    return np.argmax(predictions, axis=1)
                else:
                    return (predictions > 0.5).astype(int).flatten()
            else:
                # Scikit-learn model prediction - ensure proper feature names
                if hasattr(model, 'feature_names_in_') and isinstance(X, np.ndarray):
                    # Convert numpy array to DataFrame with correct feature names
                    X_df = pd.DataFrame(X, columns=model.feature_names_in_)
                    return model.predict(X_df)
                elif hasattr(model, 'n_features_in_') and isinstance(X, np.ndarray):
                    # For older scikit-learn versions, check if we have feature names in metadata
                    if self.model_info[model_name].get('has_metadata') and 'feature_importance' in self.model_info[model_name]:
                        feature_names = self.model_info[model_name]['feature_importance']['feature'].tolist()
                        if len(feature_names) == X.shape[1]:
                            X_df = pd.DataFrame(X, columns=feature_names)
                            return model.predict(X_df)
                # Use as-is (already DataFrame or no feature names to preserve)
                return model.predict(X)
                
        except Exception as e:
            raise Exception(f"Error making predictions with model {model_name}: {e}")
    
    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities using a specific model
        
        Args:
            model_name: Name of the model to use
            X: Feature matrix (numpy array or pandas DataFrame)
            
        Returns:
            Probability predictions array
        """
        model = self.load_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found or could not be loaded")
        
        try:
            if self.model_info[model_name]['type'] == 'neural':
                # Neural network probability prediction
                probabilities = model.predict(X)
                # For binary classification, return probability of positive class
                if probabilities.shape[1] == 2:
                    return probabilities[:, 1]  # Probability of class 1
                else:
                    return probabilities.flatten()
            else:
                # Scikit-learn model probability prediction - ensure proper feature names
                if hasattr(model, 'feature_names_in_') and isinstance(X, np.ndarray):
                    # Convert numpy array to DataFrame with correct feature names
                    X_df = pd.DataFrame(X, columns=model.feature_names_in_)
                    if hasattr(model, 'predict_proba'):
                        return model.predict_proba(X_df)[:, 1]  # Probability of class 1
                    else:
                        # If model doesn't support probabilities, return predictions
                        return model.predict(X_df)
                elif hasattr(model, 'n_features_in_') and isinstance(X, np.ndarray):
                    # For older scikit-learn versions, check if we have feature names in metadata
                    if self.model_info[model_name].get('has_metadata') and 'feature_importance' in self.model_info[model_name]:
                        feature_names = self.model_info[model_name]['feature_importance']['feature'].tolist()
                        if len(feature_names) == X.shape[1]:
                            X_df = pd.DataFrame(X, columns=feature_names)
                            if hasattr(model, 'predict_proba'):
                                return model.predict_proba(X_df)[:, 1]  # Probability of class 1
                            else:
                                # If model doesn't support probabilities, return predictions
                                return model.predict(X_df)
                # Use as-is (already DataFrame or no feature names to preserve)
                if hasattr(model, 'predict_proba'):
                    return model.predict_proba(X)[:, 1]  # Probability of class 1
                else:
                    # If model doesn't support probabilities, return predictions
                    return model.predict(X)
                
        except Exception as e:
            raise Exception(f"Error getting probabilities with model {model_name}: {e}")
    
    def get_model_summary(self, model_name: str) -> Dict:
        """
        Get summary information about a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        if model_name not in self.model_info:
            return {}
        
        info = self.model_info[model_name].copy()
        model = self.load_model(model_name)
        
        if model is not None:
            if info['type'] == 'neural':
                info['layers'] = len(model.layers)
                info['parameters'] = model.count_params()
                info['input_shape'] = model.input_shape
                info['output_shape'] = model.output_shape
            else:
                info['model_type'] = type(model).__name__
                if hasattr(model, 'n_features_in_'):
                    info['n_features'] = model.n_features_in_
        
        return info
    
 