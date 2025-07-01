"""
Data Processor Module
Responsibility: Process data for prediction using the same preprocessing as training
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import os
import sys

# Add parent directory to path to import from ml_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml_pipeline'))

from feature_engineer import FeatureEngineer


class DataProcessor:
    """Process data for poverty prediction using trained preprocessing components"""
    
    def __init__(self, preprocessing_path: str = None):
        """
        Initialize DataProcessor
        
        Args:
            preprocessing_path: Path to saved preprocessing components
        """
        self.engineer = FeatureEngineer()
        self.preprocessing_path = preprocessing_path
        self.is_fitted = False
        
        # Try to load preprocessing components from centralized location
        if not preprocessing_path:
            # Look for preprocessing components in the models directory
            centralized_preprocessing = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'preprocessing_components.pkl')
            if os.path.exists(centralized_preprocessing):
                preprocessing_path = centralized_preprocessing
                print(f"Found preprocessing components: {preprocessing_path}")
        
        # Load preprocessing components if available
        if preprocessing_path and os.path.exists(preprocessing_path):
            self.load_preprocessing_components(preprocessing_path)
        else:
            print("⚠️ No preprocessing components found. Will use default preprocessing.")
    
    def load_preprocessing_components(self, filepath: str) -> bool:
        """
        Load preprocessing components from file
        
        Args:
            filepath: Path to saved preprocessing components
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.engineer.load_preprocessing_components(filepath)
            self.is_fitted = True
            return True
        except Exception as e:
            print(f"Error loading preprocessing components: {e}")
            return False
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process data for prediction using the same preprocessing as training
        
        Args:
            df: Input dataframe with raw data
            
        Returns:
            Processed dataframe ready for prediction
        """
        try:
            # Make a copy to avoid modifying original
            df_processed = df.copy()
            
            # Handle missing values
            print("Handling missing values...")
            df_clean = self.engineer.handle_missing_values(df_processed, strategy='median')
            
            # Prepare features (exclude target-related columns)
            print("Preparing features...")
            exclude_cols = ['id_persona', 'tiempo_id', 'ingreso_per_capita', 'ingreso_laboral']
            X = df_clean.drop(columns=[col for col in exclude_cols if col in df_clean.columns])
            
            # Extract statistical features
            print("Extracting statistical features...")
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            X_engineered = self.engineer.extract_statistical_features(X, numerical_cols)
            
            # Encode categorical features
            print("Encoding categorical features...")
            categorical_cols = X_engineered.select_dtypes(include=['object']).columns.tolist()
            X_encoded = self.engineer.encode_categorical_features(X_engineered, categorical_cols, method='label')
            
            # Handle any remaining categorical features (binned features)
            remaining_categorical = X_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
            if remaining_categorical:
                X_encoded = self.engineer.encode_categorical_features(X_encoded, remaining_categorical, method='label')
            
            # Scale features
            print("Scaling features...")
            X_scaled = self.engineer.scale_features(X_encoded, fit=False)  # Use transform only
            
            # Final cleanup: Check for any remaining NaN values
            print("Final cleanup: Checking for any remaining NaN values...")
            if X_scaled.isnull().any().any():
                print(f"Warning: Found NaN values in {X_scaled.isnull().sum().sum()} cells. Cleaning...")
                # Fill remaining NaN values with median for each column
                for col in X_scaled.columns:
                    if X_scaled[col].isnull().any():
                        X_scaled[col] = X_scaled[col].fillna(X_scaled[col].median())
                print("NaN values cleaned.")
            else:
                print("No NaN values found in final feature matrix.")
            
            print(f"Final feature matrix shape: {X_scaled.shape}")
            
            return X_scaled
            
        except Exception as e:
            print(f"Error processing data: {e}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after processing"""
        if hasattr(self.engineer, 'feature_names') and self.engineer.feature_names:
            return self.engineer.feature_names
        return []
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create sample data for testing"""
        sample_data = {
            'persona_key': [1, 2, 3, 4, 5],
            'tiempo_id': [202505, 202505, 202505, 202505, 202505],
            'anio': [2025, 2025, 2025, 2025, 2025],
            'mes': [5, 5, 5, 5, 5],
            'sector_id': [2, 1, 0, 2, 1],
            'condact_id': [4, 1, 9, 2, 4],
            'sexo': [2, 1, 2, 1, 2],
            'ciudad_id': [10150, 10150, 10150, 10150, 10150],
            'nivel_instruccion': [3, 5, 3, 4, 5],
            'estado_civil': [1, 6, 6, 1, 1],
            'edad': [67, 64, 78, 49, 29],
            'ingreso_laboral': [200.0, 0.0, 0.0, 5.0, 470.0],
            'ingreso_per_capita': [100.0, 100.0, 110.0, 196.0, 196.0],
            'horas_trabajo_semana': [20, 0, 0, 10, 32],
            'desea_trabajar_mas': [4, 0, 0, 4, 4],
            'disponible_trabajar_mas': [0, 0, 0, 0, 0]
        }
        
        return pd.DataFrame(sample_data)
    
    def validate_processed_data(self, X: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate processed data before prediction
        
        Args:
            X: Processed feature matrix
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if data is empty
        if X.empty:
            return False, ["Los datos procesados están vacíos"]
        
        # Check for NaN values
        if X.isnull().any().any():
            nan_count = X.isnull().sum().sum()
            errors.append(f"Los datos procesados contienen {nan_count} valores NaN")
        
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
            errors.append(f"Los datos procesados contienen {inf_count} valores infinitos")
        
        # Check minimum number of features
        if X.shape[1] < 10:
            errors.append(f"Pocas características procesadas: {X.shape[1]} (esperado al menos 10)")
        
        return len(errors) == 0, errors 