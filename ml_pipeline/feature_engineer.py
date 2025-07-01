"""
Feature Engineer Module
Responsibility: Extract, transform, and engineer features for machine learning
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.decomposition import PCA
import pickle
import os


class FeatureEngineer:
    """Handles feature engineering and preprocessing for poverty prediction"""
    
    def __init__(self):
        """Initialize FeatureEngineer with preprocessing components"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_selector = None
        self.pca = None
        self.feature_names = []
        self.is_fitted = False
        
    def extract_statistical_features(self, df: pd.DataFrame, numerical_cols: List[str]) -> pd.DataFrame:
        """
        Extract statistical features from numerical columns
        
        Args:
            df: Input dataframe
            numerical_cols: List of numerical column names
            
        Returns:
            DataFrame with additional statistical features
        """
        df_engineered = df.copy()
        
        # Dictionary to collect all new features before adding them at once
        new_features = {}
        
        # Create interaction features
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                # Multiplication interaction
                new_features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                # Ratio interaction (avoid division by zero)
                new_features[f'{col1}_div_{col2}'] = np.where(
                    df[col2] != 0, df[col1] / df[col2], 0
                )
        
        # Create polynomial features for key numerical columns
        key_features = ['edad', 'ingreso_laboral', 'ingreso_per_capita', 'horas_trabajo_semana']
        for col in key_features:
            if col in numerical_cols:
                new_features[f'{col}_squared'] = df[col] ** 2
                new_features[f'{col}_cubed'] = df[col] ** 3
        
        # Create binned features
        if 'edad' in numerical_cols:
            # Handle NaN values in age before binning
            age_data = df['edad'].dropna()
            if len(age_data) > 0:
                edad_bin = pd.cut(df['edad'], bins=5, labels=['very_young', 'young', 'middle', 'senior', 'elderly'], include_lowest=True)
                # Fill any remaining NaN values with the most common bin
                if edad_bin.isnull().any():
                    most_common_bin = edad_bin.mode()[0] if not edad_bin.mode().empty else 'middle'
                    edad_bin = edad_bin.fillna(most_common_bin)
                new_features['edad_bin'] = edad_bin
        
        if 'ingreso_per_capita' in numerical_cols:
            # Handle NaN values in income before binning
            income_data = df['ingreso_per_capita'].dropna()
            if len(income_data) > 0:
                ingreso_bin = pd.cut(df['ingreso_per_capita'], bins=4, labels=['low', 'medium', 'high', 'very_high'], include_lowest=True)
                # Fill any remaining NaN values with the most common bin
                if ingreso_bin.isnull().any():
                    most_common_bin = ingreso_bin.mode()[0] if not ingreso_bin.mode().empty else 'medium'
                    ingreso_bin = ingreso_bin.fillna(most_common_bin)
                new_features['ingreso_bin'] = ingreso_bin
        
        # Add all new features at once using pd.concat to avoid fragmentation
        if new_features:
            new_features_df = pd.DataFrame(new_features, index=df.index)
            df_engineered = pd.concat([df_engineered, new_features_df], axis=1)
        
        return df_engineered
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str], method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical features using specified method
        
        Args:
            df: Input dataframe
            categorical_cols: List of categorical column names
            method: 'label' for LabelEncoder, 'onehot' for OneHotEncoder
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        if method == 'label':
            for col in categorical_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    # Handle NaN values in categorical columns before encoding
                    col_data = df[col].copy()
                    
                    # Handle different dtypes appropriately
                    if pd.api.types.is_categorical_dtype(col_data):
                        # For Categorical columns, add 'missing' to categories if needed
                        if col_data.isnull().any():
                            if 'missing' not in col_data.cat.categories:
                                col_data = col_data.cat.add_categories('missing')
                            col_data = col_data.fillna('missing')
                    else:
                        # For object columns, simple fillna works
                        col_data = col_data.fillna('missing')
                    
                    # Convert to string for LabelEncoder
                    col_data = col_data.astype(str)
                    df_encoded[col] = le.fit_transform(col_data)
                    self.label_encoders[col] = le
        
        elif method == 'onehot':
            # Prepare categorical data for one-hot encoding
            cat_data = df[categorical_cols].copy()
            
            # Handle NaN values for each column appropriately
            for col in cat_data.columns:
                if pd.api.types.is_categorical_dtype(cat_data[col]):
                    # For Categorical columns, add 'missing' to categories if needed
                    if cat_data[col].isnull().any():
                        if 'missing' not in cat_data[col].cat.categories:
                            cat_data[col] = cat_data[col].cat.add_categories('missing')
                        cat_data[col] = cat_data[col].fillna('missing')
                else:
                    # For object columns, simple fillna works
                    cat_data[col] = cat_data[col].fillna('missing')
            
            onehot_features = self.onehot_encoder.fit_transform(cat_data)
            
            # Create feature names
            feature_names = []
            for i, col in enumerate(categorical_cols):
                unique_values = cat_data[col].unique()
                for val in unique_values:
                    feature_names.append(f"{col}_{val}")
            
            # Create DataFrame with one-hot features
            onehot_df = pd.DataFrame(onehot_features, columns=feature_names, index=df.index)
            
            # Drop original categorical columns and add one-hot features
            df_encoded = df_encoded.drop(columns=categorical_cols)
            df_encoded = pd.concat([df_encoded, onehot_df], axis=1)
        
        return df_encoded
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values using specified strategy
        
        Args:
            df: Input dataframe
            strategy: 'median', 'mean', 'mode', or 'drop'
            
        Returns:
            DataFrame with handled missing values
        """
        df_clean = df.copy()
        
        if strategy == 'drop':
            df_clean = df_clean.dropna()
        else:
            numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
            categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
            
            # Handle numerical columns
            for col in numerical_cols:
                if df_clean[col].isnull().any():
                    if strategy == 'median':
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    elif strategy == 'mean':
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            
            # Handle categorical columns
            for col in categorical_cols:
                if df_clean[col].isnull().any():
                    if strategy == 'mode':
                        mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'missing'
                        # Handle Categorical columns properly
                        if pd.api.types.is_categorical_dtype(df_clean[col]):
                            if mode_value not in df_clean[col].cat.categories:
                                df_clean[col] = df_clean[col].cat.add_categories(mode_value)
                        df_clean[col] = df_clean[col].fillna(mode_value)
                    else:
                        # Handle Categorical columns properly
                        if pd.api.types.is_categorical_dtype(df_clean[col]):
                            if 'missing' not in df_clean[col].cat.categories:
                                df_clean[col] = df_clean[col].cat.add_categories('missing')
                        df_clean[col] = df_clean[col].fillna('missing')
        
        return df_clean
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'correlation', k: int = 10) -> pd.DataFrame:
        """
        Select the most important features
        
        Args:
            X: Feature matrix
            y: Target variable
            method: 'correlation', 'mutual_info', 'f_regression', 'f_classif'
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        if method == 'correlation':
            # Select features based on correlation with target with proper error handling
            try:
                # Use pandas corrwith with min_periods to handle missing data
                correlations = abs(X.corrwith(y, min_periods=1))
                # Handle NaN values
                correlations = correlations.fillna(0)
                selected_features = correlations.nlargest(k).index.tolist()
                X_selected = X[selected_features]
            except Exception as e:
                print(f"Warning: Could not calculate correlations for feature selection: {e}")
                # Fallback: select first k features
                selected_features = X.columns[:min(k, len(X.columns))].tolist()
                X_selected = X[selected_features]
            
        elif method in ['f_regression', 'f_classif']:
            # Use sklearn feature selection
            if method == 'f_regression':
                self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            else:
                self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        self.feature_names = X_selected.columns.tolist()
        return X_selected
    
    def apply_pca(self, X: pd.DataFrame, n_components: int = 0.95) -> pd.DataFrame:
        """
        Apply Principal Component Analysis for dimensionality reduction
        
        Args:
            X: Feature matrix
            n_components: Number of components or explained variance ratio
            
        Returns:
            DataFrame with PCA components
        """
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        # Create column names for PCA components
        pca_columns = [f'PC_{i+1}' for i in range(X_pca.shape[1])]
        X_pca_df = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        
        print(f"PCA: {X.shape[1]} features reduced to {X_pca.shape[1]} components")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        return X_pca_df
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using StandardScaler
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler or use existing fit
            
        Returns:
            DataFrame with scaled features
        """
        # Check for NaN values before scaling
        if X.isnull().any().any():
            print(f"Warning: Found NaN values in {X.isnull().sum().sum()} cells before scaling. Cleaning...")
            # Fill NaN values with median for each column
            for col in X.columns:
                if X[col].isnull().any():
                    X[col] = X[col].fillna(X[col].median())
            print("NaN values cleaned before scaling.")
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        return X_scaled_df
    
    def create_poverty_target(self, df: pd.DataFrame, method: str = 'income_threshold') -> pd.Series:
        """
        Create binary poverty target variable
        
        Args:
            df: Input dataframe
            method: Method to create poverty indicator ('income_threshold' or 'multi_factor')
            
        Returns:
            Series with binary poverty indicator
        """
        if df is None or df.empty:
            print("Warning: Empty dataframe provided for poverty target creation")
            return pd.Series(dtype=int)
        
        if method == 'income_threshold':
            # Use income per capita to create poverty indicator
            if 'ingreso_per_capita' in df.columns:
                # Handle missing values
                income_data = df['ingreso_per_capita'].dropna()
                if len(income_data) > 0:
                    # Define poverty threshold (below median)
                    threshold = income_data.median()
                    poverty_target = (df['ingreso_per_capita'] < threshold).astype(int)
                    # Fill NaN values with 0 (not in poverty)
                    poverty_target = poverty_target.fillna(0)
                else:
                    print("Warning: No valid income per capita data found")
                    poverty_target = pd.Series(0, index=df.index)
            elif 'ingreso_laboral' in df.columns:
                # Fallback to labor income
                income_data = df['ingreso_laboral'].dropna()
                if len(income_data) > 0:
                    threshold = income_data.median()
                    poverty_target = (df['ingreso_laboral'] < threshold).astype(int)
                    poverty_target = poverty_target.fillna(0)
                else:
                    print("Warning: No valid labor income data found")
                    poverty_target = pd.Series(0, index=df.index)
            else:
                print("Warning: No income columns found for poverty target creation")
                poverty_target = pd.Series(0, index=df.index)
        
        elif method == 'multi_factor':
            # Create poverty indicator based on multiple factors
            factors = []
            
            # Income factor
            if 'ingreso_per_capita' in df.columns:
                income_data = df['ingreso_per_capita'].dropna()
                if len(income_data) > 0:
                    income_factor = (df['ingreso_per_capita'] < income_data.median()).astype(int)
                    income_factor = income_factor.fillna(0)
                    factors.append(income_factor)
            
            # Age factor (elderly)
            if 'edad' in df.columns:
                age_data = df['edad'].dropna()
                if len(age_data) > 0:
                    age_factor = (df['edad'] > 65).astype(int)  # Elderly factor
                    age_factor = age_factor.fillna(0)
                    factors.append(age_factor)
            
            # Work hours factor (low work hours)
            if 'horas_trabajo_semana' in df.columns:
                work_data = df['horas_trabajo_semana'].dropna()
                if len(work_data) > 0:
                    work_factor = (df['horas_trabajo_semana'] < 20).astype(int)  # Low work hours
                    work_factor = work_factor.fillna(0)
                    factors.append(work_factor)
            
            # Employment condition factor (underemployment)
            if 'condact_id' in df.columns:
                underemployment_conditions = [2, 3]  # Subempleado, Ocupado parcial
                condition_factor = df['condact_id'].isin(underemployment_conditions).astype(int)
                factors.append(condition_factor)
            
            if factors:
                # Sum all factors and create binary target (at least 2 factors indicate poverty)
                poverty_target = pd.concat(factors, axis=1).sum(axis=1)
                poverty_target = (poverty_target >= 2).astype(int)
            else:
                print("Warning: No valid factors found for multi-factor poverty target")
                poverty_target = pd.Series(0, index=df.index)
        
        else:
            print(f"Warning: Unknown method '{method}' for poverty target creation. Using income threshold.")
            return self.create_poverty_target(df, method='income_threshold')
        
        # Validate the target
        if poverty_target.isnull().any():
            print("Warning: Found NaN values in poverty target, filling with 0")
            poverty_target = poverty_target.fillna(0)
        
        print(f"Created poverty target using '{method}' method:")
        print(f"  Total samples: {len(poverty_target)}")
        print(f"  In poverty: {poverty_target.sum()} ({poverty_target.sum()/len(poverty_target)*100:.2f}%)")
        print(f"  Not in poverty: {(poverty_target == 0).sum()} ({(poverty_target == 0).sum()/len(poverty_target)*100:.2f}%)")
        
        return poverty_target
    
    def save_preprocessing_components(self, filepath: str):
        """Save preprocessing components for later use"""
        components = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'onehot_encoder': self.onehot_encoder,
            'feature_selector': self.feature_selector,
            'pca': self.pca,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(components, f)
        
        print(f"Preprocessing components saved to {filepath}")
    
    def load_preprocessing_components(self, filepath: str):
        """Load preprocessing components from file"""
        with open(filepath, 'rb') as f:
            components = pickle.load(f)
        
        self.scaler = components['scaler']
        self.label_encoders = components['label_encoders']
        self.onehot_encoder = components['onehot_encoder']
        self.feature_selector = components['feature_selector']
        self.pca = components['pca']
        self.feature_names = components['feature_names']
        self.is_fitted = components['is_fitted']
        
        print(f"Preprocessing components loaded from {filepath}")


if __name__ == "__main__":
    # Test feature engineering
    from data_analyzer import DataAnalyzer
    
    # Load and analyze data
    analyzer = DataAnalyzer("../data/poverty_dataset.csv")
    analyzer.load_data()
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Handle missing values
    df_clean = engineer.handle_missing_values(analyzer.df)
    
    # Extract statistical features
    numerical_cols = analyzer.df.select_dtypes(include=[np.number]).columns.tolist()
    df_engineered = engineer.extract_statistical_features(df_clean, numerical_cols)
    
    print(f"Original shape: {analyzer.df.shape}")
    print(f"Engineered shape: {df_engineered.shape}") 