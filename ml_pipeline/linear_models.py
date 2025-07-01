"""
Linear Models Module
Responsibility: Train and evaluate linear regression and logistic regression models
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress sklearn convergence warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.linear_model._sag')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.linear_model._logistic')
warnings.filterwarnings('ignore', message='The max_iter was reached which means the coef_ did not converge')


class LinearModels:
    """Handles linear and logistic regression models for poverty prediction"""
    
    def __init__(self):
        """Initialize LinearModels with model containers"""
        self.models = {}
        self.best_params = {}
        self.feature_importance = {}
        self.training_history = {}
        
    def train_linear_regression(self, X_train: pd.DataFrame, y_train: pd.Series, 
                               model_name: str = 'linear_regression', 
                               use_grid_search: bool = False) -> Dict:
        """
        Train linear regression model
        
        Args:
            X_train: Training features
            y_train: Training target
            model_name: Name for the model
            use_grid_search: Whether to use GridSearchCV for hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        print(f"\n=== TRAINING {model_name.upper()} ===")
        
        if use_grid_search:
            # Grid search for hyperparameter tuning
            param_grid = {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
            
            grid_search = GridSearchCV(
                LinearRegression(), 
                param_grid, 
                cv=5, 
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
        
        # Store model
        self.models[model_name] = model
        
        # Calculate feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        self.feature_importance[model_name] = feature_importance
        
        # Training metrics
        y_pred_train = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        
        results = {
            'model': model,
            'train_mse': train_mse,
            'train_r2': train_r2,
            'feature_importance': feature_importance,
            'intercept': model.intercept_
        }
        
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Intercept: {model.intercept_:.4f}")
        
        return results
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 model_name: str = 'logistic_regression',
                                 use_grid_search: bool = True) -> Dict:
        """
        Train logistic regression model for classification
        
        Args:
            X_train: Training features
            y_train: Training target (binary)
            model_name: Name for the model
            use_grid_search: Whether to use GridSearchCV for hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        print(f"\n=== TRAINING {model_name.upper()} ===")
        
        if use_grid_search:
            # Grid search for hyperparameter tuning
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            
            grid_search = GridSearchCV(
                LogisticRegression(random_state=42), 
                param_grid, 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)
        
        # Store model
        self.models[model_name] = model
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        self.feature_importance[model_name] = feature_importance
        
        # Training metrics
        y_pred_train = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        
        results = {
            'model': model,
            'train_accuracy': train_accuracy,
            'feature_importance': feature_importance,
            'intercept': model.intercept_[0]
        }
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Intercept: {model.intercept_[0]:.4f}")
        
        return results
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate a trained model on test data
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        # Determine if it's a classification or regression model
        if hasattr(model, 'predict_proba'):
            # Classification model
            test_accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            results = {
                'test_accuracy': test_accuracy,
                'classification_report': classification_rep,
                'confusion_matrix': conf_matrix,
                'predictions': y_pred
            }
            
            print(f"\n=== EVALUATION RESULTS FOR {model_name.upper()} ===")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
        else:
            # Regression model
            test_mse = mean_squared_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            
            results = {
                'test_mse': test_mse,
                'test_r2': test_r2,
                'predictions': y_pred
            }
            
            print(f"\n=== EVALUATION RESULTS FOR {model_name.upper()} ===")
            print(f"Test MSE: {test_mse:.4f}")
            print(f"Test R²: {test_r2:.4f}")
        
        return results
    
    def cross_validate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series, 
                           cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation on a model
        
        Args:
            model_name: Name of the model to validate
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Determine scoring metric based on model type
        if hasattr(model, 'predict_proba'):
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
        
        results = {
            'cv_scores': cv_scores,
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_folds': cv_folds
        }
        
        print(f"\n=== CROSS-VALIDATION RESULTS FOR {model_name.upper()} ===")
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def plot_feature_importance(self, model_name: str, top_n: int = 10, save_path: str = None):
        """
        Plot feature importance for a model
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to plot
            save_path: Path to save the plot
        """
        if model_name not in self.feature_importance:
            raise ValueError(f"Feature importance for {model_name} not found")
        
        importance_df = self.feature_importance[model_name].head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_df)), importance_df['coefficient'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Coefficient Value')
        plt.title(f'Feature Importance - {model_name.replace("_", " ").title()}')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_linear_regression(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series, save_path: str = None):
        """
        Plot linear regression results: actual vs predicted values
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test target
            save_path: Path to save the plot
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        # Actual vs Predicted scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        
        # Residuals plot
        plt.subplot(2, 2, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        # Residuals histogram
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        
        # Q-Q plot for residuals
        plt.subplot(2, 2, 4)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Linear regression plots saved to {save_path}")
        
        plt.show()
    
    def plot_logistic_regression(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series, save_path: str = None):
        """
        Plot logistic regression results: ROC curve, confusion matrix, and precision-recall curve
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test target
            save_path: Path to save the plot
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Create the plot
        plt.figure(figsize=(15, 5))
        
        # ROC Curve
        plt.subplot(1, 3, 1)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        # Confusion Matrix
        plt.subplot(1, 3, 2)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Poverty', 'Poverty'],
                   yticklabels=['No Poverty', 'Poverty'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Precision-Recall Curve
        plt.subplot(1, 3, 3)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Logistic regression plots saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model_name: str, filepath: str):
        """Save a trained model to file"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_data = {
            'model': self.models[model_name],
            'feature_importance': self.feature_importance.get(model_name),
            'best_params': self.best_params.get(model_name)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """Load a trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models[model_name] = model_data['model']
        if 'feature_importance' in model_data:
            self.feature_importance[model_name] = model_data['feature_importance']
        if 'best_params' in model_data:
            self.best_params[model_name] = model_data['best_params']
        
        print(f"Model {model_name} loaded from {filepath}")

    def plot_combined_model_comparison(self, results_dict, save_path=None):
        """
        Combined plot showing both classification and regression models with their respective evaluation metrics.
        
        Args:
            results_dict: Dictionary containing model results with 'type', 'model_category', and 'score' keys
            save_path: Path to save the plot
        """
        nombre_amigable = {
            'logistic_regression': 'Regresión Logística',
            'linear_regression': 'Regresión Lineal',
            # Add other linear models here if needed
        }
        
        # Separate classification and regression models
        classification_models = []
        classification_scores = []
        regression_models = []
        regression_scores = []
        
        for nombre, info in results_dict.items():
            if info.get('model_category') == 'linear':
                friendly_name = nombre_amigable.get(nombre, nombre)
                score = info.get('score', 0)
                
                if info.get('type') == 'classification':
                    classification_models.append(friendly_name)
                    classification_scores.append(score)
                elif info.get('type') == 'regression':
                    regression_models.append(friendly_name)
                    regression_scores.append(score)
        
        # Create combined plot
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Classification models subplot
        if classification_models:
            bars1 = ax1.bar(classification_models, classification_scores, color='royalblue', alpha=0.8)
            ax1.set_xlabel('Modelos de Clasificación', fontweight='bold')
            ax1.set_ylabel('Precisión (Accuracy)', fontweight='bold')
            ax1.set_title('Modelos Lineales de Clasificación', fontweight='bold')
            ax1.set_ylim(0, 1)
            ax1.grid(axis='y', alpha=0.2)
            ax1.tick_params(axis='x', rotation=30)
            ax1.set_xticklabels(ax1.get_xticklabels(), ha='right')
            
            # Add value labels on bars
            for bar, score in zip(bars1, classification_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No hay modelos de clasificación', ha='center', va='center',
                    fontsize=14, style='italic', transform=ax1.transAxes)
            ax1.set_title('Modelos Lineales de Clasificación', fontweight='bold')
            ax1.axis('off')
        
        # Regression models subplot
        if regression_models:
            bars2 = ax2.bar(regression_models, regression_scores, color='seagreen', alpha=0.8)
            ax2.set_xlabel('Modelos de Regresión', fontweight='bold')
            ax2.set_ylabel('R²', fontweight='bold')
            ax2.set_title('Modelos Lineales de Regresión', fontweight='bold')
            ax2.set_ylim(0, 1)
            ax2.grid(axis='y', alpha=0.2)
            ax2.tick_params(axis='x', rotation=30)
            ax2.set_xticklabels(ax2.get_xticklabels(), ha='right')
            
            # Add value labels on bars
            for bar, score in zip(bars2, regression_scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No hay modelos de regresión', ha='center', va='center',
                    fontsize=14, style='italic', transform=ax2.transAxes)
            ax2.set_title('Modelos Lineales de Regresión', fontweight='bold')
            ax2.axis('off')
        
        plt.suptitle('Comparativa de Modelos Lineales - Métodos de Evaluación', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Combined model comparison plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Test linear models
    from data_analyzer import DataAnalyzer
    from feature_engineer import FeatureEngineer
    
    # Load and prepare data
    analyzer = DataAnalyzer("../data/poverty_dataset.csv")
    analyzer.load_data()
    
    engineer = FeatureEngineer()
    df_clean = engineer.handle_missing_values(analyzer.df)
    
    # Create target variable
    y = engineer.create_poverty_target(df_clean)
    
    # Prepare features (exclude target-related columns)
    exclude_cols = ['id_persona', 'tiempo_id', 'ingreso_per_capita', 'ingreso_laboral']
    X = df_clean.drop(columns=[col for col in exclude_cols if col in df_clean.columns])
    
    # Encode categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X_encoded = engineer.encode_categorical_features(X, categorical_cols, method='label')
    
    # Scale features
    X_scaled = engineer.scale_features(X_encoded)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train models
    linear_models = LinearModels()
    
    # Train logistic regression
    results = linear_models.train_logistic_regression(X_train, y_train)
    
    # Evaluate model
    eval_results = linear_models.evaluate_model('logistic_regression', X_test, y_test)
    
    # Plot feature importance
    linear_models.plot_feature_importance('logistic_regression', top_n=10) 