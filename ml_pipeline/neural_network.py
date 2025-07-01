"""
Neural Network Module
Responsibility: Train and evaluate neural network models using Keras
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os


class NeuralNetwork:
    """Handles neural network models for poverty prediction using Keras"""
    
    def __init__(self):
        """Initialize NeuralNetwork with model containers"""
        self.models = {}
        self.training_history = {}
        self.model_configs = {}
        
    def create_regression_model(self, input_dim: int, model_name: str = 'regression_nn',
                               layers_config: List[Dict] = None) -> keras.Model:
        """
        Create a neural network for regression
        
        Args:
            input_dim: Number of input features
            model_name: Name for the model
            layers_config: Configuration for hidden layers
            
        Returns:
            Compiled Keras model
        """
        if layers_config is None:
            layers_config = [
                {'units': 128, 'activation': 'relu', 'dropout': 0.3},
                {'units': 64, 'activation': 'relu', 'dropout': 0.2},
                {'units': 32, 'activation': 'relu', 'dropout': 0.1}
            ]
        
        model = models.Sequential(name=model_name)
        
        # Input layer
        model.add(layers.Dense(
            layers_config[0]['units'], 
            activation=layers_config[0]['activation'],
            input_shape=(input_dim,),
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
        ))
        model.add(layers.Dropout(layers_config[0]['dropout']))
        model.add(layers.BatchNormalization())
        
        # Hidden layers
        for config in layers_config[1:]:
            model.add(layers.Dense(
                config['units'],
                activation=config['activation'],
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
            ))
            model.add(layers.Dropout(config['dropout']))
            model.add(layers.BatchNormalization())
        
        # Output layer for regression
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.models[model_name] = model
        self.model_configs[model_name] = {
            'type': 'regression',
            'input_dim': input_dim,
            'layers_config': layers_config
        }
        
        print(f"Regression model '{model_name}' created with {model.count_params()} parameters")
        return model
    
    def create_classification_model(self, input_dim: int, num_classes: int = 2,
                                   model_name: str = 'classification_nn',
                                   layers_config: List[Dict] = None) -> keras.Model:
        """
        Create a neural network for classification
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            model_name: Name for the model
            layers_config: Configuration for hidden layers
            
        Returns:
            Compiled Keras model
        """
        if layers_config is None:
            layers_config = [
                {'units': 128, 'activation': 'relu', 'dropout': 0.3},
                {'units': 64, 'activation': 'relu', 'dropout': 0.2},
                {'units': 32, 'activation': 'relu', 'dropout': 0.1}
            ]
        
        model = models.Sequential(name=model_name)
        
        # Input layer
        model.add(layers.Dense(
            layers_config[0]['units'], 
            activation=layers_config[0]['activation'],
            input_shape=(input_dim,),
            kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
        ))
        model.add(layers.Dropout(layers_config[0]['dropout']))
        model.add(layers.BatchNormalization())
        
        # Hidden layers
        for config in layers_config[1:]:
            model.add(layers.Dense(
                config['units'],
                activation=config['activation'],
                kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
            ))
            model.add(layers.Dropout(config['dropout']))
            model.add(layers.BatchNormalization())
        
        # Output layer for classification
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy', 'precision', 'recall']
        else:
            model.add(layers.Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        self.models[model_name] = model
        self.model_configs[model_name] = {
            'type': 'classification',
            'input_dim': input_dim,
            'num_classes': num_classes,
            'layers_config': layers_config
        }
        
        print(f"Classification model '{model_name}' created with {model.count_params()} parameters")
        return model
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   epochs: int = 100, batch_size: int = 32,
                   early_stopping: bool = True, patience: int = 10) -> Dict:
        """
        Train a neural network model
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            
        Returns:
            Dictionary with training results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        print(f"\n=== TRAINING {model_name.upper()} ===")
        print(f"Training data shape: {X_train.shape}")
        print(f"Target data shape: {y_train.shape}")
        
        # Prepare callbacks
        callbacks_list = []
        
        if early_stopping:
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks_list.append(early_stop)
        
        # Add model checkpoint
        checkpoint = callbacks.ModelCheckpoint(
            f'models/{model_name}_best.h5',
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            verbose=1
        )
        callbacks_list.append(checkpoint)
        
        # Add learning rate reduction
        lr_reducer = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
        callbacks_list.append(lr_reducer)
        
        # Train the model
        if X_val is not None and y_val is not None:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                verbose=1
            )
        else:
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                verbose=1
            )
        
        # Store training history
        self.training_history[model_name] = history.history
        
        # Training metrics
        train_loss = history.history['loss'][-1]
        train_metrics = {key: value[-1] for key, value in history.history.items() if not key.startswith('val_')}
        
        results = {
            'model': model,
            'history': history.history,
            'train_loss': train_loss,
            'train_metrics': train_metrics,
            'epochs_trained': len(history.history['loss'])
        }
        
        print(f"Training completed in {len(history.history['loss'])} epochs")
        print(f"Final training loss: {train_loss:.4f}")
        
        return results
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate a trained neural network model
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        model_config = self.model_configs[model_name]
        
        print(f"\n=== EVALUATING {model_name.upper()} ===")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # For binary classification, convert probabilities to classes
        if model_config['type'] == 'classification' and model_config['num_classes'] == 2:
            y_pred_classes = (y_pred > 0.5).astype(int)
        else:
            y_pred_classes = y_pred
        
        # Calculate metrics based on model type
        if model_config['type'] == 'regression':
            test_mse = mean_squared_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            test_mae = np.mean(np.abs(y_test - y_pred))
            
            results = {
                'test_mse': test_mse,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'predictions': y_pred
            }
            
            print(f"Test MSE: {test_mse:.4f}")
            print(f"Test R²: {test_r2:.4f}")
            print(f"Test MAE: {test_mae:.4f}")
            
        else:  # Classification
            test_accuracy = accuracy_score(y_test, y_pred_classes)
            classification_rep = classification_report(y_test, y_pred_classes, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred_classes)
            
            results = {
                'test_accuracy': test_accuracy,
                'classification_report': classification_rep,
                'confusion_matrix': conf_matrix,
                'predictions': y_pred,
                'predictions_classes': y_pred_classes
            }
            
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred_classes))
        
        return results
    
    def plot_training_history(self, model_name: str, save_path: str = None):
        """
        Plot training history for a model
        
        Args:
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if model_name not in self.training_history:
            raise ValueError(f"Training history for {model_name} not found")
        
        history = self.training_history[model_name]
        
        # Determine metrics to plot
        metrics = [key for key in history.keys() if not key.startswith('val_')]
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            axes[i].plot(history[metric], label=f'Training {metric}')
            if f'val_{metric}' in history:
                axes[i].plot(history[f'val_{metric}'], label=f'Validation {metric}')
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model_name: str, filepath: str):
        """Save a trained model to file"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        self.models[model_name].save(filepath)
        
        # Save additional information
        info_filepath = filepath.replace('.h5', '_info.pkl')
        model_info = {
            'model_config': self.model_configs[model_name],
            'training_history': self.training_history.get(model_name)
        }
        
        with open(info_filepath, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"Model {model_name} saved to {filepath}")
        print(f"Model info saved to {info_filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """Load a trained model from file"""
        # Load the model
        self.models[model_name] = models.load_model(filepath)
        
        # Load additional information
        info_filepath = filepath.replace('.h5', '_info.pkl')
        if os.path.exists(info_filepath):
            with open(info_filepath, 'rb') as f:
                model_info = pickle.load(f)
            
            self.model_configs[model_name] = model_info['model_config']
            if 'training_history' in model_info:
                self.training_history[model_name] = model_info['training_history']
        
        print(f"Model {model_name} loaded from {filepath}")
    
    def get_model_summary(self, model_name: str) -> str:
        """Get a summary of the model architecture"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Capture model summary
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        
        return '\n'.join(summary_list)

    def plot_nn_classification_comparison(self, results_dict, save_path=None):
        """Gráfica de accuracy de todos los modelos neuronales de clasificación con nombres amigables."""
        nombre_amigable = {
            'poverty_classifier': 'Clasificador de Pobreza',
            # Agrega aquí otros modelos neuronales de clasificación si los tienes
        }
        modelos = []
        accuracies = []
        for nombre, info in results_dict.items():
            if info.get('type') == 'classification' and info.get('model_category') == 'neural':
                modelos.append(nombre_amigable.get(nombre, nombre))
                accuracies.append(info.get('score', 0))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        if modelos:
            barras = plt.bar(modelos, accuracies, color='darkorange', alpha=0.8)
            plt.xlabel('Modelos Neuronales de Clasificación', fontweight='bold')
            plt.ylabel('Precisión (Accuracy)', fontweight='bold')
            plt.title('Comparativa de Precisión - Modelos Neuronales de Clasificación', fontweight='bold')
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.2)
            plt.xticks(rotation=30, ha='right')
            for bar, acc in zip(barras, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No hay modelos neuronales de clasificación', ha='center', va='center',
                     fontsize=14, style='italic', transform=plt.gca().transAxes)
            plt.title('Comparativa de Precisión - Modelos Neuronales de Clasificación', fontweight='bold')
            plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def plot_nn_regression_comparison(self, results_dict, save_path=None):
        """Gráfica de R² de todos los modelos neuronales de regresión con nombres amigables."""
        nombre_amigable = {
            'poverty_regressor': 'Regresor de Pobreza',
            # Agrega aquí otros modelos neuronales de regresión si los tienes
        }
        modelos = []
        r2_scores = []
        for nombre, info in results_dict.items():
            if info.get('type') == 'regression' and info.get('model_category') == 'neural':
                modelos.append(nombre_amigable.get(nombre, nombre))
                r2_scores.append(info.get('score', 0))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        if modelos:
            barras = plt.bar(modelos, r2_scores, color='purple', alpha=0.8)
            plt.xlabel('Modelos Neuronales de Regresión', fontweight='bold')
            plt.ylabel('R²', fontweight='bold')
            plt.title('Comparativa de R² - Modelos Neuronales de Regresión', fontweight='bold')
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.2)
            plt.xticks(rotation=30, ha='right')
            for bar, r2 in zip(barras, r2_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No hay modelos neuronales de regresión', ha='center', va='center',
                     fontsize=14, style='italic', transform=plt.gca().transAxes)
            plt.title('Comparativa de R² - Modelos Neuronales de Regresión', fontweight='bold')
            plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def plot_combined_nn_comparison(self, results_dict, save_path=None):
        """
        Combined plot showing both classification and regression neural network models with their respective evaluation metrics.
        
        Args:
            results_dict: Dictionary containing model results with 'type', 'model_category', and 'score' keys
            save_path: Path to save the plot
        """
        nombre_amigable = {
            'poverty_classifier': 'Clasificador de Pobreza',
            'poverty_regressor': 'Regresor de Pobreza',
            # Add other neural network models here if needed
        }
        
        # Separate classification and regression models
        classification_models = []
        classification_scores = []
        regression_models = []
        regression_scores = []
        
        for nombre, info in results_dict.items():
            if info.get('model_category') == 'neural':
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
            bars1 = ax1.bar(classification_models, classification_scores, color='darkorange', alpha=0.8)
            ax1.set_xlabel('Modelos de Clasificación', fontweight='bold')
            ax1.set_ylabel('Precisión (Accuracy)', fontweight='bold')
            ax1.set_title('Modelos Neuronales de Clasificación', fontweight='bold')
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
            ax1.set_title('Modelos Neuronales de Clasificación', fontweight='bold')
            ax1.axis('off')
        
        # Regression models subplot
        if regression_models:
            bars2 = ax2.bar(regression_models, regression_scores, color='purple', alpha=0.8)
            ax2.set_xlabel('Modelos de Regresión', fontweight='bold')
            ax2.set_ylabel('R²', fontweight='bold')
            ax2.set_title('Modelos Neuronales de Regresión', fontweight='bold')
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
            ax2.set_title('Modelos Neuronales de Regresión', fontweight='bold')
            ax2.axis('off')
        
        plt.suptitle('Comparativa de Modelos Neuronales - Métodos de Evaluación', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Combined neural network comparison plot saved to {save_path}")
        
        plt.show()

    def generate_confusion_matrix(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray,
                                 save_path: str = None, class_names: List[str] = None) -> np.ndarray:
        """
        Generate and plot confusion matrix for classification neural network models
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test target
            save_path: Path to save the confusion matrix plot
            class_names: List of class names for labels (default: ['No Poverty', 'Poverty'])
            
        Returns:
            Confusion matrix as numpy array
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        model_config = self.model_configs[model_name]
        
        # Check if it's a classification model
        if model_config['type'] != 'classification':
            raise ValueError(f"Model {model_name} is not a classification model")
        
        # Make predictions
        y_pred_proba = model.predict(X_test)
        
        # Convert probabilities to classes
        if model_config['num_classes'] == 2:
            y_pred_classes = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred_classes = np.argmax(y_pred_proba, axis=1)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # Set default class names if not provided
        if class_names is None:
            if model_config['num_classes'] == 2:
                class_names = ['No Poverty', 'Poverty']
            else:
                class_names = [f'Class {i}' for i in range(model_config['num_classes'])]
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.xlabel('Predicted', fontweight='bold')
        plt.ylabel('Actual', fontweight='bold')
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}', fontweight='bold')
        
        # Add metrics text for binary classification
        if model_config['num_classes'] == 2:
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1_score:.3f}'
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
        # Print detailed metrics
        print(f"\n=== CONFUSION MATRIX DETAILS FOR {model_name.upper()} ===")
        if model_config['num_classes'] == 2:
            print(f"True Negatives (TN): {tn}")
            print(f"False Positives (FP): {fp}")
            print(f"False Negatives (FN): {fn}")
            print(f"True Positives (TP): {tp}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1_score:.4f}")
        else:
            print(f"Confusion Matrix Shape: {cm.shape}")
            print(f"Total Predictions: {np.sum(cm)}")
        
        return cm

    def generate_regression_metrics_matrix(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray,
                                         save_path: str = None) -> Dict:
        """
        Generate comprehensive metrics matrix for regression neural network models
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test target
            save_path: Path to save the metrics visualization
            
        Returns:
            Dictionary with regression metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        model_config = self.model_configs[model_name]
        
        # Check if it's a regression model
        if model_config['type'] != 'regression':
            raise ValueError(f"Model {model_name} is not a regression model")
        
        # Make predictions
        y_pred = model.predict(X_test).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        residuals = y_test - y_pred
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'predictions': y_pred,
            'residuals': residuals
        }
        
        # Create metrics visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Actual vs Predicted
        ax1.scatter(y_test, y_pred, alpha=0.6, color='purple')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Actual vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        ax2.scatter(y_pred, residuals, alpha=0.6, color='purple')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals Plot')
        ax2.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='purple')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residuals Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Metrics summary
        metrics_text = f'R² Score: {r2:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nMAPE: {mape:.2f}%'
        ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax4.set_title('Regression Metrics Summary')
        ax4.axis('off')
        
        plt.suptitle(f'Neural Network Regression Analysis - {model_name.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Regression metrics visualization saved to {save_path}")
        
        plt.show()
        
        # Print detailed metrics
        print(f"\n=== NEURAL NETWORK REGRESSION METRICS FOR {model_name.upper()} ===")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        return metrics

    def generate_all_confusion_matrices(self, X_test: np.ndarray, y_test: np.ndarray,
                                      save_dir: str = None) -> Dict:
        """
        Generate confusion matrices for all classification neural network models
        
        Args:
            X_test: Test features
            y_test: Test target
            save_dir: Directory to save confusion matrix plots
            
        Returns:
            Dictionary with confusion matrices for each model
        """
        confusion_matrices = {}
        
        for model_name in self.models.keys():
            try:
                model_config = self.model_configs[model_name]
                if model_config['type'] == 'classification':
                    # Classification model
                    if save_dir:
                        save_path = f"{save_dir}/confusion_matrix_{model_name}.png"
                    else:
                        save_path = None
                    
                    cm = self.generate_confusion_matrix(model_name, X_test, y_test, save_path)
                    confusion_matrices[model_name] = cm
                    
            except Exception as e:
                print(f"Error generating confusion matrix for {model_name}: {e}")
                continue
        
        return confusion_matrices

    def generate_all_regression_matrices(self, X_test: np.ndarray, y_test: np.ndarray,
                                       save_dir: str = None) -> Dict:
        """
        Generate regression metrics matrices for all regression neural network models
        
        Args:
            X_test: Test features
            y_test: Test target
            save_dir: Directory to save regression metrics visualizations
            
        Returns:
            Dictionary with regression metrics for each model
        """
        regression_matrices = {}
        
        for model_name in self.models.keys():
            try:
                model_config = self.model_configs[model_name]
                if model_config['type'] == 'regression':
                    # Regression model
                    if save_dir:
                        save_path = f"{save_dir}/regression_metrics_{model_name}.png"
                    else:
                        save_path = None
                    
                    metrics = self.generate_regression_metrics_matrix(model_name, X_test, y_test, save_path)
                    regression_matrices[model_name] = metrics
                    
            except Exception as e:
                print(f"Error generating regression metrics for {model_name}: {e}")
                continue
        
        return regression_matrices


if __name__ == "__main__":
    # Test neural network
    from data_analyzer import DataAnalyzer
    from feature_engineer import FeatureEngineer
    
    # Load and prepare data
    analyzer = DataAnalyzer("../data/poverty_dataset.csv")
    analyzer.load_data()
    
    engineer = FeatureEngineer()
    df_clean = engineer.handle_missing_values(analyzer.df)
    
    # Create target variable
    y = engineer.create_poverty_target(df_clean)
    
    # Prepare features
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
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create and train neural network
    nn = NeuralNetwork()
    
    # Create classification model
    model = nn.create_classification_model(
        input_dim=X_train.shape[1],
        num_classes=2,
        model_name='poverty_classifier'
    )
    
    # Train model
    results = nn.train_model(
        'poverty_classifier',
        X_train.values, y_train.values,
        X_val.values, y_val.values,
        epochs=50,
        batch_size=32
    )
    
    # Evaluate model
    eval_results = nn.evaluate_model('poverty_classifier', X_test.values, y_test.values)
    
    # Plot training history
    nn.plot_training_history('poverty_classifier') 