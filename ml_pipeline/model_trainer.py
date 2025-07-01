"""
Model Trainer Module
Responsibility: Orchestrate the training of all models and manage the complete ML pipeline
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from data_analyzer import DataAnalyzer
from feature_engineer import FeatureEngineer
from linear_models import LinearModels
from neural_network import NeuralNetwork


class ModelTrainer:
    """Orchestrates the complete machine learning pipeline for poverty prediction"""
    
    def __init__(self, data_path: str, output_dir: str = "ml_results", model_selection: str = "both"):
        """
        Initialize ModelTrainer
        
        Args:
            data_path: Path to the dataset
            output_dir: Directory to save results
            model_selection: 'linear', 'neural' o 'both'
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_selection = model_selection
        
        # Initialize components
        self.analyzer = DataAnalyzer(data_path)
        self.engineer = FeatureEngineer()
        self.linear_models = LinearModels()
        self.neural_networks = NeuralNetwork()
        
        # Results storage
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
        # Store test data for plotting
        self.X_test = None
        self.y_test = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)
    
    def run_data_analysis(self) -> Dict:
        """Run complete data analysis"""
        print("=" * 60)
        print("STEP 1: DATA ANALYSIS")
        print("=" * 60)
        
        analysis_results = self.analyzer.run_complete_analysis()
        self.analyzer.save_analysis_report(f"{self.output_dir}/reports/data_analysis_{self.timestamp}.txt")
        
        return analysis_results
    
    def prepare_features(self, target_method: str = 'income_threshold') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling"""
        print("=" * 60)
        print("STEP 2: FEATURE ENGINEERING")
        print("=" * 60)
        
        # Load data
        df = self.analyzer.load_data()
        
        # Handle missing values
        print("Handling missing values...")
        df_clean = self.engineer.handle_missing_values(df, strategy='median')
        
        # Create target variable
        print("Creating target variable...")
        y = self.engineer.create_poverty_target(df_clean, method=target_method)
        
        # Prepare features
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
        X_scaled = self.engineer.scale_features(X_encoded)
        
        # Final check for any remaining NaN values and clean them
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
        print(f"Target distribution:\n{y.value_counts()}")
        
        return X_scaled, y
    
    def train_linear_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Train linear models"""
        print("=" * 60)
        print("STEP 3: TRAINING LINEAR MODELS")
        print("=" * 60)
        
        linear_results = {}
        
        # Train Linear Regression
        print("\nTraining Linear Regression...")
        try:
            lr_results = self.linear_models.train_linear_regression(X_train, y_train, 'linear_regression')
            lr_eval = self.linear_models.evaluate_model('linear_regression', X_test, y_test)
            linear_results['linear_regression'] = {**lr_results, 'evaluation': lr_eval}
        except Exception as e:
            print(f"Error training Linear Regression: {e}")
        
        # Train Logistic Regression (for classification)
        print("\nTraining Logistic Regression...")
        try:
            log_results = self.linear_models.train_logistic_regression(X_train, y_train, 'logistic_regression')
            log_eval = self.linear_models.evaluate_model('logistic_regression', X_test, y_test)
            linear_results['logistic_regression'] = {**log_results, 'evaluation': log_eval}
        except Exception as e:
            print(f"Error training Logistic Regression: {e}")
        
        return linear_results
    
    def train_neural_networks(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series,
                             X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Train neural network models"""
        print("=" * 60)
        print("STEP 4: TRAINING NEURAL NETWORKS")
        print("=" * 60)
        
        nn_results = {}
        
        # Determine if it's classification or regression based on target
        is_classification = len(y_train.unique()) == 2
        
        if is_classification:
            # Classification neural network
            print("\nTraining Classification Neural Network...")
            try:
                model = self.neural_networks.create_classification_model(
                    input_dim=X_train.shape[1],
                    num_classes=2,
                    model_name='poverty_classifier'
                )
                
                train_results = self.neural_networks.train_model(
                    'poverty_classifier',
                    X_train.values, y_train.values,
                    X_val.values, y_val.values,
                    epochs=50,
                    batch_size=32
                )
                
                eval_results = self.neural_networks.evaluate_model(
                    'poverty_classifier',
                    X_test.values, y_test.values
                )
                
                nn_results['poverty_classifier'] = {**train_results, 'evaluation': eval_results}
                
            except Exception as e:
                print(f"Error training Classification Neural Network: {e}")
        
        else:
            # Regression neural network
            print("\nTraining Regression Neural Network...")
            try:
                model = self.neural_networks.create_regression_model(
                    input_dim=X_train.shape[1],
                    model_name='poverty_regressor'
                )
                
                train_results = self.neural_networks.train_model(
                    'poverty_regressor',
                    X_train.values, y_train.values,
                    X_val.values, y_val.values,
                    epochs=50,
                    batch_size=32
                )
                
                eval_results = self.neural_networks.evaluate_model(
                    'poverty_regressor',
                    X_test.values, y_test.values
                )
                
                nn_results['poverty_regressor'] = {**train_results, 'evaluation': eval_results}
                
            except Exception as e:
                print(f"Error training Regression Neural Network: {e}")
        
        return nn_results
    
    def compare_models(self) -> Dict:
        """Compare all trained models and select the best one"""
        print("=" * 60)
        print("STEP 5: MODEL COMPARISON")
        print("=" * 60)
        
        comparison_results = {}
        evaluation_log = []
        
        # Compare linear models
        print("\n--- LINEAR MODELS EVALUATION ---")
        for model_name, results in self.results.get('linear_models', {}).items():
            if 'evaluation' in results:
                eval_results = results['evaluation']
                print(f"\n📊 {model_name.upper()}:")
                
                if 'test_accuracy' in eval_results:
                    # Classification model
                    accuracy = eval_results['test_accuracy']
                    comparison_results[model_name] = {
                        'type': 'classification',
                        'metric': 'accuracy',
                        'score': accuracy,
                        'model_category': 'linear'
                    }
                    
                    # Log classification metrics
                    print(f"   ✅ Model Type: Classification")
                    print(f"   📈 Test Accuracy: {accuracy:.4f}")
                    
                    if 'classification_report' in eval_results:
                        report = eval_results['classification_report']
                        if '1' in report:
                            precision = report['1']['precision']
                            recall = report['1']['recall']
                            f1 = report['1']['f1-score']
                            print(f"   🎯 Precision: {precision:.4f}")
                            print(f"   🔍 Recall: {recall:.4f}")
                            print(f"   ⚖️  F1-Score: {f1:.4f}")
                    
                    evaluation_log.append({
                        'model': model_name,
                        'type': 'classification',
                        'category': 'linear',
                        'accuracy': accuracy,
                        'precision': precision if 'classification_report' in eval_results and '1' in eval_results['classification_report'] else None,
                        'recall': recall if 'classification_report' in eval_results and '1' in eval_results['classification_report'] else None,
                        'f1_score': f1 if 'classification_report' in eval_results and '1' in eval_results['classification_report'] else None
                    })
                    
                elif 'test_r2' in eval_results:
                    # Regression model
                    r2 = eval_results['test_r2']
                    mse = eval_results.get('test_mse', 0)
                    comparison_results[model_name] = {
                        'type': 'regression',
                        'metric': 'r2',
                        'score': r2,
                        'model_category': 'linear'
                    }
                    
                    # Log regression metrics
                    print(f"   ✅ Model Type: Regression")
                    print(f"   📈 Test R²: {r2:.4f}")
                    print(f"   📊 Test MSE: {mse:.4f}")
                    
                    evaluation_log.append({
                        'model': model_name,
                        'type': 'regression',
                        'category': 'linear',
                        'r2': r2,
                        'mse': mse
                    })
        
        # Compare neural networks
        print("\n--- NEURAL NETWORKS EVALUATION ---")
        for model_name, results in self.results.get('neural_networks', {}).items():
            if 'evaluation' in results:
                eval_results = results['evaluation']
                print(f"\n🧠 {model_name.upper()}:")
                
                if 'test_accuracy' in eval_results:
                    # Classification model
                    accuracy = eval_results['test_accuracy']
                    comparison_results[model_name] = {
                        'type': 'classification',
                        'metric': 'accuracy',
                        'score': accuracy,
                        'model_category': 'neural'
                    }
                    
                    # Log classification metrics
                    print(f"   ✅ Model Type: Classification")
                    print(f"   📈 Test Accuracy: {accuracy:.4f}")
                    
                    if 'classification_report' in eval_results:
                        report = eval_results['classification_report']
                        if '1' in report:
                            precision = report['1']['precision']
                            recall = report['1']['recall']
                            f1 = report['1']['f1-score']
                            print(f"   🎯 Precision: {precision:.4f}")
                            print(f"   🔍 Recall: {recall:.4f}")
                            print(f"   ⚖️  F1-Score: {f1:.4f}")
                    
                    evaluation_log.append({
                        'model': model_name,
                        'type': 'classification',
                        'category': 'neural',
                        'accuracy': accuracy,
                        'precision': precision if 'classification_report' in eval_results and '1' in eval_results['classification_report'] else None,
                        'recall': recall if 'classification_report' in eval_results and '1' in eval_results['classification_report'] else None,
                        'f1_score': f1 if 'classification_report' in eval_results and '1' in eval_results['classification_report'] else None
                    })
                    
                elif 'test_r2' in eval_results:
                    # Regression model
                    r2 = eval_results['test_r2']
                    mse = eval_results.get('test_mse', 0)
                    mae = eval_results.get('test_mae', 0)
                    comparison_results[model_name] = {
                        'type': 'regression',
                        'metric': 'r2',
                        'score': r2,
                        'model_category': 'neural'
                    }
                    
                    # Log regression metrics
                    print(f"   ✅ Model Type: Regression")
                    print(f"   📈 Test R²: {r2:.4f}")
                    print(f"   📊 Test MSE: {mse:.4f}")
                    print(f"   📏 Test MAE: {mae:.4f}")
                    
                    evaluation_log.append({
                        'model': model_name,
                        'type': 'regression',
                        'category': 'neural',
                        'r2': r2,
                        'mse': mse,
                        'mae': mae
                    })
        
        # Print comprehensive evaluation summary
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 60)
        
        if evaluation_log:
            print(f"\n📊 Total Models Evaluated: {len(evaluation_log)}")
            
            # Summary by type
            classification_models = [log for log in evaluation_log if log['type'] == 'classification']
            regression_models = [log for log in evaluation_log if log['type'] == 'regression']
            
            print(f"🎯 Classification Models: {len(classification_models)}")
            print(f"📈 Regression Models: {len(regression_models)}")
            
            # Summary by category
            linear_models = [log for log in evaluation_log if log['category'] == 'linear']
            neural_models = [log for log in evaluation_log if log['category'] == 'neural']
            
            print(f"🔧 Linear Models: {len(linear_models)}")
            print(f"🧠 Neural Networks: {len(neural_models)}")
            
            # Best models by type
            if classification_models:
                best_classification = max(classification_models, key=lambda x: x['accuracy'])
                print(f"\n🏆 Best Classification Model: {best_classification['model']} (Accuracy: {best_classification['accuracy']:.4f})")
            
            if regression_models:
                best_regression = max(regression_models, key=lambda x: x['r2'])
                print(f"🏆 Best Regression Model: {best_regression['model']} (R²: {best_regression['r2']:.4f})")
        
        # Find best model overall
        if comparison_results:
            best_model_name = max(comparison_results.keys(), 
                                key=lambda x: comparison_results[x]['score'])
            self.best_model = best_model_name
            self.best_score = comparison_results[best_model_name]['score']
            
            print(f"\n🥇 OVERALL BEST MODEL:")
            print(f"   Model: {best_model_name}")
            print(f"   Score: {self.best_score:.4f}")
            print(f"   Type: {comparison_results[best_model_name]['type']}")
            print(f"   Category: {comparison_results[best_model_name]['model_category']}")
        
        # Store evaluation log for later use
        self.evaluation_log = evaluation_log
        
        return comparison_results
    
    def generate_reports(self):
        """Genera reportes y comparativas según la selección de modelos"""
        print("=" * 60)
        print("STEP 6: GENERATING REPORTS")
        print("=" * 60)
        self.engineer.save_preprocessing_components(
            f"{self.output_dir}/models/preprocessing_{self.timestamp}.pkl"
        )
        for model_name in self.linear_models.models.keys():
            self.linear_models.save_model(
                model_name,
                f"{self.output_dir}/models/{model_name}_{self.timestamp}.pkl"
            )
        for model_name in self.neural_networks.models.keys():
            self.neural_networks.save_model(
                model_name,
                f"{self.output_dir}/models/{model_name}_{self.timestamp}.h5"
            )
        # Comparativas según selección
        if self.model_selection == "both":
            self.plot_model_comparison()
        self.plot_model_results(self.model_selection)
        self.save_results_summary()
        self.generate_confusion_matrices_and_metrics()
        print(f"\nAll results saved to: {self.output_dir}")
    
    def plot_model_comparison(self, tipo: str = "both"):
        """Comparativa visual según tipo de modelo ('linear', 'neural', 'both')"""
        comparison_results = self.compare_models()
        if not comparison_results:
            print("No hay modelos para comparar")
            return
    
        # --- GANADORES ---
        if tipo == "both":
            mejor_lineal = None
            mejor_neural = None
            mejor_acc_lineal = -1
            mejor_acc_neural = -1
            for nombre, info in comparison_results.items():
                if info.get('type') == 'classification' and info.get('model_category') == 'linear':
                    if info.get('score', 0) > mejor_acc_lineal:
                        mejor_lineal = nombre
                        mejor_acc_lineal = info.get('score', 0)
                if info.get('type') == 'classification' and info.get('model_category') == 'neural':
                    if info.get('score', 0) > mejor_acc_neural:
                        mejor_neural = nombre
                        mejor_acc_neural = info.get('score', 0)
            nombres = []
            accuracies = []
            colores = []
            if mejor_lineal is not None:
                nombres.append(f"Lineal: {mejor_lineal}")
                accuracies.append(mejor_acc_lineal)
                colores.append('royalblue')
            if mejor_neural is not None:
                nombres.append(f"Neuronal: {mejor_neural}")
                accuracies.append(mejor_acc_neural)
                colores.append('darkorange')
            plt.figure(figsize=(7, 5))
            if nombres:
                barras = plt.bar(nombres, accuracies, color=colores, alpha=0.8)
                plt.xlabel('Modelo Ganador', fontweight='bold')
                plt.ylabel('Precisión (Accuracy)', fontweight='bold')
                plt.title('Comparativa de Ganadores', fontweight='bold')
                plt.ylim(0, 1)
                plt.grid(axis='y', alpha=0.2)
                for bar, acc in zip(barras, accuracies):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
            else:
                plt.text(0.5, 0.5, 'No hay modelos ganadores', ha='center', va='center',
                         fontsize=14, style='italic', transform=plt.gca().transAxes)
                plt.title('Comparativa de Ganadores', fontweight='bold')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/plots/modelos_ganadores_{self.timestamp}.png", bbox_inches='tight', dpi=300)
            plt.show()
    
    def create_metrics_table(self):
        """Create a detailed metrics table for all evaluated models"""
        if not hasattr(self, 'evaluation_log') or not self.evaluation_log:
            return
        
        print("\n📋 DETAILED METRICS TABLE")
        print("=" * 80)
        
        # Create table headers
        print(f"{'Model Name':<25} {'Type':<15} {'Category':<10} {'Primary Metric':<15} {'Secondary Metrics':<30}")
        print("-" * 80)
        
        for log in self.evaluation_log:
            model_name = log['model'][:24]  # Truncate if too long
            model_type = log['type'][:14]
            category = log['category'][:9]
            
            if log['type'] == 'classification':
                primary_metric = f"Accuracy: {log['accuracy']:.4f}"
                secondary_metrics = []
                if log.get('precision') is not None:
                    secondary_metrics.append(f"P:{log['precision']:.3f}")
                if log.get('recall') is not None:
                    secondary_metrics.append(f"R:{log['recall']:.3f}")
                if log.get('f1_score') is not None:
                    secondary_metrics.append(f"F1:{log['f1_score']:.3f}")
                secondary_str = ", ".join(secondary_metrics)
            else:  # regression
                primary_metric = f"R²: {log['r2']:.4f}"
                secondary_metrics = []
                if log.get('mse') is not None:
                    secondary_metrics.append(f"MSE:{log['mse']:.3f}")
                if log.get('mae') is not None:
                    secondary_metrics.append(f"MAE:{log['mae']:.3f}")
                secondary_str = ", ".join(secondary_metrics)
            
            print(f"{model_name:<25} {model_type:<15} {category:<10} {primary_metric:<15} {secondary_str:<30}")
        
        print("-" * 80)
    
    def plot_model_results(self, tipo: str = "both"):
        """Orquesta la visualización de resultados de modelos lineales y neuronales usando los métodos de cada clase."""
        comparison_results = self.results.get('comparison', {})
        if not comparison_results:
            print("No hay resultados de comparación para graficar")
            return
        print("\nGenerando comparativas específicas de modelos...")
        
        # Combined plot for linear models (classification and regression)
        if tipo == "both" or tipo == "linear":
            self.linear_models.plot_combined_model_comparison(
                comparison_results,
                save_path=f"{self.output_dir}/plots/lineales_combinados_{self.timestamp}.png"
            )
        
        # Combined plot for neural networks (classification and regression)
        if tipo == "both" or tipo == "neural":
            self.neural_networks.plot_combined_nn_comparison(
                comparison_results,
                save_path=f"{self.output_dir}/plots/neuronales_combinados_{self.timestamp}.png"
            )
    
    def save_results_summary(self):
        """Save a comprehensive results summary"""
        summary = {
            'timestamp': self.timestamp,
            'data_path': self.data_path,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'model_comparison': self.results.get('comparison', {}),
            'detailed_evaluation_log': getattr(self, 'evaluation_log', []),
            'training_summary': {
                'linear_models_trained': list(self.linear_models.models.keys()),
                'neural_networks_trained': list(self.neural_networks.models.keys()),
                'total_models': len(self.linear_models.models) + len(self.neural_networks.models)
            }
        }
        
        with open(f"{self.output_dir}/reports/results_summary_{self.timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📄 Results summary saved to: {self.output_dir}/reports/results_summary_{self.timestamp}.json")

    def generate_confusion_matrices_and_metrics(self):
        """Generate confusion matrices and regression metrics for all trained models"""
        print("=" * 60)
        print("STEP 6.1: GENERATING CONFUSION MATRICES AND REGRESSION METRICS")
        print("=" * 60)
        
        if self.X_test is None or self.y_test is None:
            print("❌ Test data not available. Skipping confusion matrix generation.")
            return
        
        # Create directories for confusion matrices and regression metrics
        confusion_dir = f"{self.output_dir}/confusion_matrices"
        regression_dir = f"{self.output_dir}/regression_metrics"
        os.makedirs(confusion_dir, exist_ok=True)
        os.makedirs(regression_dir, exist_ok=True)
        
        print(f"\n📁 Saving confusion matrices to: {confusion_dir}")
        print(f"📁 Saving regression metrics to: {regression_dir}")
        
        # Generate confusion matrices and regression metrics for linear models
        if self.linear_models.models:
            print("\n🔧 Generating Linear Models Matrices...")
            try:
                # Generate confusion matrices for classification models
                linear_cm = self.linear_models.generate_all_confusion_matrices(
                    X_test=self.X_test,
                    y_test=self.y_test,
                    save_dir=confusion_dir
                )
                print(f"   ✅ Confusion matrices generated: {len(linear_cm)}")
                
                # Generate regression metrics for regression models
                linear_reg = self.linear_models.generate_all_regression_matrices(
                    X_test=self.X_test,
                    y_test=self.y_test,
                    save_dir=regression_dir
                )
                print(f"   ✅ Regression metrics generated: {len(linear_reg)}")
                
            except Exception as e:
                print(f"   ❌ Error generating linear models matrices: {e}")
        
        # Generate confusion matrices and regression metrics for neural networks
        if self.neural_networks.models:
            print("\n🧠 Generating Neural Network Matrices...")
            try:
                # Generate confusion matrices for classification models
                nn_cm = self.neural_networks.generate_all_confusion_matrices(
                    X_test=self.X_test.values,
                    y_test=self.y_test.values,
                    save_dir=confusion_dir
                )
                print(f"   ✅ Confusion matrices generated: {len(nn_cm)}")
                
                # Generate regression metrics for regression models
                nn_reg = self.neural_networks.generate_all_regression_matrices(
                    X_test=self.X_test.values,
                    y_test=self.y_test.values,
                    save_dir=regression_dir
                )
                print(f"   ✅ Regression metrics generated: {len(nn_reg)}")
                
            except Exception as e:
                print(f"   ❌ Error generating neural network matrices: {e}")
        
        # Create summary report
        self.create_matrices_summary_report(confusion_dir, regression_dir)
        
        print(f"\n✅ Confusion matrices and regression metrics generation completed!")

    def create_matrices_summary_report(self, confusion_dir: str, regression_dir: str):
        """Create a summary report of generated confusion matrices and regression metrics"""
        import glob
        
        # Count generated files
        confusion_files = glob.glob(f"{confusion_dir}/*.png")
        regression_files = glob.glob(f"{regression_dir}/*.png")
        
        summary = {
            'timestamp': self.timestamp,
            'confusion_matrices': {
                'directory': confusion_dir,
                'files_generated': len(confusion_files),
                'file_list': [os.path.basename(f) for f in confusion_files]
            },
            'regression_metrics': {
                'directory': regression_dir,
                'files_generated': len(regression_files),
                'file_list': [os.path.basename(f) for f in regression_files]
            },
            'total_files': len(confusion_files) + len(regression_files)
        }
        
        # Save summary report
        summary_path = f"{self.output_dir}/reports/matrices_summary_{self.timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📊 Matrices Summary Report:")
        print(f"   📁 Confusion Matrices: {len(confusion_files)} files")
        print(f"   📁 Regression Metrics: {len(regression_files)} files")
        print(f"   📄 Summary saved to: {summary_path}")
        
        # Print file list
        if confusion_files:
            print(f"\n📋 Confusion Matrix Files:")
            for file in confusion_files:
                print(f"   - {os.path.basename(file)}")
        
        if regression_files:
            print(f"\n📋 Regression Metrics Files:")
            for file in regression_files:
                print(f"   - {os.path.basename(file)}")

    def run_complete_pipeline(self, target_method: str = 'income_threshold', 
                             train_linear: bool = True, train_neural: bool = True) -> Dict:
        """Run the complete machine learning pipeline with selective model training"""
        print("=" * 80)
        print("STARTING COMPLETE MACHINE LEARNING PIPELINE")
        print("=" * 80)
        
        # Display training configuration
        print(f"Training Configuration:")
        print(f"- Linear Models: {'Yes' if train_linear else 'No'}")
        print(f"- Neural Networks: {'Yes' if train_neural else 'No'}")
        print(f"- Target Method: {target_method}")
        print("=" * 80)
        
        # Step 1: Data Analysis
        analysis_results = self.run_data_analysis()
        
        # Step 2: Feature Engineering
        X, y = self.prepare_features(target_method)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Store test data for plotting
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"\nData splits:")
        print(f"Training: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples")
        print(f"Test: {X_test.shape[0]} samples")
        
        # Step 3: Train Linear Models (if selected)
        if train_linear:
            linear_results = self.train_linear_models(X_train, y_train, X_test, y_test)
            self.results['linear_models'] = linear_results
        else:
            self.results['linear_models'] = {}
            print("\nSkipping Linear Models training...")
        
        # Step 4: Train Neural Networks (if selected)
        if train_neural:
            nn_results = self.train_neural_networks(X_train, y_train, X_val, y_val, X_test, y_test)
            self.results['neural_networks'] = nn_results
        else:
            self.results['neural_networks'] = {}
            print("\nSkipping Neural Networks training...")
        
        # Step 5: Model Comparison (only if models were trained)
        if train_linear or train_neural:
            comparison_results = self.compare_models()
            self.results['comparison'] = comparison_results
        else:
            self.results['comparison'] = {}
            print("\nNo models trained - skipping comparison...")
        
        # Step 6: Generate Reports (only if models were trained)
        if train_linear or train_neural:
            self.generate_reports()
        else:
            print("\nNo models trained - skipping report generation...")
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return self.results


if __name__ == "__main__":
    # This file should be run through main.py for command-line options
    print("Please run the pipeline using: python main.py --help")
    print("Example usage:")
    print("  python main.py --models linear")
    print("  python main.py --models neural")
    print("  python main.py --models both") 