"""
Main Execution Script
Responsibility: Main entry point for the poverty prediction ML pipeline
"""
import sys
import os
import argparse
from pathlib import Path

# Add the parent directory to the path to import modules
sys.path.append(str(Path(__file__).parent))

from model_trainer import ModelTrainer


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Poverty Prediction ML Pipeline')
    parser.add_argument('--data_path', type=str, default='data/poverty_dataset.csv',
                       help='Path to the poverty dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='ml_results',
                       help='Directory to save results')
    parser.add_argument('--target_method', type=str, default='multi_factor',
                       choices=['income_threshold', 'multi_factor'],
                       help='Method to create poverty target variable')
    
    # Model selection arguments
    parser.add_argument('--models', type=str, default='both',
                       choices=['linear', 'neural', 'both'],
                       help='Which models to train: linear (only linear models), neural (only neural networks), or both')
    
    # New argument for centralized models directory
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory to save centralized models for prediction app')
    
    args = parser.parse_args()
    
    # Determine which models to train based on the --models argument
    train_linear = args.models in ['linear', 'both']
    train_neural = args.models in ['neural', 'both']
    
    print("=" * 80)
    print("POVERTY PREDICTION MACHINE LEARNING PIPELINE")
    print("=" * 80)
    print(f"Data Path: {args.data_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Centralized Models Directory: {args.models_dir}")
    print("=" * 80)
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        sys.exit(1)
    
    try:
        # Initialize and run the pipeline
        trainer = ModelTrainer(args.data_path, args.output_dir, model_selection=args.models)
        results = trainer.run_complete_pipeline(
            target_method=args.target_method,
            train_linear=train_linear,
            train_neural=train_neural
        )
        
        # Copy best models to centralized directory for prediction app
        copy_best_models_to_centralized_dir(trainer, args.models_dir)
        
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Best Model: {trainer.best_model}")
        print(f"Best Score: {trainer.best_score:.4f}")
        print(f"Results saved to: {args.output_dir}")
        print(f"Models copied to: {args.models_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def copy_best_models_to_centralized_dir(trainer, models_dir):
    """Copy ALL trained models to centralized directory for prediction app"""
    import shutil
    from pathlib import Path
    
    print("\n" + "=" * 60)
    print("COPYING ALL TRAINED MODELS TO CENTRALIZED DIRECTORY")
    print("=" * 60)
    
    # Create centralized models directory
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    # Copy preprocessing components
    preprocessing_files = list(Path(trainer.output_dir).glob("models/preprocessing_*.pkl"))
    if preprocessing_files:
        latest_preprocessing = max(preprocessing_files, key=lambda x: x.stat().st_mtime)
        dest_preprocessing = models_path / "preprocessing_components.pkl"
        shutil.copy2(latest_preprocessing, dest_preprocessing)
        print(f"✅ Preprocessing components copied: {dest_preprocessing}")
    
    # Copy ALL linear models (not just the best one)
    print("\n📁 Copying Linear Models:")
    
    # Copy logistic regression
    logistic_files = list(Path(trainer.output_dir).glob("models/logistic_regression_*.pkl"))
    if logistic_files:
        latest_logistic = max(logistic_files, key=lambda x: x.stat().st_mtime)
        dest_logistic = models_path / "logistic_regression.pkl"
        shutil.copy2(latest_logistic, dest_logistic)
        print(f"   ✅ Logistic Regression copied: {dest_logistic}")
    
    # Copy linear regression
    linear_files = list(Path(trainer.output_dir).glob("models/linear_regression_*.pkl"))
    if linear_files:
        latest_linear = max(linear_files, key=lambda x: x.stat().st_mtime)
        dest_linear = models_path / "linear_regression.pkl"
        shutil.copy2(latest_linear, dest_linear)
        print(f"   ✅ Linear Regression copied: {dest_linear}")
    
    # Copy ALL neural models
    print("\n🧠 Copying Neural Models:")
    neural_files = list(Path(trainer.output_dir).glob("models/*_poverty_classifier_*.h5"))
    if neural_files:
        latest_neural = max(neural_files, key=lambda x: x.stat().st_mtime)
        dest_neural = models_path / "poverty_classifier.h5"
        shutil.copy2(latest_neural, dest_neural)
        print(f"   ✅ Neural Network copied: {dest_neural}")
    
    # Copy neural model info if exists
    neural_info_files = list(Path(trainer.output_dir).glob("models/*_poverty_classifier_*_info.pkl"))
    if neural_info_files:
        latest_neural_info = max(neural_info_files, key=lambda x: x.stat().st_mtime)
        dest_neural_info = models_path / "poverty_classifier_info.pkl"
        shutil.copy2(latest_neural_info, dest_neural_info)
        print(f"   ✅ Neural Network info copied: {dest_neural_info}")
    
    # Create a comprehensive summary file with model information
    summary_info = {
        'timestamp': trainer.timestamp,
        'best_model': trainer.best_model,
        'best_score': trainer.best_score,
        'models_available': [
            str(f.relative_to(models_path)) for f in models_path.glob("*") 
            if f.is_file() and not f.name.startswith('.')
        ],
        'training_summary': {
            'linear_models_trained': list(trainer.linear_models.models.keys()),
            'neural_networks_trained': list(trainer.neural_networks.models.keys())
        }
    }
    
    import json
    summary_file = models_path / "models_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_info, f, indent=2, default=str)
    
    print(f"\n✅ Models summary saved: {summary_file}")
    print(f"📁 Centralized models directory: {models_path.absolute()}")
    
    # Show what was copied
    print(f"\n📋 Models copied to centralized directory:")
    for file_path in models_path.glob("*"):
        if file_path.is_file() and not file_path.name.startswith('.'):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   📄 {file_path.name} ({size_mb:.2f} MB)")
    
    print("=" * 60)


if __name__ == "__main__":
    main() 