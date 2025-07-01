# Poverty Prediction Machine Learning Pipeline

This project implements a comprehensive machine learning pipeline for poverty prediction using statistical feature extraction and artificial neural networks, following SOLID principles.

## Project Structure

```
ml_pipeline/
├── __init__.py
├── data_analyzer.py      # Análisis de datos y exploración
├── feature_engineer.py   # Extracción de características y preprocesamiento
├── linear_models.py      # Modelos de regresión lineal y logística
├── neural_network.py     # Redes neuronales artificiales
├── model_trainer.py      # Orquestación del pipeline
├── main.py              # Script de ejecución principal
├── requirements.txt     # Dependencias
└── README.md           # Este archivo
```

## SOLID Principles Implementation

### Single Responsibility Principle (SRP)
- **DataAnalyzer**: Responsible only for data analysis and exploration
- **FeatureEngineer**: Handles feature extraction and preprocessing
- **LinearModels**: Manages linear and logistic regression models
- **NeuralNetwork**: Handles Keras neural network models
- **ModelTrainer**: Orchestrates the complete pipeline

### Open/Closed Principle (OCP)
- Each class is open for extension but closed for modification
- New model types can be added without changing existing code
- New feature engineering methods can be implemented as separate methods

### Liskov Substitution Principle (LSP)
- All model classes follow consistent interfaces
- Evaluation methods work with any model type
- Feature engineering methods are interchangeable

### Interface Segregation Principle (ISP)
- Each class has focused, specific interfaces
- No class is forced to depend on methods it doesn't use
- Clear separation between analysis, engineering, and modeling

### Dependency Inversion Principle (DIP)
- High-level modules (ModelTrainer) don't depend on low-level modules
- Both depend on abstractions
- Easy to swap implementations (e.g., different model types)

## Features

### Statistical Feature Extraction
- **Interaction Features**: Multiplication and ratio interactions between numerical variables
- **Polynomial Features**: Squared and cubed terms for key variables
- **Binned Features**: Age and income categorization
- **Correlation Analysis**: Feature importance ranking based on correlations

### Machine Learning Models

#### Linear Models
- **Linear Regression**: Basic linear regression for continuous targets
- **Logistic Regression**: Binary classification for poverty prediction

#### Neural Networks
- **Classification Network**: Multi-layer perceptron for binary classification
- **Regression Network**: Multi-layer perceptron for continuous prediction
- **Advanced Features**: Dropout, batch normalization, early stopping

### Pipeline Components

1. **Data Analysis**: Análisis exploratorio de datos y estadísticas
2. **Feature Engineering**: Extracción de características estadísticas y preprocesamiento
3. **Model Training**: Múltiples modelos con ajuste de hiperparámetros
4. **Model Evaluation**: Validación cruzada y métricas de rendimiento
5. **Results Generation**: Generación de informes y visualizaciones

## Usage

### Basic Usage
```bash
cd ml_pipeline
python main.py
```

### Advanced Usage
```bash
python main.py --data_path ../data/poverty_dataset.csv \
               --output_dir ml_results \
               --target_method income_threshold
```

### Command Line Arguments
- `--data_path`: Path to the poverty dataset CSV file
- `--output_dir`: Directory to save results
- `--target_method`: Method to create poverty target variable
  - `income_threshold`: Binary classification based on income median
  - `multi_factor`: Multi-factor poverty indicator

## Output Structure

```
ml_results/
├── models/                    # Trained models
│   ├── preprocessing_*.pkl   # Preprocessing components
│   ├── linear_regression_*.pkl
│   ├── logistic_regression_*.pkl
│   └── poverty_classifier_*.h5
├── plots/                    # Visualizations
│   ├── model_comparison_*.png
│   ├── feature_importance_*.png
│   └── training_history_*.png
└── reports/                  # Analysis reports
    ├── data_analysis_*.txt
    └── results_summary_*.json
```

## Dataset Information

The pipeline is designed for the poverty dataset with the following characteristics:
- **Shape**: 55,493 rows × 14 columns
- **Target Variables**: `ingreso_per_capita`, `ingreso_laboral`
- **Key Features**: Age, work hours, sector, activity condition, gender
- **Data Types**: Numerical and categorical variables

## Model Performance

The pipeline automatically:
- Compares all trained models
- Selects the best performing model
- Generates performance visualizations
- Saves comprehensive evaluation reports

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

## Key Features

### Statistical Process Implementation
- **Correlation Analysis**: Identifies most important features
- **Feature Selection**: Automatic selection of top features
- **Dimensionality Reduction**: PCA for high-dimensional data
- **Statistical Preprocessing**: Standardization and normalization

### Neural Network Implementation
- **Multi-layer Architecture**: Configurable hidden layers
- **Regularization**: Dropout and L1/L2 regularization
- **Optimization**: Adam optimizer with learning rate scheduling
- **Early Stopping**: Prevents overfitting
- **Batch Normalization**: Improves training stability

## Extensibility

The pipeline is designed for easy extension:
- Add new model types by extending existing classes
- Implement new feature engineering methods
- Add custom evaluation metrics
- Integrate with different data sources

## Best Practices

- **Reproducibility**: All random seeds are set for consistent results
- **Error Handling**: Comprehensive error handling throughout the pipeline
- **Logging**: Detailed logging of all pipeline steps
- **Documentation**: Extensive docstrings and comments
- **Testing**: Modular design enables easy unit testing 