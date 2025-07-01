"""
Predictor Module
Responsibility: Main prediction orchestration for poverty prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.excel_validator import ExcelValidator
from utils.data_processor import DataProcessor
from models.model_loader import ModelLoader


class PovertyPredictor:
    """Main predictor class for poverty prediction"""
    
    def __init__(self, models_dir: str = "../models", preprocessing_path: str = None):
        """
        Initialize PovertyPredictor
        
        Args:
            models_dir: Directory containing trained models
            preprocessing_path: Path to saved preprocessing components
        """
        self.validator = ExcelValidator()
        self.processor = DataProcessor(preprocessing_path)
        self.model_loader = ModelLoader(models_dir)
        
        # Check if trained models are available
        self.models_available = self._ensure_models_available()
    
    def _ensure_models_available(self):
        """Check if trained models are available"""
        available_models = self.model_loader.get_available_models()
        
        if not available_models:
            print("⚠️ ADVERTENCIA: No se encontraron modelos entrenados.")
            print("   Para usar la aplicación, necesitas:")
            print("   1. Modelos neurales (.h5) en la carpeta ../models/")
            print("   2. Modelos lineales (.pkl) en la carpeta ../models/")
            print("   3. O ejecutar el pipeline de entrenamiento primero")
            return False
        else:
            print(f"✅ Modelos encontrados: {len(available_models)}")
            for model_name, info in available_models.items():
                print(f"   - {model_name}: {info['type']}")
            return True
    
    def predict_from_excel(self, file_path: str, model_type: str = 'neural') -> Dict:
        """
        Make predictions from Excel file using trained models
        
        Args:
            file_path: Path to Excel file
            model_type: Type of model to use ('linear' or 'neural')
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Check if models are available
            if not self.models_available:
                return {
                    'success': False,
                    'errors': [
                        "No hay modelos entrenados disponibles.",
                        "Ejecuta primero el pipeline de entrenamiento:",
                        "python ml_pipeline/main.py --models both"
                    ],
                    'predictions': None,
                    'summary': None
                }
            
            # Step 1: Validate Excel file
            print("Step 1: Validating Excel file...")
            is_valid, errors, df = self.validator.validate_file(file_path)
            
            if not is_valid:
                return {
                    'success': False,
                    'errors': errors,
                    'predictions': None,
                    'summary': None
                }
            
            # Step 2: Process data
            print("Step 2: Processing data...")
            X_processed = self.processor.process_data(df)
            
            # Validate processed data
            data_valid, data_errors = self.processor.validate_processed_data(X_processed)
            if not data_valid:
                return {
                    'success': False,
                    'errors': data_errors,
                    'predictions': None,
                    'summary': None
                }
            
            # Step 3: Select model
            print("Step 3: Selecting model...")
            model_name = self._select_model(model_type)
            if not model_name:
                return {
                    'success': False,
                    'errors': [f"No hay modelo {model_type} disponible. Modelos disponibles: {list(self.model_loader.get_available_models().keys())}"],
                    'predictions': None,
                    'summary': None
                }
            
            # Step 4: Make predictions
            print("Step 4: Making predictions...")
            predictions = self.model_loader.predict(model_name, X_processed)
            probabilities = self.model_loader.predict_proba(model_name, X_processed)
            
            # Step 5: Prepare results
            print("Step 5: Preparing results...")
            results = self._prepare_results(df, predictions, probabilities, model_name)
            
            return {
                'success': True,
                'errors': [],
                'predictions': results,
                'summary': self._create_summary(results, model_name)
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Error during prediction: {str(e)}"],
                'predictions': None,
                'summary': None
            }
    
    def _select_model(self, model_type: str) -> Optional[str]:
        """
        Select appropriate model based on type
        
        Args:
            model_type: Type of model ('linear', 'logistic', or 'neural')
            
        Returns:
            Model name or None if not available
        """
        available_models = self.model_loader.get_available_models()
        
        # Look for models of the specified type
        for model_name, info in available_models.items():
            if info['type'] == model_type:
                return model_name
        
        # If no specific type found, return first available model
        if available_models:
            return list(available_models.keys())[0]
        
        return None
    
    def _prepare_results(self, df: pd.DataFrame, predictions: np.ndarray, 
                        probabilities: np.ndarray, model_name: str) -> pd.DataFrame:
        """
        Prepare prediction results as DataFrame
        
        Args:
            df: Original input data
            predictions: Model predictions
            probabilities: Prediction probabilities
            model_name: Name of the model used
            
        Returns:
            DataFrame with results
        """
        results = df.copy()
        
        # Add prediction columns
        results['prediccion_pobreza'] = predictions
        results['probabilidad_pobreza'] = probabilities
        results['estado_pobreza'] = results['prediccion_pobreza'].map({
            0: 'No Pobre',
            1: 'Pobre'
        })
        results['confianza'] = np.where(
            results['prediccion_pobreza'] == 1,
            results['probabilidad_pobreza'],
            1 - results['probabilidad_pobreza']
        )
        
        # Add demographic information
        results['genero'] = results['sexo'].map({
            1: 'Hombre',
            2: 'Mujer'
        })
        
        results['nivel_educativo'] = results['nivel_instruccion'].map({
            0: 'Sin educación',
            1: 'Primaria',
            2: 'Secundaria',
            3: 'Bachillerato',
            4: 'Superior',
            5: 'Postgrado'
        })
        
        results['estado_civil_desc'] = results['estado_civil'].map({
            0: 'Soltero',
            1: 'Casado',
            2: 'Unión libre',
            3: 'Divorciado',
            4: 'Viudo',
            5: 'Separado',
            6: 'Otro'
        })
        
        # Add employment analysis
        results['sector_economico'] = results['sector_id'].map({
            0: 'Sin sector',
            1: 'Agricultura',
            2: 'Industria',
            3: 'Construcción',
            4: 'Comercio',
            5: 'Servicios',
            6: 'Transporte',
            7: 'Finanzas',
            8: 'Administración',
            9: 'Otros'
        })
        
        results['condicion_actividad'] = results['condact_id'].map({
            0: 'Ocupado',
            1: 'Desempleado',
            2: 'Inactivo',
            3: 'Jubilado',
            4: 'Estudiante',
            5: 'Ama de casa',
            6: 'Incapacitado',
            7: 'Otro inactivo',
            8: 'Trabajador familiar',
            9: 'Sin información'
        })
        
        # Add income analysis
        results['categoria_ingreso'] = pd.cut(
            results['ingreso_laboral'], 
            bins=[-float('inf'), 0, 200, 500, 1000, float('inf')],
            labels=['Sin ingreso', 'Bajo', 'Medio-bajo', 'Medio', 'Alto']
        )
        
        results['categoria_ingreso_per_capita'] = pd.cut(
            results['ingreso_per_capita'],
            bins=[-float('inf'), 100, 300, 600, 1000, float('inf')],
            labels=['Muy bajo', 'Bajo', 'Medio', 'Medio-alto', 'Alto']
        )
        
        # Add work intensity analysis
        results['intensidad_laboral'] = pd.cut(
            results['horas_trabajo_semana'],
            bins=[-float('inf'), 0, 20, 40, 60, float('inf')],
            labels=['Sin trabajo', 'Parcial', 'Tiempo completo', 'Extendido', 'Excesivo']
        )
        
        # Add subemployment indicators
        results['subempleo_por_horas'] = ((results['horas_trabajo_semana'] > 0) & 
                                         (results['horas_trabajo_semana'] < 40) & 
                                         (results['desea_trabajar_mas'] > 0)).astype(int)
        
        results['subempleo_por_ingresos'] = ((results['ingreso_laboral'] > 0) & 
                                            (results['ingreso_laboral'] < 500)).astype(int)
        
        results['indicador_subempleo'] = ((results['subempleo_por_horas'] == 1) | 
                                         (results['subempleo_por_ingresos'] == 1)).astype(int)
        
        # Add risk factors
        results['factores_riesgo'] = (
            (results['ingreso_laboral'] < 300).astype(int) +
            (results['ingreso_per_capita'] < 200).astype(int) +
            (results['nivel_instruccion'] < 3).astype(int) +
            (results['edad'] > 65).astype(int) +
            (results['condact_id'] == 1).astype(int)  # Desempleado
        )
        
        results['nivel_riesgo'] = pd.cut(
            results['factores_riesgo'],
            bins=[-float('inf'), 0, 1, 2, 3, float('inf')],
            labels=['Sin riesgo', 'Bajo', 'Medio', 'Alto', 'Muy alto']
        )
        
        # Add model information
        results['modelo_usado'] = model_name
        
        # Convert numpy types to native Python types for JSON serialization
        results['prediccion_pobreza'] = results['prediccion_pobreza'].astype(int)
        results['probabilidad_pobreza'] = results['probabilidad_pobreza'].astype(float)
        results['confianza'] = results['confianza'].astype(float)
        results['subempleo_por_horas'] = results['subempleo_por_horas'].astype(int)
        results['subempleo_por_ingresos'] = results['subempleo_por_ingresos'].astype(int)
        results['indicador_subempleo'] = results['indicador_subempleo'].astype(int)
        results['factores_riesgo'] = results['factores_riesgo'].astype(int)
        
        # Reorder columns for better readability
        important_cols = [
            'persona_key', 'prediccion_pobreza', 'estado_pobreza', 
            'probabilidad_pobreza', 'confianza', 'modelo_usado',
            'genero', 'edad', 'nivel_educativo', 'estado_civil_desc',
            'sector_economico', 'condicion_actividad', 'intensidad_laboral',
            'categoria_ingreso', 'categoria_ingreso_per_capita',
            'indicador_subempleo', 'nivel_riesgo'
        ]
        other_cols = [col for col in results.columns if col not in important_cols]
        
        return results[important_cols + other_cols]
    
    def _create_summary(self, results: pd.DataFrame, model_name: str) -> Dict:
        """
        Create summary statistics for predictions
        
        Args:
            results: Prediction results DataFrame
            model_name: Name of the model used
            
        Returns:
            Dictionary with summary statistics
        """
        total_records = len(results)
        pobre_count = (results['prediccion_pobreza'] == 1).sum()
        no_pobre_count = (results['prediccion_pobreza'] == 0).sum()
        
        pobre_percentage = (pobre_count / total_records) * 100
        no_pobre_percentage = (no_pobre_count / total_records) * 100
        
        avg_confidence = results['confianza'].mean()
        avg_probability = results['probabilidad_pobreza'].mean()
        
        # Confidence distribution
        high_confidence = (results['confianza'] >= 0.8).sum()
        medium_confidence = ((results['confianza'] >= 0.6) & (results['confianza'] < 0.8)).sum()
        low_confidence = (results['confianza'] < 0.6).sum()
        
        # Demographic analysis
        gender_distribution = results['genero'].value_counts().to_dict()
        age_groups = pd.cut(results['edad'], bins=[0, 25, 45, 65, 120], 
                           labels=['Joven', 'Adulto', 'Maduro', 'Mayor']).value_counts().to_dict()
        
        # Education analysis
        education_distribution = results['nivel_educativo'].value_counts().to_dict()
        
        # Employment analysis
        sector_distribution = results['sector_economico'].value_counts().to_dict()
        activity_distribution = results['condicion_actividad'].value_counts().to_dict()
        
        # Income analysis
        income_distribution = results['categoria_ingreso'].value_counts().to_dict()
        per_capita_distribution = results['categoria_ingreso_per_capita'].value_counts().to_dict()
        
        # Subemployment analysis
        subemployment_count = results['indicador_subempleo'].sum()
        subemployment_percentage = (subemployment_count / total_records) * 100
        
        # Risk analysis
        risk_distribution = results['nivel_riesgo'].value_counts().to_dict()
        high_risk_count = (results['nivel_riesgo'].isin(['Alto', 'Muy alto'])).sum()
        high_risk_percentage = (high_risk_count / total_records) * 100
        
        # Work intensity analysis
        work_intensity_distribution = results['intensidad_laboral'].value_counts().to_dict()
        
        # Poverty by demographic groups
        pobre_by_gender = results[results['prediccion_pobreza'] == 1]['genero'].value_counts().to_dict()
        pobre_by_education = results[results['prediccion_pobreza'] == 1]['nivel_educativo'].value_counts().to_dict()
        pobre_by_sector = results[results['prediccion_pobreza'] == 1]['sector_economico'].value_counts().to_dict()
        
        # Average income by poverty status
        avg_income_pobre = results[results['prediccion_pobreza'] == 1]['ingreso_laboral'].mean()
        avg_income_no_pobre = results[results['prediccion_pobreza'] == 0]['ingreso_laboral'].mean()
        avg_per_capita_pobre = results[results['prediccion_pobreza'] == 1]['ingreso_per_capita'].mean()
        avg_per_capita_no_pobre = results[results['prediccion_pobreza'] == 0]['ingreso_per_capita'].mean()
        
        return {
            # Basic statistics
            'total_registros': int(total_records),
            'pobre_count': int(pobre_count),
            'no_pobre_count': int(no_pobre_count),
            'pobre_percentage': round(float(pobre_percentage), 2),
            'no_pobre_percentage': round(float(no_pobre_percentage), 2),
            'avg_confidence': round(float(avg_confidence), 3),
            'avg_probability': round(float(avg_probability), 3),
            'high_confidence_count': int(high_confidence),
            'medium_confidence_count': int(medium_confidence),
            'low_confidence_count': int(low_confidence),
            'model_name': model_name,
            'model_type': self.model_loader.model_info[model_name]['type'],
            
            # Demographic statistics
            'gender_distribution': gender_distribution,
            'age_groups': age_groups,
            'education_distribution': education_distribution,
            
            # Employment statistics
            'sector_distribution': sector_distribution,
            'activity_distribution': activity_distribution,
            'work_intensity_distribution': work_intensity_distribution,
            
            # Income statistics
            'income_distribution': income_distribution,
            'per_capita_distribution': per_capita_distribution,
            'avg_income_pobre': round(float(avg_income_pobre), 2),
            'avg_income_no_pobre': round(float(avg_income_no_pobre), 2),
            'avg_per_capita_pobre': round(float(avg_per_capita_pobre), 2),
            'avg_per_capita_no_pobre': round(float(avg_per_capita_no_pobre), 2),
            
            # Subemployment statistics
            'subemployment_count': int(subemployment_count),
            'subemployment_percentage': round(float(subemployment_percentage), 2),
            
            # Risk statistics
            'risk_distribution': risk_distribution,
            'high_risk_count': int(high_risk_count),
            'high_risk_percentage': round(float(high_risk_percentage), 2),
            
            # Poverty by groups
            'pobre_by_gender': pobre_by_gender,
            'pobre_by_education': pobre_by_education,
            'pobre_by_sector': pobre_by_sector
        }
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Get information about available models"""
        return self.model_loader.get_available_models()
    
    def get_sample_excel_structure(self) -> Dict:
        """Get sample Excel structure for template"""
        return self.validator.get_sample_excel_structure()
    
    def validate_excel_structure(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate Excel file structure without processing
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        is_valid, errors, _ = self.validator.validate_file(file_path)
        return is_valid, errors
    
    def get_model_info(self, model_name: str) -> Dict:
        """
        Get detailed information about a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        return self.model_loader.get_model_summary(model_name) 