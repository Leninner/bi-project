"""
Excel Validator Module
Responsibility: Validate Excel files for poverty prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os


class ExcelValidator:
    """Validates Excel files for poverty prediction"""
    
    def __init__(self):
        """Initialize validator with required column specifications"""
        self.required_columns = [
            'persona_key', 'tiempo_id', 'anio', 'mes', 'sector_id', 
            'condact_id', 'sexo', 'ciudad_id', 'nivel_instruccion', 
            'estado_civil', 'edad', 'ingreso_laboral', 'ingreso_per_capita', 
            'horas_trabajo_semana', 'desea_trabajar_mas', 'disponible_trabajar_mas'
        ]
        
        self.column_types = {
            'persona_key': 'int64',
            'tiempo_id': 'int64', 
            'anio': 'int64',
            'mes': 'int64',
            'sector_id': 'int64',
            'condact_id': 'int64',
            'sexo': 'int64',
            'ciudad_id': 'int64',
            'nivel_instruccion': 'int64',
            'estado_civil': 'int64',
            'edad': 'int64',
            'ingreso_laboral': 'float64',
            'ingreso_per_capita': 'float64',
            'horas_trabajo_semana': 'int64',
            'desea_trabajar_mas': 'int64',
            'disponible_trabajar_mas': 'int64'
        }
        
        self.value_ranges = {
            'anio': (2000, 2030),
            'mes': (1, 12),
            'sector_id': (0, 9),
            'condact_id': (0, 9),
            'sexo': (1, 2),
            'nivel_instruccion': (0, 5),
            'estado_civil': (0, 6),
            'edad': (0, 120),
            'ingreso_laboral': (0, float('inf')),
            'ingreso_per_capita': (0, float('inf')),
            'horas_trabajo_semana': (0, 168),
            'desea_trabajar_mas': (0, 4),
            'disponible_trabajar_mas': (0, 1)
        }
    
    def validate_file(self, file_path: str) -> Tuple[bool, List[str], Optional[pd.DataFrame]]:
        """
        Validate Excel file for poverty prediction
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Tuple of (is_valid, error_messages, dataframe)
        """
        errors = []
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False, ["El archivo no existe"], None
            
            # Check file extension
            if not file_path.lower().endswith(('.xlsx', '.xls')):
                return False, ["El archivo debe ser un Excel (.xlsx o .xls)"], None
            
            # Read Excel file
            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                return False, [f"Error al leer el archivo Excel: {str(e)}"], None
            
            # Validate basic structure
            structure_valid, structure_errors = self._validate_structure(df)
            errors.extend(structure_errors)
            
            if not structure_valid:
                return False, errors, None
            
            # Validate data types
            type_valid, type_errors = self._validate_data_types(df)
            errors.extend(type_errors)
            
            # Validate value ranges
            range_valid, range_errors = self._validate_value_ranges(df)
            errors.extend(range_errors)
            
            # Validate tiempo_id format
            tiempo_valid, tiempo_errors = self._validate_tiempo_id(df)
            errors.extend(tiempo_errors)
            
            # Check for missing values
            missing_valid, missing_errors = self._validate_missing_values(df)
            errors.extend(missing_errors)
            
            is_valid = len(errors) == 0
            
            return is_valid, errors, df if is_valid else None
            
        except Exception as e:
            return False, [f"Error inesperado durante la validación: {str(e)}"], None
    
    def _validate_structure(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate basic structure of the dataframe"""
        errors = []
        
        # Check if dataframe is empty
        if df.empty:
            return False, ["El archivo está vacío"]
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Columnas faltantes: {', '.join(missing_columns)}")
        
        # Check for extra columns
        extra_columns = set(df.columns) - set(self.required_columns)
        if extra_columns:
            errors.append(f"Columnas adicionales detectadas (serán ignoradas): {', '.join(extra_columns)}")
        
        # Check minimum number of rows
        if len(df) < 1:
            errors.append("El archivo debe contener al menos una fila de datos")
        
        return len(errors) == 0, errors
    
    def _validate_data_types(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data types of columns"""
        errors = []
        
        for col, expected_type in self.column_types.items():
            if col not in df.columns:
                continue
                
            try:
                # Convert to expected type
                if expected_type == 'int64':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif expected_type == 'float64':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Check for conversion errors
                if df[col].isna().any():
                    errors.append(f"La columna '{col}' contiene valores no numéricos")
                    
            except Exception as e:
                errors.append(f"Error al convertir la columna '{col}' a tipo numérico: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _validate_value_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate value ranges for categorical and numerical columns"""
        errors = []
        
        for col, (min_val, max_val) in self.value_ranges.items():
            if col not in df.columns:
                continue
                
            # Check for values outside range
            invalid_mask = (df[col] < min_val) | (df[col] > max_val)
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                invalid_values = df.loc[invalid_mask, col].unique()
                errors.append(
                    f"La columna '{col}' contiene {invalid_count} valores fuera del rango válido "
                    f"({min_val}-{max_val}): {invalid_values[:5]}{'...' if len(invalid_values) > 5 else ''}"
                )
        
        return len(errors) == 0, errors
    
    def _validate_tiempo_id(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate tiempo_id format (YYYYMM)"""
        errors = []
        
        if 'tiempo_id' not in df.columns:
            return True, errors
        
        # Check if tiempo_id matches año and mes
        if 'anio' in df.columns and 'mes' in df.columns:
            expected_tiempo = df['anio'] * 100 + df['mes']
            mismatched = (df['tiempo_id'] != expected_tiempo).sum()
            
            if mismatched > 0:
                errors.append(
                    f"Se encontraron {mismatched} registros donde tiempo_id no coincide "
                    "con el formato YYYYMM basado en año y mes"
                )
        
        return len(errors) == 0, errors
    
    def _validate_missing_values(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check for missing values and provide warnings"""
        errors = []
        
        missing_counts = df[self.required_columns].isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0]
        
        if not columns_with_missing.empty:
            for col, count in columns_with_missing.items():
                percentage = (count / len(df)) * 100
                errors.append(
                    f"Advertencia: La columna '{col}' tiene {count} valores faltantes ({percentage:.1f}%)"
                )
        
        return True, errors  # Warnings only, don't fail validation
    
    def get_sample_excel_structure(self) -> Dict:
        """Get sample data structure for Excel template"""
        sample_data = {
            'persona_key': [1, 2, 3],
            'tiempo_id': [202505, 202505, 202505],
            'anio': [2025, 2025, 2025],
            'mes': [5, 5, 5],
            'sector_id': [2, 1, 0],
            'condact_id': [4, 1, 9],
            'sexo': [2, 1, 2],
            'ciudad_id': [10150, 10150, 10150],
            'nivel_instruccion': [3, 5, 3],
            'estado_civil': [1, 6, 6],
            'edad': [67, 64, 78],
            'ingreso_laboral': [200.0, 0.0, 0.0],
            'ingreso_per_capita': [100.0, 100.0, 110.0],
            'horas_trabajo_semana': [20, 0, 0],
            'desea_trabajar_mas': [4, 0, 0],
            'disponible_trabajar_mas': [0, 0, 0]
        }
        
        return sample_data 