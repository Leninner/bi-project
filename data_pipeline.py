import pandas as pd
import mysql.connector
from mysql.connector import Error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

def connect_to_mysql():
    """Conectar a la base de datos MySQL"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            port=3306,
            database='poverty_analysis',
            user='analyst',
            password='analyst123'
        )
        return connection
    except Error as e:
        print(f"Error conectando a MySQL: {e}")
        return None

def extract_full_dataset():
    """Extraer dataset completo de MySQL"""
    connection = connect_to_mysql()
    
    if connection is None:
        return None
    
    try:
        print("Extrayendo datos de MySQL...")
        query = """
SELECT f.persona_key,
       f.tiempo_id,
       t.anio,
       t.mes,
       s.sector_id,
       ca.condact_id,
       p.sexo,
       c.ciudad_id,
       p.nivel_instruccion,
       p.estado_civil,
       p.edad,
       f.ingreso_laboral,
       f.ingreso_per_capita,
       f.horas_trabajo_semana,
       f.desea_trabajar_mas,
       f.disponible_trabajar_mas
FROM fact_ocupacion f
         LEFT JOIN dim_ciudad c ON f.ciudad_id = c.ciudad_id
         LEFT JOIN dim_persona p ON f.persona_key = p.persona_key
         LEFT JOIN dim_tiempo t ON f.tiempo_id = t.tiempo_id
         LEFT JOIN dim_sectorlaboral s ON f.sector_id = s.sector_id
         LEFT JOIN dim_condicionactividad ca ON f.condact_id = ca.condact_id
        """
        df = pd.read_sql(query, connection)
        print(f"Dataset extraído: {df.shape}")
        return df
        
    except Error as e:
        print(f"Error ejecutando consulta: {e}")
        return None
    finally:
        if connection.is_connected():
            connection.close()

def save_to_csv(df, filename='poverty_dataset.csv', folder='data'):
    """Guardar dataset como CSV"""
    # Crear carpeta si no existe
    os.makedirs(folder, exist_ok=True)
    
    # Construir ruta completa
    filepath = os.path.join(folder, filename)
    df.to_csv(filepath, index=False)
    print(f"Dataset guardado como {filepath}")

def prepare_ml_data(df):
    """Preparar datos para ML - identificar variables de pobreza"""
    print("\n=== ANÁLISIS DE VARIABLES PARA PREDICCIÓN DE POBREZA ===")
    
    # Mostrar columnas disponibles
    print(f"Columnas disponibles: {list(df.columns)}")
    
    # Identificar variables numéricas y categóricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nVariables numéricas ({len(numeric_cols)}): {numeric_cols}")
    print(f"Variables categóricas ({len(categorical_cols)}): {categorical_cols}")
    
    # Mostrar estadísticas básicas
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Identificar posibles variables objetivo (pobreza)
    poverty_indicators = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['pobreza', 'pobre', 'ingreso', 'salario', 'renta', 'nbi']):
            poverty_indicators.append(col)
    
    print(f"\nPosibles indicadores de pobreza: {poverty_indicators}")
    
    return df, numeric_cols, categorical_cols, poverty_indicators

def create_train_test_splits(df, target_column=None, test_size=0.2, val_size=0.2):
    """Crear conjuntos de entrenamiento, validación y test"""
    print(f"\n=== CREANDO CONJUNTOS DE DATOS ===")
    
    # Si no hay columna objetivo, usar la primera numérica como ejemplo
    if target_column is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            target_column = numeric_cols[0]
            print(f"Usando {target_column} como variable objetivo (ejemplo)")
    
    # Separar features y target
    X = df.drop(columns=[target_column]) if target_column else df
    y = df[target_column] if target_column else None
    
    # Crear train/val/test splits
    if y is not None:
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
        
        print(f"Train: {X_train.shape[0]} muestras")
        print(f"Validation: {X_val.shape[0]} muestras") 
        print(f"Test: {X_test.shape[0]} muestras")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        # Solo features
        X_temp, X_test = train_test_split(X, test_size=test_size, random_state=42)
        X_train, X_val = train_test_split(X_temp, test_size=val_size, random_state=42)
        
        print(f"Train: {X_train.shape[0]} muestras")
        print(f"Validation: {X_val.shape[0]} muestras")
        print(f"Test: {X_test.shape[0]} muestras")
        
        return X_train, X_val, X_test, None, None, None

def main():
    """Pipeline principal"""
    print("=== PIPELINE DE DATOS PARA ANÁLISIS DE POBREZA ===")
    
    # Crear carpeta de datos
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)
    print(f"Carpeta de datos: {data_folder}")
    
    # 1. Extraer datos de MySQL
    df = extract_full_dataset()
    if df is None:
        return
    
    # 2. Guardar como CSV
    save_to_csv(df, folder=data_folder)
    
    # 3. Preparar para ML
    df, numeric_cols, categorical_cols, poverty_indicators = prepare_ml_data(df)
    
    # 4. Crear splits
    splits = create_train_test_splits(df)
    
    # 5. Guardar splits
    if len(splits) == 6:
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        pd.concat([X_train, y_train], axis=1).to_csv(os.path.join(data_folder, 'train_data.csv'), index=False)
        pd.concat([X_val, y_val], axis=1).to_csv(os.path.join(data_folder, 'val_data.csv'), index=False)
        pd.concat([X_test, y_test], axis=1).to_csv(os.path.join(data_folder, 'test_data.csv'), index=False)
    else:
        X_train, X_val, X_test = splits[:3]
        X_train.to_csv(os.path.join(data_folder, 'train_data.csv'), index=False)
        X_val.to_csv(os.path.join(data_folder, 'val_data.csv'), index=False)
        X_test.to_csv(os.path.join(data_folder, 'test_data.csv'), index=False)
    
    print("\n=== ARCHIVOS CREADOS ===")
    print(f"- {data_folder}/poverty_dataset.csv (dataset completo)")
    print(f"- {data_folder}/train_data.csv (entrenamiento)")
    print(f"- {data_folder}/val_data.csv (validación)")
    print(f"- {data_folder}/test_data.csv (test)")
    
    print("\n✅ Pipeline completado. Datos listos para Keras!")

if __name__ == "__main__":
    main() 