"""
Script to create sample Excel file for testing the prediction app
"""
import pandas as pd
import numpy as np
import os

def create_sample_excel():
    """Create a sample Excel file with valid data structure"""
    
    # Sample data structure
    sample_data = {
        'persona_key': list(range(1, 101)),  # 100 records
        'tiempo_id': [202505] * 100,  # May 2025
        'anio': [2025] * 100,
        'mes': [5] * 100,
        'sector_id': np.random.randint(0, 10, 100),
        'condact_id': np.random.randint(0, 10, 100),
        'sexo': np.random.randint(1, 3, 100),
        'ciudad_id': np.random.randint(10000, 10200, 100),
        'nivel_instruccion': np.random.randint(0, 6, 100),
        'estado_civil': np.random.randint(0, 7, 100),
        'edad': np.random.randint(18, 80, 100),
        'ingreso_laboral': np.random.uniform(0, 2000, 100),
        'ingreso_per_capita': np.random.uniform(50, 1000, 100),
        'horas_trabajo_semana': np.random.randint(0, 60, 100),
        'desea_trabajar_mas': np.random.randint(0, 5, 100),
        'disponible_trabajar_mas': np.random.randint(0, 2, 100)
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Ensure some realistic patterns
    # People with higher education tend to have higher income
    high_education_mask = df['nivel_instruccion'] >= 4
    df.loc[high_education_mask, 'ingreso_laboral'] *= 1.5
    df.loc[high_education_mask, 'ingreso_per_capita'] *= 1.3
    
    # Working people have more hours
    working_mask = df['ingreso_laboral'] > 0
    df.loc[working_mask, 'horas_trabajo_semana'] = np.maximum(df.loc[working_mask, 'horas_trabajo_semana'], 20)
    
    # Round numerical values
    df['ingreso_laboral'] = df['ingreso_laboral'].round(2)
    df['ingreso_per_capita'] = df['ingreso_per_capita'].round(2)
    
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Save to Excel
    filename = 'uploads/sample_data.xlsx'
    df.to_excel(filename, index=False)
    
    print(f"Sample Excel file created: {filename}")
    print(f"Records: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Show some statistics
    print("\nSample Statistics:")
    print(f"Age range: {df['edad'].min()} - {df['edad'].max()}")
    print(f"Income range: ${df['ingreso_laboral'].min():.2f} - ${df['ingreso_laboral'].max():.2f}")
    print(f"Per capita income range: ${df['ingreso_per_capita'].min():.2f} - ${df['ingreso_per_capita'].max():.2f}")
    
    return filename

if __name__ == "__main__":
    create_sample_excel() 