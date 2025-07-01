"""
Script para verificar modelos disponibles
"""
import os
import sys
from pathlib import Path

def check_models():
    """Verificar qué modelos están disponibles"""
    
    print("=" * 60)
    print("VERIFICACIÓN DE MODELOS DISPONIBLES")
    print("=" * 60)
    
    # Buscar carpeta de modelos centralizada
    models_dir = Path("../models")
    
    if not models_dir.exists():
        print("❌ Carpeta de modelos centralizada no encontrada:")
        print(f"   Buscando en: {models_dir.absolute()}")
        print("\n📁 Estructura esperada:")
        print("   project/")
        print("   ├── models/")
        print("   │   ├── poverty_classifier.h5")
        print("   │   ├── logistic_regression.pkl")
        print("   │   ├── preprocessing_components.pkl")
        print("   │   └── models_summary.json")
        print("   └── prediction_app/")
        print("       └── app.py")
        print("\n🔧 Para crear la carpeta y modelos:")
        print("   cd ..")
        print("   python ml_pipeline/main.py --models both")
        return False
    
    print(f"✅ Carpeta de modelos centralizada encontrada: {models_dir.absolute()}")
    
    # Buscar archivos de modelos
    model_files = []
    
    # Modelos neurales (.h5, .hdf5)
    neural_models = list(models_dir.glob("*.h5")) + list(models_dir.glob("*.hdf5"))
    for model_file in neural_models:
        if model_file.name != "preprocessing_components.pkl":  # Exclude preprocessing
            model_files.append({
                'name': model_file.stem,
                'type': 'neural',
                'file': model_file,
                'size': model_file.stat().st_size
            })
    
    # Modelos lineales (.pkl)
    linear_models = list(models_dir.glob("*.pkl"))
    for model_file in linear_models:
        if model_file.name != "preprocessing_components.pkl":  # Exclude preprocessing
            model_files.append({
                'name': model_file.stem,
                'type': 'linear',
                'file': model_file,
                'size': model_file.stat().st_size
            })
    
    # Verificar componentes de preprocesamiento
    preprocessing_file = models_dir / "preprocessing_components.pkl"
    if preprocessing_file.exists():
        print(f"✅ Componentes de preprocesamiento encontrados: {preprocessing_file.name}")
    else:
        print("⚠️ Componentes de preprocesamiento no encontrados")
    
    # Verificar archivo de resumen
    summary_file = models_dir / "models_summary.json"
    if summary_file.exists():
        print(f"✅ Resumen de modelos encontrado: {summary_file.name}")
        try:
            import json
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            print(f"   - Mejor modelo: {summary.get('best_model', 'N/A')}")
            print(f"   - Mejor score: {summary.get('best_score', 'N/A')}")
        except:
            pass
    else:
        print("⚠️ Resumen de modelos no encontrado")
    
    if not model_files:
        print("❌ No se encontraron modelos en la carpeta")
        print("\n📋 Para crear modelos, ejecuta:")
        print("   cd ..")
        print("   python ml_pipeline/main.py --models both")
        return False
    
    print(f"\n✅ Modelos encontrados: {len(model_files)}")
    
    # Mostrar detalles de cada modelo
    for i, model in enumerate(model_files, 1):
        size_mb = model['size'] / (1024 * 1024)
        print(f"\n{i}. {model['name']}")
        print(f"   Tipo: {model['type']}")
        print(f"   Archivo: {model['file'].name}")
        print(f"   Tamaño: {size_mb:.2f} MB")
    
    # Verificar que tenemos al menos un modelo de cada tipo
    neural_count = len([m for m in model_files if m['type'] == 'neural'])
    linear_count = len([m for m in model_files if m['type'] == 'linear'])
    
    print(f"\n📊 Resumen:")
    print(f"   Modelos neurales: {neural_count}")
    print(f"   Modelos lineales: {linear_count}")
    
    if neural_count == 0:
        print("⚠️ No hay modelos neurales disponibles")
    if linear_count == 0:
        print("⚠️ No hay modelos lineales disponibles")
    
    if neural_count > 0 and linear_count > 0:
        print("✅ Tienes modelos de ambos tipos disponibles")
        return True
    else:
        print("\n🔧 Para crear modelos faltantes:")
        print("   cd ..")
        print("   python ml_pipeline/main.py --models both")
        return False

def test_model_loading():
    """Probar la carga de modelos"""
    print("\n" + "=" * 60)
    print("PRUEBA DE CARGA DE MODELOS")
    print("=" * 60)
    
    try:
        from models.predictor import PovertyPredictor
        
        predictor = PovertyPredictor()
        
        if predictor.models_available:
            print("✅ Los modelos se cargan correctamente")
            
            available_models = predictor.get_available_models()
            print(f"   Modelos cargados: {len(available_models)}")
            
            for model_name, info in available_models.items():
                print(f"   - {model_name}: {info['type']}")
                
            return True
        else:
            print("❌ Los modelos no se cargan correctamente")
            return False
            
    except Exception as e:
        print(f"❌ Error al cargar modelos: {e}")
        return False

if __name__ == "__main__":
    # Verificar modelos disponibles
    models_ok = check_models()
    
    if models_ok:
        # Probar carga de modelos
        loading_ok = test_model_loading()
        
        if loading_ok:
            print("\n" + "=" * 60)
            print("✅ TODO LISTO PARA USAR LA APLICACIÓN")
            print("=" * 60)
            print("Ejecuta: python app.py")
        else:
            print("\n❌ Hay problemas con la carga de modelos")
    else:
        print("\n❌ Necesitas crear modelos primero") 