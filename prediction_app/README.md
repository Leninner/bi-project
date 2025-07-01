# Aplicación de Predicción de Pobreza

## Estructura del Proyecto

```
prediction_app/
├── app.py                 # Aplicación principal Flask
├── models/
│   ├── model_loader.py    # Cargador de modelos entrenados
│   └── predictor.py       # Lógica de predicción
├── utils/
│   ├── excel_validator.py # Validación de archivos Excel
│   └── data_processor.py  # Procesamiento de datos
├── static/
│   ├── css/
│   │   └── style.css      # Estilos CSS
│   └── js/
│       └── main.js        # JavaScript del frontend
├── templates/
│   ├── index.html         # Página principal
│   └── results.html       # Página de resultados
├── uploads/               # Carpeta temporal para archivos
├── requirements.txt       # Dependencias
└── README.md             # Este archivo
```

## Estructura del Excel Requerida

El archivo Excel debe contener las siguientes columnas:

### Columnas Obligatorias:
- `persona_key`: Identificador único de la persona (entero)
- `tiempo_id`: ID del tiempo (entero, formato: YYYYMM)
- `anio`: Año (entero, 4 dígitos)
- `mes`: Mes (entero, 1-12)
- `sector_id`: ID del sector laboral (entero, 0-9)
- `condact_id`: ID de condición de actividad (entero, 0-9)
- `sexo`: Sexo (entero, 1=Hombre, 2=Mujer)
- `ciudad_id`: ID de la ciudad (entero)
- `nivel_instruccion`: Nivel de instrucción (entero, 0-5)
- `estado_civil`: Estado civil (entero, 0-6)
- `edad`: Edad en años (entero, 0-120)
- `ingreso_laboral`: Ingreso laboral (decimal, >= 0)
- `ingreso_per_capita`: Ingreso per cápita (decimal, >= 0)
- `horas_trabajo_semana`: Horas de trabajo por semana (entero, 0-168)
- `desea_trabajar_mas`: Desea trabajar más (entero, 0-4)
- `disponible_trabajar_mas`: Disponible para trabajar más (entero, 0-1)

### Validaciones:
- Todas las columnas deben estar presentes
- Valores numéricos en rangos válidos
- No valores negativos en ingresos y horas
- Edad entre 0 y 120 años
- Códigos de categorías en rangos válidos

## Plan de Implementación

### Fase 1: Backend
1. Crear aplicación Flask
2. Implementar cargador de modelos
3. Crear sistema de validación de Excel
4. Implementar lógica de predicción

### Fase 2: Frontend
1. Crear interfaz de carga de archivos
2. Implementar selector de modelo
3. Crear tabla de resultados
4. Añadir estilos y UX

### Fase 3: Integración
1. Conectar frontend con backend
2. Manejo de errores
3. Testing y optimización

## Uso

### Prerrequisito: Modelos Entrenados

**IMPORTANTE**: Antes de usar la aplicación, necesitas tener modelos entrenados.

1. **Verificar modelos disponibles**:
   ```bash
   python check_models.py
   ```

2. **Si no hay modelos, entrenarlos**:
   ```bash
   cd ..
   python ml_pipeline/main.py --models both
   ```
   
   Esto creará una carpeta `models/` centralizada con:
   - `poverty_classifier.h5` (modelo neural)
   - `logistic_regression.pkl` (modelo lineal)
   - `preprocessing_components.pkl` (componentes de preprocesamiento)
   - `models_summary.json` (resumen de entrenamiento)

### Usar la Aplicación

1. **Instalar dependencias**: `pip install -r requirements.txt`
2. **Verificar modelos**: `python check_models.py`
3. **Ejecutar aplicación**: `python app.py`
4. **Abrir navegador**: `http://localhost:5000`
5. **Subir archivo Excel** con estructura correcta
6. **Seleccionar modelo** (Lineal o Neural)
7. **Ver resultados** de predicción

## Modelos Disponibles

- **Modelo Lineal**: Regresión logística para clasificación binaria
- **Modelo Neural**: Red neuronal para clasificación binaria

Ambos modelos predicen si una persona está en situación de pobreza (1) o no (0). 