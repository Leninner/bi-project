# Documentación del Proyecto de Predicción de Pobreza

## Módulo de Feature Engineering (`ml_pipeline/feature_engineer.py`)

### ¿Qué es Feature Engineering?

**Feature Engineering** es el proceso de crear, transformar, extraer y seleccionar las variables (features) que se usarán para entrenar modelos de machine learning. Es una disciplina fundamental que combina:

- **Diseño y Construcción**: Crear nuevas variables a partir de las existentes
- **Optimización y Mejora**: Seleccionar las mejores características y reducir dimensionalidad
- **Aplicación de Técnicas Especializadas**: Transformar datos brutos en representaciones útiles

### Analogía con Ingeniería Tradicional

Así como un ingeniero civil diseña puentes o un ingeniero mecánico diseña máquinas, un **Feature Engineer** diseña y construye las características que alimentarán el modelo de ML. Requiere:

- **Creatividad**: Inventar nuevas formas de representar los datos
- **Conocimiento técnico**: Aplicar técnicas matemáticas y estadísticas
- **Experiencia**: Saber qué funciona en diferentes contextos
- **Iteración**: Mejorar continuamente las features

### Componentes Importados y sus Funciones

#### Librerías Principales

##### **pandas (pd)**
- **DataFrame**: Estructura de datos tabular bidimensional con columnas etiquetadas
- **Series**: Array unidimensional etiquetado (como una columna de DataFrame)

##### **numpy (np)**
- **where()**: Función condicional que permite aplicar lógica if/else vectorizada
- **number**: Tipo de datos para identificar columnas numéricas

##### **typing**
- **Dict, List, Tuple, Optional**: Anotaciones de tipo para documentar tipos de datos esperados

#### Scikit-learn Components

##### **sklearn.preprocessing**
- **StandardScaler**: Normaliza features restando la media y dividiendo por la desviación estándar
- **LabelEncoder**: Convierte variables categóricas en números (0, 1, 2, etc.)
- **OneHotEncoder**: Convierte variables categóricas en múltiples columnas binarias (dummy variables)

##### **sklearn.feature_selection**
- **SelectKBest**: Selecciona las k mejores features según un criterio
- **f_regression**: Método estadístico F-test para regresión (evalúa correlación lineal)
- **f_classif**: Método estadístico F-test para clasificación (evalúa separabilidad de clases)

##### **sklearn.decomposition**
- **PCA**: Análisis de Componentes Principales - reduce dimensionalidad manteniendo la mayor varianza

#### Librerías del Sistema
- **pickle**: Para serializar/deserializar objetos Python (guardar/cargar modelos)
- **os**: Para operaciones del sistema operativo (manejo de rutas)

### Métodos de la Clase FeatureEngineer

#### **extract_statistical_features()**
- Crea features de interacción (multiplicación y división entre variables numéricas)
- Genera features polinomiales (cuadrado, cubo) para variables clave
- Crea features binadas (agrupa valores en rangos)

#### **encode_categorical_features()**
- Convierte variables categóricas a numéricas usando LabelEncoder o OneHotEncoder
- Maneja valores faltantes en variables categóricas

#### **handle_missing_values()**
- Rellena valores faltantes usando media, mediana, moda o elimina filas
- Aplica estrategias diferentes para variables numéricas y categóricas

#### **select_features()**
- Selecciona las features más importantes usando correlación o tests estadísticos
- Reduce dimensionalidad manteniendo solo las k mejores features

#### **apply_pca()**
- Reduce dimensionalidad usando análisis de componentes principales
- Mantiene un porcentaje específico de varianza explicada

#### **scale_features()**
- Normaliza features para que tengan media 0 y desviación estándar 1
- Permite ajustar el scaler en training y aplicarlo en test

#### **create_poverty_target()**
- Crea variable objetivo binaria para clasificación de pobreza
- Usa umbrales de ingreso o múltiples factores

#### **save_preprocessing_components() / load_preprocessing_components()**
- Guarda/carga los componentes de preprocesamiento para reutilización
- Permite aplicar la misma transformación en datos nuevos

### Contexto del Proyecto de Pobreza

En este proyecto específico, la clase está "ingeniando" features para predecir pobreza:
- Combina variables de ingreso, edad, horas de trabajo
- Crea indicadores compuestos
- Transforma variables categóricas en numéricas
- Selecciona las características más relevantes

### Ejemplos de Feature Engineering Aplicado

```python
# Crear features de interacción
df_engineered[f'{col1}_x_{col2}'] = df[col1] * df[col2]

# Features polinomiales
df_engineered[f'{col}_squared'] = df[col] ** 2

# Binning (agrupación en rangos)
df_engineered['edad_bin'] = pd.cut(df['edad'], bins=5, 
                                  labels=['very_young', 'young', 'middle', 'senior', 'elderly'])
```

### Importancia en el Pipeline de ML

Este módulo es fundamental para preparar los datos antes del entrenamiento de modelos de machine learning, aplicando técnicas estándar de feature engineering y preprocesamiento que mejoran significativamente el rendimiento del modelo final. 