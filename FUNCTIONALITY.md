# Función del método create_poverty_target

---

**User**

el metodo de create_poverty_target de que se encarga?
 
en que parte se define el indicador de 1 o 0 de si una persona puede es pobre o no?



---

**Cursor**

I'll analyze the `create_poverty_target` method to explain its purpose and how it defines the poverty indicator.


## Análisis del método `create_poverty_target`

El método `create_poverty_target` se encarga de **crear la variable objetivo binaria (target) para el modelo de machine learning**, transformando los datos socioeconómicos en un indicador de pobreza (1 = pobre, 0 = no pobre).

### Responsabilidades principales:

1. **Crear variable objetivo binaria** para entrenar modelos de clasificación
2. **Definir criterios de pobreza** usando diferentes métodos
3. **Manejar datos faltantes** de manera robusta
4. **Validar la distribución** del target creado

### Definición del indicador 1/0:

El método define la pobreza en **dos formas diferentes** según el parámetro `method`:

#### 1. **Método `income_threshold` (por defecto):**
```python
# Define poverty threshold (below median)
threshold = income_data.median()
poverty_target = (df['ingreso_per_capita'] < threshold).astype(int)
```

**Criterio:** Una persona es pobre (1) si su ingreso per cápita está **por debajo de la mediana** de la población.

#### 2. **Método `multi_factor`:**
```python
# Income factor
income_factor = (df['ingreso_per_capita'] < income_data.median()).astype(int)

# Age factor (elderly)
age_factor = (df['edad'] > 65).astype(int)

# Work hours factor (low work hours)
work_factor = (df['horas_trabajo_semana'] < 20).astype(int)

# Employment condition factor (underemployment)
underemployment_conditions = [2, 3, 1]  # Subempleado, Ocupado parcial, Desempleado
condition_factor = df['condact_id'].isin(underemployment_conditions).astype(int)

# Suma todos los factores y crea un target binario (al menos 1 factor indica pobreza)
poverty_target = pd.concat(factors, axis=1).sum(axis=1)
poverty_target = (poverty_target >= 1).astype(int)
```

**Criterio:** Una persona es pobre (1) si cumple **al menos 1 de estos factores**:
- Ingreso per cápita bajo (por debajo de la mediana)
- Edad mayor a 65 años
- Menos de 20 horas de trabajo por semana
- Condición de empleo desfavorable (subempleado, ocupado parcial, desempleado)

### Lógica de asignación:
- **1 (Pobre):** Cumple los criterios de pobreza definidos
- **0 (No pobre):** No cumple los criterios de pobreza definidos

El método también incluye manejo robusto de datos faltantes, llenando valores NaN con 0 (asumiendo que no están en pobreza) y proporcionando estadísticas de la distribución final del target.

```python
# Define poverty threshold (below median)
threshold = income_data.median()
poverty_target = (df['ingreso_per_capita'] < threshold).astype(int)
```

```python
# Income factor
income_factor = (df['ingreso_per_capita'] < income_data.median()).astype(int)

# Age factor (elderly)
age_factor = (df['edad'] > 65).astype(int)

# Work hours factor (low work hours)
work_factor = (df['horas_trabajo_semana'] < 20).astype(int)

# Employment condition factor (underemployment)
underemployment_conditions = [2, 3, 1]  # Subempleado, Ocupado parcial, Desempleado
condition_factor = df['condact_id'].isin(underemployment_conditions).astype(int)

# Suma todos los factores y crea un target binario (al menos 1 factor indica pobreza)
poverty_target = pd.concat(factors, axis=1).sum(axis=1)
poverty_target = (poverty_target >= 1).astype(int)
```

---

**User**

hmmm, como es el multi factor?

debe cumplir al menos 1?
y si cumple todos? como se comporta esto?

---

**Cursor**

Analicemos el comportamiento del método `multi_factor` en detalle:


## Análisis del método `multi_factor`

Hay una **inconsistencia** en el código. Veamos el comportamiento actual:

### Código actual (líneas 380-382):
```python
# Suma todos los factores y crea un target binario (al menos 2 factores indican pobreza)
poverty_target = pd.concat(factors, axis=1).sum(axis=1)
poverty_target = (poverty_target >= 1).astype(int)
```

### Comportamiento actual:

**El comentario dice "al menos 2 factores" pero el código implementa "al menos 1 factor"**

- **Si cumple 0 factores:** `suma = 0` → `0 >= 1` = `False` → **0 (No pobre)**
- **Si cumple 1 factor:** `suma = 1` → `1 >= 1` = `True` → **1 (Pobre)**
- **Si cumple 2 factores:** `suma = 2` → `2 >= 1` = `True` → **1 (Pobre)**
- **Si cumple 3 factores:** `suma = 3` → `3 >= 1` = `True` → **1 (Pobre)**
- **Si cumple 4 factores:** `suma = 4` → `4 >= 1` = `True` → **1 (Pobre)**

### Los 4 factores que se evalúan:

1. **Factor de ingreso:** `ingreso_per_capita < mediana` → 1 o 0
2. **Factor de edad:** `edad > 65` → 1 o 0  
3. **Factor de horas de trabajo:** `horas_trabajo_semana < 20` → 1 o 0
4. **Factor de condición laboral:** `condact_id in [2,3,1]` → 1 o 0

### Problema identificado:

El comentario en la línea 380 dice **"al menos 2 factores indican pobreza"** pero el código usa `>= 1`, lo que significa que **con cumplir 1 factor ya se considera pobre**.

### Posibles interpretaciones:

1. **Implementación actual (>= 1):** Cualquier factor de riesgo hace que la persona sea considerada pobre
2. **Comentario sugerido (>= 2):** Se necesitan al menos 2 factores de riesgo para considerar pobreza

### Ejemplo práctico:

Una persona con:
- Ingreso bajo: ✅ (1 punto)
- Edad 30 años: ❌ (0 puntos)  
- 40 horas de trabajo: ❌ (0 puntos)
- Empleado formal: ❌ (0 puntos)
- **Total: 1 punto** → **Resultado: POBRE (1)**

¿Quieres que corrija esta inconsistencia para que coincida con el comentario (requerir al menos 2 factores) o prefieres mantener la implementación actual (al menos 1 factor)?

```python
# Suma todos los factores y crea un target binario (al menos 2 factores indican pobreza)
poverty_target = pd.concat(factors, axis=1).sum(axis=1)
poverty_target = (poverty_target >= 1).astype(int)
```

---

**User**

quiero tener al menos 2 factores requeridos\

---

**Cursor**

Perfecto, voy a corregir la inconsistencia para que requiera al menos 2 factores como indica el comentario.


```python
            if factors:
                # Suma todos los factores y crea un target binario (al menos 2 factores indican pobreza)
                poverty_target = pd.concat(factors, axis=1).sum(axis=1)
                poverty_target = (poverty_target >= 1).astype(int)

->

            if factors:
                # Suma todos los factores y crea un target binario (al menos 2 factores indican pobreza)
                poverty_target = pd.concat(factors, axis=1).sum(axis=1)
                poverty_target = (poverty_target >= 2).astype(int)
```

Perfecto! He corregido la inconsistencia. Ahora el método `multi_factor` funciona correctamente:

## Comportamiento corregido:

### Criterio actualizado:
```python
poverty_target = (poverty_target >= 2).astype(int)
```

### Comportamiento con al menos 2 factores requeridos:

- **Si cumple 0 factores:** `suma = 0` → `0 >= 2` = `False` → **0 (No pobre)**
- **Si cumple 1 factor:** `suma = 1` → `1 >= 2` = `False` → **0 (No pobre)**
- **Si cumple 2 factores:** `suma = 2` → `2 >= 2` = `True` → **1 (Pobre)**
- **Si cumple 3 factores:** `suma = 3` → `3 >= 2` = `True` → **1 (Pobre)**
- **Si cumple 4 factores:** `suma = 4` → `4 >= 2` = `True` → **1 (Pobre)**

### Ejemplo práctico:

**Persona A (NO POBRE - 1 factor):**
- Ingreso bajo: ✅ (1 punto)
- Edad 30 años: ❌ (0 puntos)  
- 40 horas de trabajo: ❌ (0 puntos)
- Empleado formal: ❌ (0 puntos)
- **Total: 1 punto** → **Resultado: NO POBRE (0)**

**Persona B (POBRE - 2 factores):**
- Ingreso bajo: ✅ (1 punto)
- Edad 70 años: ✅ (1 punto)
- 40 horas de trabajo: ❌ (0 puntos)
- Empleado formal: ❌ (0 puntos)
- **Total: 2 puntos** → **Resultado: POBRE (1)**

**Persona C (POBRE - 3 factores):**
- Ingreso bajo: ✅ (1 punto)
- Edad 70 años: ✅ (1 punto)
- 15 horas de trabajo: ✅ (1 punto)
- Empleado formal: ❌ (0 puntos)
- **Total: 3 puntos** → **Resultado: POBRE (1)**

Ahora el método es más estricto y requiere evidencia de múltiples factores de riesgo para clasificar a alguien como pobre, lo que debería resultar en una clasificación más precisa y conservadora.

```python
poverty_target = (poverty_target >= 2).astype(int)
```

---

**User**

explicame como funciona grid search en este contexto

---

**Cursor**

Te explico cómo funciona Grid Search en este contexto específico de regresión logística para clasificación de pobreza:

## ¿Qué es Grid Search?

Grid Search es una técnica de **optimización de hiperparámetros** que prueba sistemáticamente todas las combinaciones posibles de parámetros para encontrar la mejor configuración del modelo.

## Contexto en el código:

```python
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
```

### Parámetros que se optimizan:

1. **`C` (Regularización):**
   - Controla la fuerza de la regularización
   - Valores más pequeños = más regularización (modelo más simple)
   - Valores más grandes = menos regularización (modelo más complejo)
   - Rango: `[0.001, 0.01, 0.1, 1, 10, 100]`

2. **`penalty` (Tipo de regularización):**
   - `'l1'` (Lasso): Reduce algunos coeficientes a exactamente 0 (selección de características)
   - `'l2'` (Ridge): Reduce coeficientes pero no los hace exactamente 0

3. **`solver` (Algoritmo de optimización):**
   - `'liblinear'`: Eficiente para datasets pequeños
   - `'saga'`: Eficiente para datasets grandes

## Proceso de Grid Search:

### 1. **Combinaciones totales:**
```
6 valores de C × 2 penalties × 2 solvers = 24 combinaciones
```

### 2. **Validación cruzada (CV=5):**
Para cada combinación:
- Divide los datos en 5 folds
- Entrena en 4 folds, evalúa en 1 fold
- Repite 5 veces (cada fold como test una vez)
- Promedia los 5 scores

### 3. **Evaluación:**
```python
scoring='accuracy'  # Métrica de evaluación
```

## Ejemplo práctico:

### Iteración 1:
```python
C=0.001, penalty='l1', solver='liblinear'
# Entrena modelo → CV accuracy = 0.72
```

### Iteración 2:
```python
C=0.001, penalty='l1', solver='saga'
# Entrena modelo → CV accuracy = 0.73
```

### Iteración 3:
```python
C=0.001, penalty='l2', solver='liblinear'
# Entrena modelo → CV accuracy = 0.71
```

### ... (continúa para las 24 combinaciones)

### Resultado final:
```python
Best parameters: {'C': 1, 'penalty': 'l2', 'solver': 'liblinear'}
Best CV score: 0.85
```

## Ventajas en el contexto de pobreza:

1. **Robustez:** Encuentra la configuración más estable para clasificar pobreza
2. **Generalización:** Evita overfitting mediante validación cruzada
3. **Automatización:** No requiere ajuste manual de parámetros
4. **Reproducibilidad:** Mismo proceso para diferentes datasets

## Consideraciones:

- **Tiempo:** 24 combinaciones × 5 CV = 120 entrenamientos
- **Recursos:** `n_jobs=-1` usa todos los cores disponibles
- **Métrica:** `accuracy` es apropiada para clasificación binaria equilibrada

El Grid Search asegura que el modelo de regresión logística esté optimizado para clasificar correctamente si una persona está en pobreza o no, basándose en múltiples factores socioeconómicos.

```python
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
```

```plaintext
6 valores de C × 2 penalties × 2 solvers = 24 combinaciones
```

```python
scoring='accuracy'  # Métrica de evaluación
```

```python
C=0.001, penalty='l1', solver='liblinear'
# Entrena modelo → CV accuracy = 0.72
```

```python
C=0.001, penalty='l1', solver='saga'
# Entrena modelo → CV accuracy = 0.73
```

```python
C=0.001, penalty='l2', solver='liblinear'
# Entrena modelo → CV accuracy = 0.71
```

```python
Best parameters: {'C': 1, 'penalty': 'l2', 'solver': 'liblinear'}
Best CV score: 0.85
```

---

**User**

dime como funciona el metoodo de guardar modelo
como es el archivo pkl

---

**Cursor**

Te explico cómo funciona el método de guardar modelo y la estructura del archivo `.pkl`:

## Método `save_model`:

```python
def save_model(self, model_name: str, filepath: str):
    """Save a trained model to file"""
    if model_name not in self.models:
        raise ValueError(f"Model {model_name} not found")
    
    model_data = {
        'model': self.models[model_name],
        'feature_importance': self.feature_importance.get(model_name),
        'best_params': self.best_params.get(model_name)
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model {model_name} saved to {filepath}")
```

## Proceso de guardado:

### 1. **Validación:**
```python
if model_name not in self.models:
    raise ValueError(f"Model {model_name} not found")
```
Verifica que el modelo existe antes de intentar guardarlo.

### 2. **Estructura de datos:**
```python
model_data = {
    'model': self.models[model_name],           # Modelo entrenado
    'feature_importance': self.feature_importance.get(model_name),  # Importancia de features
    'best_params': self.best_params.get(model_name)                 # Mejores hiperparámetros
}
```

### 3. **Serialización con pickle:**
```python
with open(filepath, 'wb') as f:  # 'wb' = write binary
    pickle.dump(model_data, f)
```

## Estructura del archivo `.pkl`:

### **Contenido del archivo:**
```python
{
    'model': LogisticRegression(C=1.0, penalty='l2', solver='liblinear', ...),
    'feature_importance': {
        'edad': 0.15,
        'ingreso_laboral': 0.45,
        'horas_trabajo_semana': 0.12,
        'condact_id': 0.28
    },
    'best_params': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear'
    }
}
```

### **Componentes guardados:**

1. **`model`:** El modelo entrenado completo con:
   - Coeficientes aprendidos
   - Intercepto
   - Hiperparámetros finales
   - Estado interno del modelo

2. **`feature_importance`:** Importancia de cada característica:
   - Valores entre 0 y 1
   - Suma total = 1.0
   - Indica qué variables son más importantes para predecir pobreza

3. **`best_params`:** Mejores hiperparámetros encontrados por Grid Search:
   - Configuración óptima
   - Necesario para reproducir el modelo

## Ejemplo de uso:

### **Guardado:**
```python
linear_models.save_model('logistic_regression', 'models/logistic_regression_20241201.pkl')
```

### **Carga posterior:**
```python
with open('models/logistic_regression_20241201.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_importance = model_data['feature_importance']
best_params = model_data['best_params']
```

## Ventajas del formato `.pkl`:

1. **Preservación completa:** Guarda el estado exacto del modelo
2. **Eficiencia:** Formato binario compacto
3. **Compatibilidad:** Funciona con cualquier objeto Python
4. **Metadatos:** Incluye información adicional útil

## Consideraciones de seguridad:

- **Solo cargar archivos de fuentes confiables**
- **Los archivos `.pkl` pueden ejecutar código malicioso**
- **Usar `joblib` para modelos scikit-learn (más seguro)**

## Estructura de archivos típica:

```
ml_results/
├── models/
│   ├── logistic_regression_20241201_143022.pkl
│   ├── linear_regression_20241201_143022.pkl
│   └── preprocessing_20241201_143022.pkl
├── plots/
└── reports/
```

El archivo `.pkl` contiene todo lo necesario para reproducir exactamente el mismo modelo y hacer predicciones consistentes sobre pobreza.

```python
def save_model(self, model_name: str, filepath: str):
    """Save a trained model to file"""
    if model_name not in self.models:
        raise ValueError(f"Model {model_name} not found")
    
    model_data = {
        'model': self.models[model_name],
        'feature_importance': self.feature_importance.get(model_name),
        'best_params': self.best_params.get(model_name)
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model {model_name} saved to {filepath}")
```

```python
if model_name not in self.models:
    raise ValueError(f"Model {model_name} not found")
```

```python
model_data = {
    'model': self.models[model_name],           # Modelo entrenado
    'feature_importance': self.feature_importance.get(model_name),  # Importancia de features
    'best_params': self.best_params.get(model_name)                 # Mejores hiperparámetros
}
```

```python
with open(filepath, 'wb') as f:  # 'wb' = write binary
    pickle.dump(model_data, f)
```

```python
{
    'model': LogisticRegression(C=1.0, penalty='l2', solver='liblinear', ...),
    'feature_importance': {
        'edad': 0.15,
        'ingreso_laboral': 0.45,
        'horas_trabajo_semana': 0.12,
        'condact_id': 0.28
    },
    'best_params': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear'
    }
}
```

```python
linear_models.save_model('logistic_regression', 'models/logistic_regression_20241201.pkl')
```

```python
with open('models/logistic_regression_20241201.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_importance = model_data['feature_importance']
best_params = model_data['best_params']
```

```plaintext
ml_results/
├── models/
│   ├── logistic_regression_20241201_143022.pkl
│   ├── linear_regression_20241201_143022.pkl
│   └── preprocessing_20241201_143022.pkl
├── plots/
└── reports/
```

