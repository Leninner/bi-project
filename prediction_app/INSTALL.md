# Instrucciones de Instalación y Uso

## Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Navegador web moderno

## Instalación

### 1. Clonar o descargar el proyecto

```bash
# Si tienes git
git clone <url-del-repositorio>
cd prediction_app

# O descargar y extraer el archivo ZIP
```

### 2. Crear entorno virtual (recomendado)

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate

# En macOS/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalación

```bash
python test_app.py
```

Si todo está correcto, verás un mensaje "ALL TESTS PASSED! ✅"

## Uso de la Aplicación

### 1. Iniciar la aplicación

```bash
python app.py
```

La aplicación se iniciará en `http://localhost:5000`

### 2. Acceder a la interfaz web

Abre tu navegador y ve a: `http://localhost:5000`

### 3. Usar la aplicación

1. **Cargar archivo Excel**: 
   - Arrastra y suelta un archivo Excel o haz clic para seleccionar
   - El archivo debe tener la estructura especificada

2. **Seleccionar modelo**:
   - **Red Neuronal**: Mayor precisión, más complejo
   - **Modelo Lineal**: Más rápido, interpretable

3. **Validar archivo** (opcional):
   - Haz clic en "Validar Archivo" para verificar la estructura

4. **Realizar predicción**:
   - Haz clic en "Realizar Predicción"
   - Espera a que se procese el archivo

5. **Ver resultados**:
   - Los resultados se mostrarán en una tabla
   - Puedes exportar a Excel o CSV

## Estructura del Archivo Excel

Tu archivo Excel debe contener las siguientes columnas:

| Columna                 | Tipo    | Rango     | Descripción                       |
| ----------------------- | ------- | --------- | --------------------------------- |
| persona_key             | Entero  | ID único  | Identificador único de la persona |
| tiempo_id               | Entero  | YYYYMM    | ID del tiempo (ej: 202505)        |
| anio                    | Entero  | 2000-2030 | Año                               |
| mes                     | Entero  | 1-12      | Mes                               |
| sector_id               | Entero  | 0-9       | ID del sector laboral             |
| condact_id              | Entero  | 0-9       | ID de condición de actividad      |
| sexo                    | Entero  | 1-2       | Sexo (1=Hombre, 2=Mujer)          |
| ciudad_id               | Entero  | ID ciudad | ID de la ciudad                   |
| nivel_instruccion       | Entero  | 0-5       | Nivel de instrucción              |
| estado_civil            | Entero  | 0-6       | Estado civil                      |
| edad                    | Entero  | 0-120     | Edad en años                      |
| ingreso_laboral         | Decimal | ≥ 0       | Ingreso laboral                   |
| ingreso_per_capita      | Decimal | ≥ 0       | Ingreso per cápita                |
| horas_trabajo_semana    | Entero  | 0-168     | Horas de trabajo por semana       |
| desea_trabajar_mas      | Entero  | 0-4       | Desea trabajar más                |
| disponible_trabajar_mas | Entero  | 0-1       | Disponible para trabajar más      |

## Descargar Plantilla

Puedes descargar una plantilla de ejemplo haciendo clic en "Descargar Plantilla" en la interfaz web.

## Solución de Problemas

### Error: "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### Error: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Error: "No module named 'openpyxl'"
```bash
pip install openpyxl
```

### Error: "Model not found"
- La aplicación creará modelos dummy automáticamente
- Para usar modelos reales, colócalos en la carpeta `../models/`

### Error: "File too large"
- El tamaño máximo del archivo es 16MB
- Divide tu archivo en partes más pequeñas si es necesario

### Error: "Invalid file type"
- Solo se aceptan archivos Excel (.xlsx o .xls)
- Asegúrate de que el archivo tenga la extensión correcta

## Características de la Aplicación

### ✅ Funcionalidades Implementadas

- **Carga de archivos Excel**: Drag & drop o selección manual
- **Validación de datos**: Verificación automática de estructura y rangos
- **Selección de modelo**: Lineal o Neural
- **Predicción en tiempo real**: Procesamiento automático de datos
- **Visualización de resultados**: Tabla con predicciones y confianza
- **Exportación**: Excel y CSV
- **Interfaz moderna**: Diseño responsive y amigable
- **Manejo de errores**: Mensajes claros y útiles

### 📊 Resultados de Predicción

La aplicación proporciona:

- **Predicción binaria**: Informal (1) o Formal (0)
- **Probabilidad**: Porcentaje de confianza en la predicción
- **Resumen estadístico**: Distribución de predicciones

### 🔧 Configuración Avanzada

#### Cambiar puerto de la aplicación
Edita `app.py` línea 255:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

#### Cambiar tamaño máximo de archivo
Edita `app.py` línea 18:
```python
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
```

#### Usar modelos personalizados
Coloca tus modelos entrenados en la carpeta `../models/`:
- Modelos neurales: archivos `.h5` o `.hdf5`
- Modelos lineales: archivos `.pkl`

## Notas Técnicas

- La aplicación usa Flask como framework web
- Los modelos se cargan dinámicamente desde la carpeta `models/`
- El preprocesamiento es idéntico al usado en el entrenamiento
- Los archivos se procesan temporalmente y se eliminan automáticamente
- La interfaz es completamente responsive y funciona en móviles 