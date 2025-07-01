# Sistema de Predicción de Pobreza Laboral

## ¿Qué estamos prediciendo?

El sistema está diseñado para **predecir el estado de pobreza** de las personas basándose en múltiples indicadores laborales y socioeconómicos. Específicamente:

### Variable Objetivo
- **Estado de Pobreza**: Clasificación binaria (0 = No Pobre, 1 = Pobre)
- **Probabilidad**: Probabilidad de estar en estado de pobreza (0-100%)
- **Confianza**: Nivel de confianza de la predicción del modelo

### Método de Clasificación
El sistema utiliza un enfoque **multi-factor** que considera:
1. **Ingresos laborales y per cápita**
2. **Condiciones de empleo**
3. **Nivel educativo**
4. **Edad y género**
5. **Sector económico**
6. **Horas de trabajo**

## Variables de Entrada (Columnas del Excel)

### Identificación
- `persona_key`: Identificador único de la persona
- `tiempo_id`: Período temporal (formato YYYYMM)
- `anio`: Año del análisis
- `mes`: Mes del análisis

### Demográficas
- `sexo`: Género (1 = Hombre, 2 = Mujer)
- `edad`: Edad de la persona (0-120 años)
- `estado_civil`: Estado civil (0-6)
- `nivel_instruccion`: Nivel educativo (0-5)

### Laborales
- `sector_id`: Sector económico (0-9)
- `condact_id`: Condición de actividad (0-9)
- `horas_trabajo_semana`: Horas trabajadas por semana (0-168)
- `desea_trabajar_mas`: Deseo de trabajar más horas (0-4)
- `disponible_trabajar_mas`: Disponibilidad para trabajar más (0-1)

### Económicas
- `ingreso_laboral`: Ingreso por trabajo (≥ 0)
- `ingreso_per_capita`: Ingreso per cápita del hogar (≥ 0)

### Geográficas
- `ciudad_id`: Identificador de la ciudad

## Mapeo de Categorías

### Género
- 1 = Hombre
- 2 = Mujer

### Nivel Educativo
- 0 = Sin educación
- 1 = Primaria
- 2 = Secundaria
- 3 = Bachillerato
- 4 = Superior
- 5 = Postgrado

### Estado Civil
- 0 = Soltero
- 1 = Casado
- 2 = Unión libre
- 3 = Divorciado
- 4 = Viudo
- 5 = Separado
- 6 = Otro

### Sector Económico
- 0 = Sin sector
- 1 = Agricultura
- 2 = Industria
- 3 = Construcción
- 4 = Comercio
- 5 = Servicios
- 6 = Transporte
- 7 = Finanzas
- 8 = Administración
- 9 = Otros

### Condición de Actividad
- 0 = Ocupado
- 1 = Desempleado
- 2 = Inactivo
- 3 = Jubilado
- 4 = Estudiante
- 5 = Ama de casa
- 6 = Incapacitado
- 7 = Otro inactivo
- 8 = Trabajador familiar
- 9 = Sin información

## Variables de Salida (Resultados)

### Predicciones Principales
- `prediccion_pobreza`: Clasificación (0/1)
- `estado_pobreza`: Etiqueta ("Pobre"/"No Pobre")
- `probabilidad_pobreza`: Probabilidad (0-1)
- `confianza`: Nivel de confianza (0-1)

### Análisis Demográfico
- `genero`: Género (Hombre/Mujer)
- `nivel_educativo`: Nivel educativo descriptivo
- `estado_civil_desc`: Estado civil descriptivo

### Análisis Laboral
- `sector_economico`: Sector económico descriptivo
- `condicion_actividad`: Condición de actividad descriptiva
- `intensidad_laboral`: Intensidad de trabajo (Sin trabajo/Parcial/Tiempo completo/Extendido/Excesivo)

### Análisis de Ingresos
- `categoria_ingreso`: Categoría de ingreso laboral (Sin ingreso/Bajo/Medio-bajo/Medio/Alto)
- `categoria_ingreso_per_capita`: Categoría de ingreso per cápita (Muy bajo/Bajo/Medio/Medio-alto/Alto)

### Indicadores de Subempleo
- `subempleo_por_horas`: Subempleo por horas insuficientes (0/1)
- `subempleo_por_ingresos`: Subempleo por ingresos bajos (0/1)
- `indicador_subempleo`: Indicador general de subempleo (0/1)

### Análisis de Riesgo
- `factores_riesgo`: Número de factores de riesgo (0-5)
- `nivel_riesgo`: Nivel de riesgo (Sin riesgo/Bajo/Medio/Alto/Muy alto)

## Factores de Riesgo Considerados

El sistema evalúa los siguientes factores de riesgo:

1. **Ingreso laboral bajo** (< $300)
2. **Ingreso per cápita bajo** (< $200)
3. **Bajo nivel educativo** (< Bachillerato)
4. **Edad avanzada** (> 65 años)
5. **Desempleo** (condact_id = 1)

## Indicadores de Subempleo

### Subempleo por Horas
- Trabaja menos de 40 horas por semana
- Desea trabajar más horas
- Está disponible para trabajar más

### Subempleo por Ingresos
- Tiene ingresos laborales
- Ingresos menores a $500

## Modelos Disponibles

1. **Red Neuronal**: Análisis avanzado de patrones complejos
2. **Modelo Lineal**: Análisis rápido e interpretable
3. **Modelo Logístico**: Clasificación binaria interpretable

## Interpretación de Resultados

### Niveles de Confianza
- **Alto** (≥80%): Predicción muy confiable
- **Medio** (60-79%): Predicción moderadamente confiable
- **Bajo** (<60%): Predicción menos confiable

### Niveles de Riesgo
- **Sin riesgo** (0 factores): Baja probabilidad de pobreza
- **Bajo** (1 factor): Riesgo moderado
- **Medio** (2 factores): Riesgo significativo
- **Alto** (3 factores): Alto riesgo
- **Muy alto** (4-5 factores): Riesgo muy alto

## Contexto de Aplicación

Este sistema está diseñado para:
- **Análisis de políticas públicas** de empleo y pobreza
- **Identificación de grupos vulnerables** para programas sociales
- **Evaluación de impacto** de intervenciones laborales
- **Planificación estratégica** de desarrollo económico

## Limitaciones

- Los modelos se basan en datos históricos y patrones observados
- Las predicciones son probabilísticas, no determinísticas
- Se requiere actualización regular de los modelos con nuevos datos
- El contexto económico puede cambiar los patrones de pobreza 