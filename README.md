# Sistema de Inteligencia Artificial Explicable para la Mejora en la Precisión de Proyecciones Financieras del Portafolio de Clientes de la Consultora

## Descripción
El desarrollo de este proyecto responde a la necesidad de mejorar la calidad, consistencia y confiabilidad de las proyecciones financieras en la consultora. Desde una perspectiva organizacional, la implementación de un sistema basado en inteligencia artificial permite optimizar los tiempos de análisis, reducir la dependencia del criterio humano y estandarizar los procesos de evaluación financiera.

Desde el punto de vista tecnológico, el proyecto integra modelos de machine learning con técnicas de inteligencia artificial explicable (XAI), lo cual es especialmente relevante en entornos financieros donde la transparencia y trazabilidad de los resultados son fundamentales.

En el ámbito académico, el proyecto contribuye al análisis y validación de modelos predictivos aplicados a datos financieros reales, permitiendo evaluar su desempeño y su capacidad de interpretación.

---

## 💡 Solución propuesta
Se implementó una solución que permite:

- Cargar y limpiar los datos históricos  
- Agrupar el saldo por año y período  
- Construir una variable temporal continua  
- Entrenar múltiples modelos predictivos  
- Generar proyecciones financieras  
- Exportar resultados y evidencias  

---

## Modelos utilizados
- Regresión Lineal  
- Random Forest  
- Red Neuronal (MLPRegressor)  

---

## Pipeline

### 1. Preprocesamiento de datos
- Lectura del archivo Excel  
- Limpieza de datos  
- Conversión de variables a formato numérico  
- Agrupación por año y período  

### 2. Entrenamiento de modelos
- Entrenamiento de modelos de regresión  
- Evaluación mediante métricas (MAE, RMSE, R²)  

### 3. Proyección
- Generación de períodos futuros  
- Predicción con cada modelo  
- Construcción del dataset de resultados  

### 4. Visualización
- Comparación entre datos reales y predicciones  
- Generación de gráficos  

## Requisitos técnicos
Python 3.10 o superior  

Librerías:
- pandas  
- numpy  
- matplotlib  
- scikit-learn  
- openpyxl  
- joblib  

---

## Instalación

```bash
pip install -r requirements.txt
```

## Instrucciones de ejecución

python scripts/preprocess.py
python scripts/modelos.py
python scripts/proyeccion.py
python scripts/grafico.py
