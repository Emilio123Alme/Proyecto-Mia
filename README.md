# Proyección del saldo financiero hasta 2035

## Descripción
Sistema de Inteligencia Artificial Explicable para la Mejora en la Precisión de Proyecciones Financieras del Portafolio de Clientes de la Consultora

## Modelos utilizados
- Regresión Lineal
- Random Forest
- Red Neuronal (MLPRegressor)

## Pipeline
1. Preprocesamiento de datos
2. Entrenamiento de modelos
3. Evaluación
4. Proyección futura

## Ejecución

```bash
python scripts/preprocess.py
python scripts/train.py
python scripts/forecast.py
python scripts/grafico.py
