# ==========================================================================
# 12. GENERACIÓN DE PROYECCIONES
# ==========================================================================

df_resultados = X_test.copy()
df_resultados['Saldo_Real'] = y_test.values
df_resultados['Saldo_Proyectado'] = resultados["Random Forest"]["pred"]
df_resultados['Error'] = df_resultados['Saldo_Real'] - df_resultados['Saldo_Proyectado']

cols = ['Saldo_Real', 'Saldo_Proyectado', 'Error'] + [c for c in df_resultados.columns if c not in ['Saldo_Real', 'Saldo_Proyectado', 'Error']]
df_resultados = df_resultados[cols]

df_resultados.to_csv("proyeccion_costos.csv", index=False)

print("Archivo generado: proyeccion_costos.csv")
print(df_resultados.head())

# ==========================================================================
# 13. PROYECCIÓN FUTURA 2026
# ==========================================================================

df_future = df.copy()
last_year = df_future['Year'].max()
df_2025 = df_future[df_future['Year'] == last_year].copy()
future_rows = []

for acc in df_2025['Account'].unique():
    df_acc = df_2025[df_2025['Account'] == acc].sort_values('Period')
    last_values = df_acc.tail(2)  # para Lag_1 y Lag_2

    for period in range(1, 13):
        row = {}
        row['Account'] = acc
        row['Year'] = 2026
        row['Period'] = period
        row['Year_Period'] = 2026 * 100 + period

if len(last_values) >= 2:
    row['Lag_1'] = last_values.iloc[-1]['Costo']
    row['Lag_2'] = last_values.iloc[-2]['Costo']
    row['Rolling_mean_3'] = last_values['Costo'].mean()

elif len(last_values) == 1:
    row['Lag_1'] = last_values.iloc[-1]['Costo']
    row['Lag_2'] = last_values.iloc[-1]['Costo'] 
    row['Rolling_mean_3'] = last_values.iloc[-1]['Costo']

else:
    row['Lag_1'] = 0
    row['Lag_2'] = 0
    row['Rolling_mean_3'] = 0

row['Vendor Name'] = df_acc.iloc[-1]['Vendor Name']
row['Location'] = df_acc.iloc[-1]['Location']

future_rows.append(row)

df_2026 = pd.DataFrame(future_rows)

# ==========================================================================
# PREPARAR COMO TRAIN
# ==========================================================================

df_2026_model = pd.get_dummies(df_2026, drop_first=True)
df_2026_model = df_2026_model.reindex(columns=X_train.columns, fill_value=0)

# ==========================================================================
# PREDICCIÓN
# ==========================================================================

pred_2026 = resultados["Random Forest"]["modelo"].predict(df_2026_model)
df_2026['Saldo_Proyectado'] = pred_2026

# ==========================================================================
# EXPORTAR
# ==========================================================================

df_2026.to_csv("proyeccion_2026.csv", index=False)

print("Proyección 2026 generada")
print(df_2026.head())

# ==========================================================================
# PROYECCIÓN / EVALUACIÓN FINAL CORRECTA
# ==========================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================================
# 1. ASEGURAR FECHA
# ==========================================================================

df['GL Date'] = pd.to_numeric(df['GL Date'], errors='coerce')
df['fecha'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df['GL Date'], unit='D')

# ==========================================================================
# 2. ORDENAR
# ==========================================================================

df = df.sort_values('fecha').reset_index(drop=True)

# ==========================================================================
# 3. SPLIT CORRECTO
# ==========================================================================

split = int(len(df) * 0.8)

df_train = df.iloc[:split].copy()
df_test = df.iloc[split:].copy()

# ==========================================================================
# 4. VARIABLES TRAIN
# ==========================================================================

X_train = df_train.select_dtypes(include=['number']).copy()
X_train = X_train.drop(columns=['Costo'], errors='ignore')

y_train = df_train['Costo']

X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_train = X_train.fillna(0)

# ==========================================================================
# 5. VARIABLES TEST
# ==========================================================================

X_test = df_test.select_dtypes(include=['number']).copy()
X_test = X_test.drop(columns=['Costo'], errors='ignore')

X_test = X_test.apply(pd.to_numeric, errors='coerce')
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# ==========================================================================
# 6. MODELOS
# ==========================================================================

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

modelo_lr = LinearRegression()
modelo_tree = DecisionTreeRegressor(max_depth=10, random_state=42)
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)

modelo_lr.fit(X_train, y_train)
modelo_tree.fit(X_train, y_train)
modelo_rf.fit(X_train, y_train)

# ==========================================================================
# 7. PREDICCIONES CORRECTAS
# ==========================================================================

y_pred_lr = modelo_lr.predict(X_test)
y_pred_tree = modelo_tree.predict(X_test)
y_pred_rf = modelo_rf.predict(X_test)

# ==========================================================================
# 8. DATAFRAME FINAL
# ==========================================================================

df_plot = df_test.copy().reset_index(drop=True)

df_plot['lr'] = y_pred_lr
df_plot['tree'] = y_pred_tree
df_plot['rf'] = y_pred_rf

# ==========================================================================
# 9. EXPORTAR PARA FRONT
# ==========================================================================

df_plot.to_csv("df_modelos.csv", index=False)

# ==========================================================================
# 10. GRÁFICA CLARA
# ==========================================================================

df_plot = df_plot.tail(200)

plt.figure(figsize=(12,6))

plt.plot(df_plot['Costo'].values, label='Real', linewidth=2)
plt.plot(df_plot['lr'].values, label='LR', linestyle='--')
plt.plot(df_plot['tree'].values, label='Tree', linestyle=':')
plt.plot(df_plot['rf'].values, label='RF', linestyle='-.')

plt.title("Comparación de Modelos (Real vs Predicho)")
plt.legend()
plt.show()

# ==========================================================================
# 10. TABLA COMPARATIVA
# ==========================================================================

import pandas as pd

tabla = pd.DataFrame({
    "Modelo": ["Regresión Lineal", "Árbol de Decisión", "Random Forest"],
    "MAE": [957.60, 253.55, 130.61],
    "Mejora MAE": ["—", "73.5% mejor", "48.5% mejor vs Árbol"],
    "RMSE": [3071.91, 2235.34, 1531.79],
    "Mejora RMSE": ["—", "27.2% mejor", "31.5% mejor vs Árbol"],
    "R2": [0.139, 0.544, 0.7859],
    "Mejora R2": ["—", "+291%", "+44.5%"]
})

tabla

# ==========================================================================
# GRÁFICA DE MÉTRICAS
# ==========================================================================

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10,6))
x = np.arange(len(tabla["Modelo"]))
width = 0.25

tabla["R2_%"] = tabla["R2"] * 100

color_mae = "#1f77b4" 
color_rmse = "#ff7f0e" 
color_r2 = "#2ca02c" 

ax.bar(x - width, tabla["MAE"], width, label='MAE', color=color_mae)
ax.bar(x, tabla["RMSE"], width, label='RMSE', color=color_rmse)
ax.bar(x + width, tabla["R2_%"], width, label='R2 (%)', color=color_r2)

ax.set_xticks(x)
ax.set_xticklabels(tabla["Modelo"], rotation=15)
ax.set_ylabel("Valor de Métrica")
ax.grid(axis='y', linestyle='--', alpha=0.3)

ax.legend()
plt.title("Comparación de Métricas (Escalado)")
plt.tight_layout()
plt.show()

tabla.style.highlight_max(axis=0, subset=["R2"]).highlight_min(axis=0, subset=["MAE", "RMSE"])

# ==========================================================================
# GRAFICAS
# ==========================================================================

import matplotlib.pyplot as plt
import pandas as pd

modelos = ["Regresión Lineal", "Árbol", "Random Forest"]
mae = [957.60, 253.55, 130.61]
rmse = [3071.91, 2235.34, 1531.79]
r2 = [0.139, 0.544, 0.7859]

df_metricas = pd.DataFrame({
    "Modelo": modelos,
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
})

plt.figure()
plt.bar(df_metricas["Modelo"], df_metricas["MAE"])
plt.title("Comparación de MAE (Error Absoluto Medio)")
plt.xlabel("Modelo")
plt.ylabel("MAE")
plt.xticks(rotation=20)
plt.show()

plt.figure()
plt.bar(df_metricas["Modelo"], df_metricas["RMSE"])
plt.title("Comparación de RMSE (Error Cuadrático Medio)")
plt.xlabel("Modelo")
plt.ylabel("RMSE")
plt.xticks(rotation=20)
plt.show()

plt.figure()
plt.bar(df_metricas["Modelo"], df_metricas["R2"])
plt.title("Comparación de R² (Coeficiente de Determinación)")
plt.xlabel("Modelo")
plt.ylabel("R²")
plt.xticks(rotation=20)
plt.show()

# ==========================================================================
# EXPORTACIÓN REAL
# ==========================================================================

df_export = pd.DataFrame()

df_export['Costo'] = y_test.values

# ==========================================================================
# PREDICCIONES CORRECTAS
# ==========================================================================

df_export['lr'] = resultados["Regresión Lineal"]["pred"]
df_export['tree'] = resultados["Árbol de Decisión"]["pred"]
df_export['rf'] = resultados["Random Forest"]["pred"]

# ==========================================================================
#  VALIDACIÓN
# ==========================================================================

print(df_export.head())
print(len(df_export))

# ==========================================================================
#  EXPORTAR
# ==========================================================================

df_export.to_csv("df_modelos.csv", index=False)

print("CSV correcto para frontend")

# ==========================================================================
# PROYECCIÓN 2026
# ==========================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ==========================================================================
# 1. ORDENAR DATOS
# ==========================================================================

df_hist = df.copy().sort_values('Year').reset_index(drop=True)

# ==========================================================================
# 2. VARIABLE TIEMPO
# ==========================================================================

df_hist['t'] = np.arange(len(df_hist))

# ==========================================================================
# 3. MODELO DE TENDENCIA
# ==========================================================================

modelo_tendencia = LinearRegression()
modelo_tendencia.fit(df_hist[['t']], df_hist['Costo'])

# ==========================================================================
# 4. CREAR FUTURO
# ==========================================================================

num_futuro = 50
t_futuro = np.arange(len(df_hist), len(df_hist) + num_futuro)

df_futuro = pd.DataFrame({'t': t_futuro})

# ==========================================================================
# 5. PREDECIR
# ==========================================================================

df_futuro['Costo_Proyectado'] = modelo_tendencia.predict(df_futuro[['t']])

# ==========================================================================
# 6. AÑADIR AÑO
# ==========================================================================

df_futuro['Year'] = df_hist['Year'].max() + 1

# ==========================================================================
# 7. EXPORTAR
# ==========================================================================

df_proyeccion = df_futuro[['Year','Costo_Proyectado']]
df_proyeccion.to_csv("proyeccion_2026.csv", index=False)

print("Proyección generada correctamente")
