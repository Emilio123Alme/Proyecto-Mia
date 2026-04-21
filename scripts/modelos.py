# ==========================================================================
# 5. VARIABLES
# ==========================================================================

columns = [
    'Costo','Account','Vendor Name','Location',
    'Year','Period','Year_Period',
    'Lag_1','Lag_2','Rolling_mean_3'
]

df_model = df[columns].dropna()
df_model = pd.get_dummies(df_model, drop_first=True)

# ==========================================================================
# 6. SPLIT TEMPORAL
# ==========================================================================

df_model = df_model.sort_values('Year_Period')
split = int(len(df_model)*0.8)

df_train = df.iloc[:split].copy()
df_test = df.iloc[split:].copy()

train = df_model.iloc[:split]
test = df_model.iloc[split:]

X_train = train.drop('Costo', axis=1)
y_train = train['Costo']

X_test = test.drop('Costo', axis=1)
y_test = test['Costo']

# ==========================================================================
# VARIABLES SOLO PARA REGRESIÓN LINEAL
# ==========================================================================

features_lr = [col for col in X_train.columns if not any(x in col for x in ["Lag", "Rolling"])]

X_train_lr = X_train[features_lr]
X_test_lr = X_test[features_lr]

# ==========================================================================
# 7. FUNCIÓN MÉTRICAS
# ==========================================================================

def evaluar(y_real, y_pred):

    return (
        mean_absolute_error(y_real, y_pred),
        np.sqrt(mean_squared_error(y_real, y_pred)),
        r2_score(y_real, y_pred)
    )

# ==========================================================================
# 8. FUNCIÓN GRÁFICA
# ==========================================================================

def graficar(y_real, y_pred, titulo):

    n = 200
    plt.figure()
    plt.plot(y_real.values[:n], label="Real")
    plt.plot(y_pred[:n], label="Predicho", linestyle='--')
    plt.title(titulo)
    plt.legend()
    plt.show()

# ==========================================================================
# 9. MODELOS
# ==========================================================================

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

features_lr = X_train.columns[:10]

# Modelos

modelos = {
    "Regresión Lineal": LinearRegression(),
    "Árbol de Decisión": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
}

resultados = {}

for nombre, modelo in modelos.items():

    if nombre == "Regresión Lineal":
        modelo.fit(X_train_lr, y_train)
        pred = modelo.predict(X_test_lr)
    else:
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)

    # Métricas

    mae, rmse, r2 = evaluar(y_test, pred)

    resultados[nombre] = {
        "pred": pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "modelo": modelo
    }

    print(f"\n {nombre}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")

# ==========================================================================
# GRÁFICA CORRECTA
# ==========================================================================

    plt.figure()
    n = 200
    plt.plot(y_test.values[:n], label="Real", linewidth=2)
    plt.plot(pred[:n], label="Predicho", linestyle='--')

    plt.title(f"{nombre} - Real vs Predicho")
    plt.xlabel("Observaciones")
    plt.ylabel("Saldo")
    plt.legend()
    plt.show()

# ==========================================================================
# 10. GRÁFICA COMPARATIVA FINAL 
# ==========================================================================

plt.figure(figsize=(12,6))

n = 200

plt.plot(y_test.values[:n], label="Real", linewidth=3)

# Modelos
plt.plot(resultados["Regresión Lineal"]["pred"][:n], label="Regresión Lineal", linestyle='--', alpha=0.7)
plt.plot(resultados["Árbol de Decisión"]["pred"][:n], label="Árbol de Decisión", linestyle='--', alpha=0.7)
plt.plot(resultados["Random Forest"]["pred"][:n], label="Random Forest", linestyle='-', linewidth=2)

plt.title("Comparación de Modelos vs Valores Reales", fontsize=14)
plt.xlabel("Observaciones", fontsize=12)
plt.ylabel("Saldo", fontsize=12)

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("comparacion_modelos.png",dpi=300)
plt.show()

# ==========================================================================
# TABLA COMPARATIVA DE MODELOS
# ==========================================================================

tabla_resultados = pd.DataFrame({
    "Modelo": [],
    "MAE": [],
    "RMSE": [],
    "R2": []
})

for nombre, res in resultados.items():
    tabla_resultados.loc[len(tabla_resultados)] = [
        nombre,
        res["mae"],
        res["rmse"],
        res["r2"]
    ]

print("\nTabla comparativa de modelos:")
print(tabla_resultados)
print("\n\nLa Regresión Lineal presentó un bajo desempeño (R² = 0.14), evidenciando una limitada capacidad para capturar la relación entre las variables, ")
print("especialmente en presencia de comportamientos no lineales y valores extremos.")
print("\n\nEl Árbol de Decisión mostró un ajuste casi perfecto a los datos, logrando replicar incluso los picos más altos. ")
print("Sin embargo, este comportamiento sugiere un posible sobreajuste, ya que el modelo podría estar memorizando los datos en lugar de generalizar patrones.")
print("\n\nPor otro lado, el modelo de Random Forest obtuvo el mejor desempeño global, con un R² de 0.7859, un MAE de 130.61 y un RMSE de 1531.80. ")
print("Este modelo logra capturar relaciones complejas sin sobreajustarse, gracias a la combinación de múltiples árboles de decisión.")
print("\n\nEn conclusión, Random Forest se selecciona como el modelo óptimo, al ofrecer el mejor equilibrio entre precisión y capacidad de generalización.")

# ==========================================================================
# 11. GUARDAR MODELOS
# ==========================================================================

import joblib

joblib.dump(resultados["Random Forest"]["modelo"], "modelo_rf.pkl")
joblib.dump(resultados["Regresión Lineal"]["modelo"], "modelo_lr.pkl")
joblib.dump(resultados["Árbol de Decisión"]["modelo"], "modelo_dt.pkl")
print("Modelos guardados correctamente")

