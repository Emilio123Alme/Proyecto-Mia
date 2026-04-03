plt.figure(figsize=(14,6))
plt.scatter(df['time_real'], df['Saldo'], label='Datos reales')

plt.plot(resultado['time_real'], resultado['Pred_LR'], label='LR')
plt.plot(resultado['time_real'], resultado['Pred_RF'], label='RF')
plt.plot(resultado['time_real'], resultado['Pred_NN'], label='NN')

plt.legend()
plt.show()
