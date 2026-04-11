import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
#Lectura de datos
df = pd.read_csv('datos_ventas_suc1.csv', encoding='latin-1')
df2 = pd.read_csv('datos_ventas_suc2.csv', encoding='latin-1')
#print(df)
#print(df2)

#Fusión de datos
#print("Df fusionado")
df_fusionado = pd.concat([df,df2], ignore_index=True)
#print(df_fusionado)

#Tratamiento de datos
df_fusionado['Fecha'] = pd.to_datetime(df_fusionado['Fecha'])
#print(df_fusionado)

#Análisis de ventas
#print("Tabla cantidad de productos totales")
productoMasVendido = df_fusionado.groupby('Producto')['Cantidad'].sum()
#print(productoMasVendido)

ordenMeses = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
mesVentas = df_fusionado.groupby(df_fusionado['Fecha'].dt.month_name())['Total Venta'].sum().reset_index()
mesVentas['Fecha'] = pd.Categorical(mesVentas['Fecha'], categories=ordenMeses, ordered=True)
mesVentas = mesVentas.sort_values('Fecha')
print(mesVentas)


##Gráfico de torta

ventasPorCategorias = df_fusionado.groupby('Producto')['Total Venta'].sum().reset_index()

#print(ventasPorCategorias)


plt.figure(figsize=(8,8))

# Si SI usaste reset_index():
plt.pie(ventasPorCategorias['Total Venta'], labels=ventasPorCategorias['Producto'], autopct='%1.1f%%', shadow=True)
plt.title('Ventas por categoría')
plt.axis('equal')


#Gráfico de lineas
plt.figure(figsize=(10,6))
plt.plot(mesVentas['Fecha'], mesVentas['Total Venta'], marker='o')
plt.title('Ventas por mes')
plt.xlabel('Mes')
plt.ylabel('Total Venta')
plt.xticks(rotation=45)
plt.grid()
plt.show()  
"""

#Lectura de datos
df = pd.read_csv('games.csv')
#print(df)

df_filtrado = df[["id", "rated", "turns", "victory_status", "white_rating", "black_rating", "winner", "opening_name"]]
#print(df_filtrado)

df_filtrado2 = df[["turns", "victory_status", "white_rating", "black_rating", "winner"]]

#Aperturas más populares
openings_used = df_filtrado.groupby("opening_name").size().reset_index(name="count").sort_values(by="count", ascending=False).head(10)
##print(openings_used)

victorias_blancas = df_filtrado2[(df_filtrado2["winner"] == "white") & (df_filtrado2["victory_status"] == "resign")].sort_values(by="white_rating", ascending=False)
print(victorias_blancas)

#Grafico cantidad de turnos por victoria
turnos_victoria = df_filtrado2.groupby("victory_status")["turns"].mean().reset_index()
print(turnos_victoria)

plt.figure(figsize=(10,6))
plt.bar(turnos_victoria["victory_status"], turnos_victoria["turns"])
plt.title("Cantidad de turnos por victoria")
plt.xlabel("Victory Status")
plt.ylabel("Turns")
plt.grid()
plt.show(block=False)  # <-- Cambiado a False para permitir ver el siguiente gráfico

# --- Análisis: Qué jugadores tardan más en rendirse ---

# 1. Filtramos solo las partidas que terminaron en "resign"
df_resign = df[df['victory_status'] == 'resign'].copy()

# 2. Encontramos el elo del jugador que se rindió (el perdedor)
df_resign['loser_rating'] = np.where(df_resign['winner'] == 'white', df_resign['black_rating'], df_resign['white_rating'])

# 3. Agrupamos esos niveles (ratings) en rangos para ver una tendencia
bins = [0, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000]
labels = ['<1000', '1000-1250', '1250-1500', '1500-1750', '1750-2000', '2000-2250', '2250-2500', '>2500']
df_resign['rating_bin'] = pd.cut(df_resign['loser_rating'], bins=bins, labels=labels)

# 4. Calculamos el promedio de turnos que tardó cada grupo antes de rendirse
turnos_por_elo = df_resign.groupby('rating_bin', observed=False)['turns'].mean().reset_index()

print("\n--- Promedio de turnos antes de rendirse según el elo del perdedor ---")
print(turnos_por_elo)

# 5. Lo graficamos
plt.figure(figsize=(10,6))
plt.plot(turnos_por_elo['rating_bin'], turnos_por_elo['turns'], marker='o', color='red', linewidth=2)
plt.title("Promedio de turnos antes de rendirse por nivel del jugador")
plt.xlabel("Nivel (Rating) del Jugador que se rinde")
plt.ylabel("Promedio de Turnos para rendirse")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
