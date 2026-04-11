import pandas as pd
import matplotlib.pyplot as plt


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
