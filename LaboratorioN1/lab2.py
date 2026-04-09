import pandas as pd


#Lectura de datos
df = pd.read_csv('datos_ventas_suc1.csv', encoding='latin-1')
df2 = pd.read_csv('datos_ventas_suc2.csv', encoding='latin-1')
#print(df)
#print(df2)

#Fusión de datos
print("Df fusionado")
df_fusionado = pd.concat([df,df2], ignore_index=True)
print(df_fusionado)

#Tratamiento de datos
df_fusionado['Fecha'] = pd.to_datetime(df_fusionado['Fecha'])
print(df_fusionado)