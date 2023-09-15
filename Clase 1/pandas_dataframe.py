import pandas as pd

print ("#### PASO 1 #####")

df = pd.DataFrame()
print (df)

print ("#### PASO 2 #####")

data = [5, 4, 3, 2 ,1 ]
df = pd.DataFrame(data)
print (df)

print ("#### PASO 3 #####")
data = [['Alex',10],['Bob',12],['Clarke',13], ['Ivan', 99]]
df = pd.DataFrame(data,columns=['Name','Age'])
print (df)

print ("#### PASO 4 #####")
#Create a DataFrame from Dict of ndarrays / Lists
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data)
print (df)

print ("#### PASO 5 #####")
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data, index=['Primer registro','Segundo registro','rank3','rank4'])
print (df)

print ("#### PASO 6 #####")
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20},{'f': 5, 'c': 10, 'd': 20}]
df = pd.DataFrame(data)
print (df)

print ("#### PASO 7 #####")
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data, index=['first', 'second'])
print (df)

print ("#### PASO 8 #####")
#varias vistas (dataframes) desde el mismo conjunto de datos
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df1 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b'])
df2 = pd.DataFrame(data, index=['primero', 'segundo'], columns=['a', 'B'])
print (df1)
print (df2)

print ("##### recuperar una columna ######")
print (df1['a']) #recuperacion de columna/variable
print (df1.iloc[1]) #recuperacion de registro/vector

print ("###operacion##")
df1['sum']=df1['a']+df1['b'] #adjunto al final la columna suma que es la suma de las columna a y b
print (df1)
print ("### eliminacion de una columna ##")
del df1['sum']
print (df1)

print ("### slicing ##")
data = [{'a': 1, 'b': 2, 'c': 34, 'd': 20},{'a': 5, 'b': 10, 'c': 20, 'd':33},{'a': 95, 'b': 12, 'c': 56, 'd':78} ,{'a': 00, 'b': 11, 'c': 22, 'd':33} ,{'a': 10, 'b': 20, 'c': 30, 'd':40}]
df = pd.DataFrame(data, index=['first', 'second','third', 'fourth', 'fifth'], columns=['a', 'b', 'c', 'd'])
print (df)
print (df[1:3]) #slicing

print ("### append ##")
df_aux = pd.DataFrame(data)
df=df.append(df_aux)
#df=df.concat(df_aux)
print (df)

print ("### drop ##")
df = df.drop('first') #elimina el registro//vector
print (df)
