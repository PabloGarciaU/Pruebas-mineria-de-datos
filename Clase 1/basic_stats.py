import pandas as pd
import math
import matplotlib.pyplot as plt #conda install matplotlib

df = pd.read_csv("Datasets/vino.csv")

print(df)

#promedio // mean
promedios = df.mean()
print (promedios)
print (df[['pH','sulfatos']].mean())

#desviacion standard
standard_dev = df.std()
print (standard_dev)
print (df[['pH','sulfatos']].std())

#max-min
max = df.max()
print (max)
print (df[['pH','sulfatos']].max())
min = df.min()
print (min)
print (df[['pH','sulfatos']].min())
#La fila que corresponde al maximo/minimo
print (df[df['pH']==df['pH'].max()])
print (df[df['pH']==df['pH'].min()])

#sumas
sum = df.sum()
print (sum)
print (df[['pH','sulfatos']].sum())

#sumas
medians = df.median()
print (medians)
print (df[['pH','sulfatos']].median())

#coeficiente de variacion
for name, values in df.iteritems():
  coef_variacion = df[name].std()/df[name].mean()
  print ("Coef. Variacion " + str(name) + ": " + str(coef_variacion))


#coeficiente de asimetria
for name, values in df.iteritems():
  dev_std_3 = df[name].std() * df[name].std() * df[name].std()
  promedio = df[name].mean()
  sum = 0
  for value in values:
    sum = sum + (value - promedio) * (value - promedio) * (value - promedio)
  print ("Coef. asimetria " + str(name) + ": " + str((sum/dev_std_3)/len(df.index)))

df.plot(x ='alcohol', y='pH', kind = 'scatter')
plt.savefig("alcohol_ph.png")
  
#sturges
max = df['alcohol'].max()
min = df['alcohol'].min()
R = max - min
c= R/(1 + math.log(len(df.index))/math.log(2))

max_2 = math.ceil(max)
min_2 = math.floor(min)

c_2 = math.ceil(c)

range_df = pd.DataFrame()
x_axis, y_axis = [], []

for x in range(min_2,max_2,c_2):
  x_t = "["+str(x)+","+str(x+c_2)+"]"
  y_t = df[(df['alcohol'] > x) & (df['alcohol'] < (x+c_2))].shape[0]
  print (x_t +"=" + str(y_t))
  x_axis.append(x_t)
  y_axis.append(y_t)

range_df['X']=x_axis
range_df['Y']=y_axis

print (range_df)
range_df.plot(x="X", y="Y", kind="bar", rot=5, fontsize=4)
#plt.show()
plt.savefig("alcohol.png")

#entropia
range_df["prob"]=range_df["Y"]/len(df.index)
entropia = 0
for index, row in range_df.iterrows():
  entropia = row["prob"]*math.log(row["prob"])/math.log(7)
entropia = -1.00 * entropia
print (entropia)
#https://www.educba.com/pandas-dataframe-plot/
