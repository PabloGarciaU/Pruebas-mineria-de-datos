import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing # conda install -c cctbx202208 scikit-learn  v. 1.1.2
import statsmodels.api as sm #conda install statsmodels
import numpy as np
from outliers import smirnov_grubbs as grubbs #pip install outlier_utils

df = pd.read_csv("Datasets/vino.csv")
print(df)

#Xminmax para todas las variables
#minimos = df.min() #el maravilloso pandas nos da el min para todas las variable
#maximos = df.max()

#for name, values in df.iteritems():
#    Xminmax = (df[name] - minimos[name]) / (maximos[name] - minimos[name])
#    print (Xminmax)

#Alternativa usando sklearn
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df)
df_scaled = pd.DataFrame(x_scaled)
print (df_scaled)

#Standarizacion normal
#df_std = pd.DataFrame()
#for name, values in df.iteritems():
#    df_std[name] = (df[name] - df[name].mean()) / df[name].std()
#    print (df_std[name])

#Alternativa usando sklearn
scaler = preprocessing.StandardScaler()
df_std = scaler.fit_transform(df)
print (df_std)


#plt.boxplot(df['alcohol'])
#plt.savefig("alcohol-boxplot.png")

#plt.figure()
#plt.scatter(df['alcohol'],df['pH'])
#plt.savefig("alcohol-pH-scatter.png")

#plt.figure()
#fig, ax = plt.subplots(figsize =(10, 7))
#ax.hist(df['alcohol'], bins = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
#plt.savefig("alcohol-hist.png")

#plt.figure()
#fig = sm.qqplot(df['alcohol'], line='45')
#plt.savefig("alcohol-qqplot.png")

#data = np.random.normal(0,1, 1000)
#data = np.random.uniform(0,1, 1000)
#plt.figure()
#fig = sm.qqplot(data, line='45')
#plt.savefig("random-normal-qqplot.png")

test = grubbs.test(df['alcohol'], alpha=.05) #This function simply returns an array with the outliers removed. 
indexes = grubbs.max_test_indices(df['alcohol'], alpha=.05)
values = grubbs.max_test_outliers(df['alcohol'], alpha=.05)
print (indexes)
print (values)
print (test)

pearson = df.corr(method='pearson')
spearman = df.corr(method='spearman')
kendall = df.corr(method='kendall')
print(pearson)
print(spearman)
print(kendall)
