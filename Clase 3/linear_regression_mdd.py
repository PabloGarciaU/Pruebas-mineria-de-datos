from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

df = pd.read_csv("Datasets/vino.csv")
print(df)

X=df[["acidez_fija","acidez_volatil","acido_citrico","azucar_res","cloruros","SO2_libre","SO2_total","densidad","pH","sulfatos", "calidad"]]
y=df["alcohol"]
#Y [alcohol] =  5.37940227e-01 * acidez_fija +  3.25509805e-01 * acidez_volatil +  5.07897233e-01 * acido_nitrico +  2.56993018e-01 * azucar_res +  -8.76304448e-01 * cloruros +  -2.72079219e-03 * SO2_libre + -8.85806917e-04 * SO2_total + -5.84946385e+02 * densidad +   3.56645440e+00 * pH +  8.96474605e-01 * sulfatos +  2.51432764e-01 * calidad 
reg = LinearRegression().fit(X, y)
print (str(reg.score(X, y)))
print (str(reg.coef_))
print (str(reg.intercept_))

print (str(reg.predict(np.array([[ 7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 5]]))))
