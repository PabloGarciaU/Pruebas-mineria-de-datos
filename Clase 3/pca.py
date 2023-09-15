from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df= pd.read_csv(r"Datasets/pd_speech_features.csv")

print (df.head())
#Standardizing the Dataset
scaler = StandardScaler()
df_std = scaler.fit_transform(df)
print (df_std)
print (df.shape)

#PCA
pca2 = PCA(n_components=2)
principalComponents = pca2.fit_transform(df_std) #Fit the model with X and apply the dimensionality reduction on X.
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['class']]], axis = 1)
print(finalDf.head())
print (finalDf.shape)

print(pca2.components_)
print(pca2.explained_variance_)
print(pca2.explained_variance_ratio_)
print(pca2.singular_values_)

#Visualizing Data in 2 Dimension Scatter Plot
plt.figure(figsize=(7,7))
plt.scatter(finalDf['principal component 1'],finalDf['principal component 2'],c=finalDf['class'],cmap='prism', s =5)
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.savefig("pca2.png")
#Applying PCA with Principal Components = 3
pca3 = PCA(n_components=3)
principalComponents = pca3.fit_transform(df_std)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

finalDf = pd.concat([principalDf, df[['class']]], axis = 1)
print (finalDf.head())
print (finalDf.shape)
print(pca3.components_)
print(pca3.explained_variance_)
print(pca3.explained_variance_ratio_)
print(pca3.singular_values_)


from mpl_toolkits.mplot3d import Axes3D

fig2 = plt.figure(figsize=(9,9), clear=True)
axes = Axes3D(fig2)
axes.set_title('PCA Representation', size=14)
axes.set_xlabel('PC1')
axes.set_ylabel('PC2')
axes.set_zlabel('PC3')

axes.scatter(finalDf['principal component 1'],finalDf['principal component 2'],finalDf['principal component 3'],c=finalDf['class'], cmap = 'prism', s=10)
plt.savefig("pca3.png")
#https://machinelearningknowledge.ai/complete-tutorial-for-pca-in-python-sklearn-with-example/
#https://www.kaggle.com/datasets/dipayanbiswas/parkinsons-disease-speech-signal-features
#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
