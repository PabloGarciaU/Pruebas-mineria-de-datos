import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn import svm #SVM
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler #Para estandarizar los datos 

df = pd.read_csv("Datasets/cred_alem.csv")
print(df)

X=df[["edad","sexo","monto_credito","duracion","proposito","clase_riesgo"]]
y=df["alojamiento"]

'''
 Framework heldout y lo que haremos es dividir el dataset en
60% training
40% evaluar (temp) => 20% evaluacion y 20% test
'''

x_train, x_temp, y_train, y_temp = train_test_split(X,y,test_size=0.4,train_size=0.6)
x_test, x_eval, y_test, y_eval = train_test_split(x_temp,y_temp,test_size = 0.5,train_size =0.5)

#escalamiento de los datos
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_eval = scaler.transform(x_eval) 
x_test = scaler.transform(x_test) 


print (len(x_train))
print (x_train)
print (len(x_eval))
print (x_eval)
print (len(x_test))
print (x_test)

#Me genera el modelo es decir entreno ... ok????
#criterion = gini|entropy

#modelo
model = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.8, multi_class='crammer_singer', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000).fit(x_train, y_train)
#model2 = svm.SVC(C=1.0, kernel='sigmoid', degree=2, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None).fit(x_train, y_train)
model3 = KNeighborsClassifier(n_neighbors=3,metric='euclidean').fit(x_train, y_train)
model2 = LogisticRegression().fit(x_train, y_train)
print (str(model2.classes_))
print (y)
#Despues de entrenar el modelo, yo voy a ver como le va en el de evaluacion (20%)
#print (str(model.predict_proba(x_eval)))
print (str(model.predict(x_eval)))


cm_svm  = confusion_matrix(y_eval, model.predict(x_eval), labels=model2.classes_)
cm_lr   = confusion_matrix(y_eval, model2.predict(x_eval), labels=model2.classes_)
cm_knn  = confusion_matrix(y_eval, model3.predict(x_eval), labels=model3.classes_)
#esta funcion construye el grafico de la matriz de confusion
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_knn,display_labels=model2.classes_)
disp_svm.plot(cmap=plt.cm.Blues,values_format='d')
plt.tight_layout()
#Graba el grafico en un archivo
plt.savefig("knn-eval.png")
#te imprime por pantalla la matriz de confusion
print (str(cm_svm))
print (str(cm_lr))
print (str(cm_knn))
#te imprime las metricas: recall, precision, accuracy
print(classification_report(y_eval, model.predict(x_eval)))
print(classification_report(y_eval, model2.predict(x_eval)))
print(classification_report(y_eval, model3.predict(x_eval)))

#print (str(model.predict_proba(x_test)))
#print (str(model.predict(x_test)))
cm_2 = confusion_matrix(y_test, model3.predict(x_test), labels=model2.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_2,display_labels=model2.classes_)
disp.plot(cmap=plt.cm.Blues,values_format='d')
plt.tight_layout()
plt.savefig("knn-test.png")
print (str(cm_2))
print(classification_report(y_test, model.predict(x_test)))
print(classification_report(y_test, model2.predict(x_test)))
print(classification_report(y_test, model3.predict(x_test)))

'''

Parameters
n_neighborsint, default=5
Number of neighbors to use by default for kneighbors queries.

weights{‘uniform’, ‘distance’} or callable, default=’uniform’
Weight function used in prediction. Possible values:

‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.

‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.

[callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.

algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
Algorithm used to compute the nearest neighbors:

‘ball_tree’ will use BallTree

‘kd_tree’ will use KDTree

‘brute’ will use a brute-force search.

‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.

Note: fitting on sparse input will override the setting of this parameter, using brute force.

leaf_sizeint, default=30
Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.

pint, default=2
Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

metricstr or callable, default=’minkowski’
The distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric. For a list of available metrics, see the documentation of DistanceMetric and the metrics listed in sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS. Note that the “cosine” metric uses cosine_distances. If metric is “precomputed”, X is assumed to be a distance matrix and must be square during fit. X may be a sparse graph, in which case only “nonzero” elements may be considered neighbors.

metric_paramsdict, default=None
Additional keyword arguments for the metric function.

n_jobsint, default=None
The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details. Doesn’t affect fit method.

Attributes
classes_array of shape (n_classes,)


https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html#sklearn.metrics.DistanceMetric

identifier

class name

args

distance function

“euclidean”

EuclideanDistance

sqrt(sum((x - y)^2))

“manhattan”

ManhattanDistance

sum(|x - y|)

“chebyshev”

ChebyshevDistance

max(|x - y|)

“minkowski”

MinkowskiDistance

p, w

sum(w * |x - y|^p)^(1/p)

“wminkowski”

WMinkowskiDistance

p, w

sum(|w * (x - y)|^p)^(1/p)

“seuclidean”

SEuclideanDistance

V

sqrt(sum((x - y)^2 / V))

“mahalanobis”

MahalanobisDistance

V or VI

sqrt((x - y)' V^-1 (x - y))
'''
