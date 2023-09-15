import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV


def generate_graphs(allPredictions, allLabels, rocFilenamePrefix, classesNames=None, colorsNames=None):
  NUM_CLASSES = len(classesNames)
  if (NUM_CLASSES==2):
    allPositiveScores = []
    max_score, min_score, pos_label = allPredictions[0][0], allPredictions[0][0], 0
       
    for prediction in allPredictions:
      for i in range(NUM_CLASSES):
        if (max_score < prediction[i]):
          max_score = prediction[i]
        if (min_score > prediction[i]):
          min_score = prediction[i]

    max_score = max_score + abs(min_score)
    min_score = min_score + abs(min_score)

    for prediction in allPredictions:
      allPositiveScores.append((prediction[pos_label]+abs(min_score))/(max_score-min_score))

    fpr, tpr, _ = roc_curve(allLabels, allPositiveScores, pos_label=pos_label)
    auc_score = auc(fpr, tpr)

    plt.figure()
    graphLabel = 'AUC={0:.3f}'.format(auc_score) 
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=graphLabel)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic ')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(rocFilenamePrefix+"-roc.png",  pad_inches=5)

  else: #Multiclass-case AUCs                                                                                                                                                                             
    roc_auc, fpr, tpr = dict(), dict(), dict()

    print(str(allPredictions))
    for i in range(NUM_CLASSES):
      max_score, min_score = allPredictions[0][i], allPredictions[0][i]
      for prediction in allPredictions:
        if (max_score < prediction[i]):
          max_score = prediction[i]
        if (min_score > prediction[i]):
          min_score = prediction[i]
          
      max_score = max_score + abs(min_score)
      min_score = min_score + abs(min_score)
       
      allPositiveScores = []
      for prediction in allPredictions:
         allPositiveScores.append((prediction[i]+abs(min_score))/(max_score-min_score))
      fpr[i], tpr[i], _ = roc_curve(allLabels, allPositiveScores, pos_label=i)
      roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
      mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"],label='macro-average (AUC={0:0.3f})' ''.format(roc_auc["macro"]),
      color='navy', linestyle=':', linewidth=4)

    if colorsNames is None:
       colorsNames = ['aqua', 'darkorange', 'cornflowerblue', 'deeppink']
       #colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink']) from itertools import cycle
    for i, color in zip(range(NUM_CLASSES), colorsNames):
      if classesNames is None:
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
          label='{0} (AUC={1:0.3f})' ''.format(i, roc_auc[i]))
      else:
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
          label='{0} (AUC={1:0.3f})' ''.format(classesNames[i], roc_auc[i]))
      
      plt.plot([0, 1], [0, 1], 'k--', lw=2)
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.0])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver Operating Characteristic')
      plt.legend(loc="lower right")
      plt.tight_layout()
      plt.savefig(rocFilenamePrefix+"-roc.png",  pad_inches=5)


df = pd.read_csv("Datasets/cred_alem.csv")
print(df)

X=df[["edad","sexo","monto_credito","duracion","proposito","clase_riesgo"]]
y=df["alojamiento"]


x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.4,train_size=0.6)
x_test, x_eval, y_test, y_eval = train_test_split(x_test,y_test,test_size = 0.5,train_size =0.5)


model = LogisticRegression(solver='liblinear', random_state=0).fit(x_train, y_train)
print (str(model.classes_))
print (str(model.predict_proba(x_eval)))
print (str(model.predict(x_eval)))

cm_1 = confusion_matrix(y_eval, model.predict(x_eval), labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_1,display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues,values_format='d')
plt.tight_layout()
plt.savefig("lr-eval.png")
print (str(cm_1))
print(classification_report(y_eval, model.predict(x_eval)))

generate_graphs(model.predict_proba(x_eval), y_eval, "roc-eval-", classesNames=model.classes_, colorsNames=None)

print (str(model.predict_proba(x_test)))
print (str(model.predict(x_test)))
cm_2 = confusion_matrix(y_test, model.predict(x_test), labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_2,display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues,values_format='d')
plt.tight_layout()
plt.savefig("lr-test.png")
print (str(cm_2))
print(classification_report(y_test, model.predict(x_test)))

generate_graphs(model.predict_proba(x_test), y_test, "roc-test-", classesNames=model.classes_, colorsNames=None)


cv_model = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
print (str(cv_model.predict(X)))
print (str(cv_model.predict_proba(X)))
print (str(cv_model.score(X,y)))
generate_graphs(model.predict_proba(X), y, "roc-cv-", classesNames=model.classes_, colorsNames=None)

#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
#https://realpython.com/logistic-regression-python/
'''

penalty is a string ('l2' by default) that decides whether there is regularization and which approach to use. Other options are 'l1', 'elasticnet', and 'none'.

dual is a Boolean (False by default) that decides whether to use primal (when False) or dual formulation (when True).

tol is a floating-point number (0.0001 by default) that defines the tolerance for stopping the procedure.

C is a positive floating-point number (1.0 by default) that defines the relative strength of regularization. Smaller values indicate stronger regularization.

fit_intercept is a Boolean (True by default) that decides whether to calculate the intercept 𝑏₀ (when True) or consider it equal to zero (when False).

intercept_scaling is a floating-point number (1.0 by default) that defines the scaling of the intercept 𝑏₀.

class_weight is a dictionary, 'balanced', or None (default) that defines the weights related to each class. When None, all classes have the weight one.

random_state is an integer, an instance of numpy.RandomState, or None (default) that defines what pseudo-random number generator to use.

solver is a string ('liblinear' by default) that decides what solver to use for fitting the model. Other options are 'newton-cg', 'lbfgs', 'sag', and 'saga'.

max_iter is an integer (100 by default) that defines the maximum number of iterations by the solver during model fitting.

multi_class is a string ('ovr' by default) that decides the approach to use for handling multiple classes. Other options are 'multinomial' and 'auto'.

verbose is a non-negative integer (0 by default) that defines the verbosity for the 'liblinear' and 'lbfgs' solvers.

warm_start is a Boolean (False by default) that decides whether to reuse the previously obtained solution.

n_jobs is an integer or None (default) that defines the number of parallel processes to use. None usually means to use one core, while -1 means to use all available cores.

l1_ratio is either a floating-point number between zero and one or None (default). It defines the relative importance of the L1 part in the elastic-net regularization.

'liblinear' solver doesn’t work without regularization.
'newton-cg', 'sag', 'saga', and 'lbfgs' don’t support L1 regularization.
'saga' is the only solver that supports elastic-net regularization.

'''
