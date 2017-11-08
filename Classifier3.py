
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.feature_extraction.text import  TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import sklearn.svm
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC


import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def classifierResults(clf,trainingData,trainingTags,testData,testTags):
    clf.fit(trainingData,trainingTags)
    labelsResults=clf.predict(testData)
    print(classification_report(testTags,labelsResults,target_names=['Action','Musical','Adventure','Comedy','Crime','Romance','War','Western','Horror']))
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(testTags, labelsResults)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Action','Musical','Adventure','Comedy','Crime','Romance','War','Western','Horror'],
                          title='Confusion matrix, without normalization')
    plt.show()


clfNSVM=sklearn.svm.SVC(kernel='linear',C=1)
classifierResults(clfNSVM,trainingMatrix,trainingTags,testMatrix,testTags)
#
#clf.fit(trainingData,trainingTags)
#labelsResults=clf.predict(testData)
#print(classification_report(testTags,labelsResults,target_names=['Action','Musical','Adventure','Comedy','Crime','Romance','War','Western','Horror']))



#trainingMatrix,trainingTags,testMatrix,testTags,validationMatrix,validationTags
#
#parameter=[0.1,0.2,0.3,0.4,0.5,0.6,0.8,0.9,1,2,3,4,5,6,7,8,15,20]
#
#for c in parameter:
#    clfC=sklearn.svm.SVC(C=c,kernel='linear')
#    clfC.fit(trainingMatrix,trainingTags)
#    labelsResults=clfC.predict(testMatrix)
#    print(classification_report(testTags,labelsResults,labels=['Action','Musical','Adventure','Comedy','Crime','Romance','War','Western'],target_names=['Action','Musical','Adventure','Comedy','Crime','Romance','War','Western']))
#    print ("Results with C"+str(c) )

    