from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.feature_extraction.text import  TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import sklearn.svm
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
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
    print(classification_report(testTags,labelsResults,target_names=['Action','Musical','Adventure','Comedy','Crime','Romance','War','Western']))
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(testTags, labelsResults)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Action','Musical','Adventure','Comedy','Crime','Romance','War','Western'],
                          title='Confusion matrix, without normalization')
    plt.show()

clfSVMFull=sklearn.svm.LinearSVC(C=1)
classifierResults(clfSVMFull,trainingMatrix,trainingTags,testMatrix,testTags)
clfSVM=sklearn.svm.LinearSVC(C=1)
classifierResults(clfSVM,trainingMatrixStem,trainingTagsStem,testMatrixStem,testTagsStem)
#clfMultiNomial=NB()
#classifierResults(clfMultiNomial,trainingMatrixStem,trainingTagsStem,testMatrixStem,testTagsStem)
clfNSVM=sklearn.svm.SVC(kernel='linear',C=1)
classifierResults(clfNSVM,trainingMatrixStem,trainingTagsStem,testMatrixStem,testTagsStem)

sklearn
#.fit(trainingMatrixStem,trainingTagsStem)
#
#labelsResults=clfSVM.predict(testMatrixStem)
#results=[True if testTagsStem[i]==labelsResults[i] else False for i in range(0,len(testTagsStem))]
#
#
#
#print(classification_report(testTagsStem,labelsResults,target_names=['Action','Musical','Adventure','Comedy','Crime','Romance','War','Western']))
#
#clfMultiNomial.fit(trainingMatrixStem,trainingTagsStem)
#clfMultiNomial.predict(testMatrixStem)
#
## Compute confusion matrix
#cnf_matrix = confusion_matrix(testTagsStem, labelsResults)
#np.set_printoptions(precision=2)
#
## Plot non-normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=['Action','Musical','Adventure','Comedy','Crime','Romance','War','Western'],
#                      title='Confusion matrix, without normalization')
#plt.show()
#

#grid search


#parameter=[1,2,3,4,5,6,7,8,15,20]
#
#for c in parameter:
#    clfC=sklearn.svm.LinearSVC(C=0.25)
#    clfC.fit(trainingMatrixStem,trainingTagsStem)
#    labelsResults=clfC.predict(testMatrixStem)
#    print(classification_report(testTagsStem,labelsResults,labels=['Action','Musical','Adventure','Comedy','Crime','Romance','War'],target_names=['Action','Musical','Adventure','Comedy','Crime','Romance','War']))
#    print ("Results with C"+str(c) )

    