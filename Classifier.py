from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import  TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import sklearn.svm
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report










clfSVM=sklearn.svm.LinearSVC().fit(trainingMatrix,trainingTags)

labelsResults=clfSVM.predict(testMatrix)
results=[True if testTags[i]==labelsResults[i] else False for i in range(0,len(testTags))]

print(classification_report(testTags,labelsResults,labels=['Action','Musical','Adventure','Comedy','Crime','Romance','War'],target_names=['Action','Musical','Adventure','Comedy','Crime','Romance','War']))






