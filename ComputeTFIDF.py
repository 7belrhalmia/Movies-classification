# -*- coding: utf-8
import io
import nltk
import os
import scipy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import  TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import sklearn.svm
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


stemmedDataPath="C:/Users/Mohamed Ali/Desktop/9raya/Paris Sud/TAL/Projet/Clean Data stemmed/"
actionsList=os.listdir(stemmedDataPath+"Action/")
adventureList=os.listdir(stemmedDataPath+"Adventure/")
comedyList=os.listdir(stemmedDataPath+"/Comedy/")
crimeList=os.listdir(stemmedDataPath+"/Crime/")
romanceList=os.listdir(stemmedDataPath+"/Romance/")
musicalList=os.listdir(stemmedDataPath+"/Musical/")
warList=os.listdir(stemmedDataPath+"/War/")


directory=("C:/Users/Mohamed Ali/Desktop/9raya/Paris Sud/TAL/Projet/Clean Data/")


def tokenize(string):
    return(nltk.tokenize.word_tokenize(string))

def populateKonwledgeBase(category,directory,subList,trainingRate=0.8):
    docsList=list()
    testList=list()
    testTags=list()
    tags=list()
    bound=range(0,int(trainingRate*len(subList)))
    for i in bound:
        f=io.open(directory+category+os.path.sep+subList[i],encoding='utf8',errors='replace')
        strFile=f.read()
        f.close();
        docsList.append(strFile)
        tags.append(category)
    testBound=range(int(0.8*len(subList))+1,len(subList))
    for i in testBound:
        f=io.open(directory+category+os.path.sep+subList[i],encoding='utf8',errors='replace')
        strFile=f.read()
        f.close();
        testList.append(strFile)
        testTags.append(category)
    return( docsList,testList,tags,testTags)




def getData(directory=directory):

    docsList=list()
    testData=list()
    testTags=list()
    docTags=list()
    
    docsClass,annex,tags,annexTags=populateKonwledgeBase("Action",directory,actionsList)
    docsList.extend(docsClass)
    testData.extend(annex)
    testTags.extend(annexTags)
    docTags.extend(tags)
    
    
    
    docsClass,annex,tags,annexTags=populateKonwledgeBase("Musical",directory,musicalList)
    docsList.extend(docsClass)
    testData.extend(annex)
    testTags.extend(annexTags)
    docTags.extend(tags)
    
    docsClass,annex,tags,annexTags=populateKonwledgeBase("Adventure",directory,adventureList)
    docsList.extend(docsClass)
    testData.extend(annex)
    testTags.extend(annexTags)
    docTags.extend(tags)
    
    docsClass,annex,tags,annexTags=populateKonwledgeBase("Comedy",directory,comedyList)
    docsList.extend(docsClass)
    docTags.extend(tags)
    testData.extend(annex)
    testTags.extend(annexTags)
    
    
    docsClass,annex,tags,annexTags=populateKonwledgeBase("Crime",directory,crimeList)
    docsList.extend(docsClass)
    testData.extend(annex)
    testTags.extend(annexTags)
    docTags.extend(tags)
    
    docsClass,annex,tags,annexTags=populateKonwledgeBase("Romance",directory,romanceList)
    docsList.extend(docsClass)
    testData.extend(annex)
    testTags.extend(annexTags)
    docTags.extend(tags)
    
    docsClass,annex,tags,annexTags=populateKonwledgeBase("War",directory,warList)
    docsList.extend(docsClass)
    testData.extend(annex)
    testTags.extend(annexTags)
    docTags.extend(tags)    
#    docTags.extend(testTags)
#    docsList.extend(testData)
#    trainingTags=docTags[0:len(docTags)-len(testTags)]    
    trainingTags=docTags
    
    tags=list()
    vectorizer = TfidfVectorizer( stop_words='english')
    tfidfscores=vectorizer.fit_transform(docsList)
    vocab = vectorizer.vocabulary_
    
#    trainingMatrix=tfidfscores[0:len(docTags)-len(testTags)]
#    trainingTags=docTags[0:len(docTags)-len(testTags)]
#    testMatrix=tfidfscores[len(docTags)-len(testTags):]
    trainingMatrix=tfidfscores
    trainingTags=docTags
    vectorizer2 = TfidfVectorizer( stop_words='english',vocabulary=vocab)
    testMatrix=vectorizer2.fit_transform(testData)
    
    return trainingMatrix,trainingTags,testMatrix,testTags


#trainingMatrix,trainingTags,testMatrix,testTags=getData(directory)

trainingMatrixStem,trainingTagsStem,testMatrixStem,testTagsStem=getData(stemmedDataPath)




scipy.sparse.save_npz("trainingMatrixStemm.npz",trainingMatrixStem)
scipy.sparse.save_npz("TestMatrixStemm.npz",testMatrixStem)

#scipy.sparse.save_npz("trainingMatrix.npz",trainingMatrix)
#scipy.sparse.save_npz("TestMatrix.npz",testMatrix)

#
#clf = MultinomialNB().fit(tfidfscores[0:350], tags[0:350])
#clf.predict(tfidfscores[51:])
#
#clfSVM=sklearn.svm.LinearSVC().fit(trainingMatrix,trainingTags)

#
#clsSVM=sklearn.svm.SVC().fit(trainingMatrix,trainingTags)
#
#labelsResults=clfSVM.predict(testMatrix)
#
#results=[True if testTags[i]==labelsResults[i] else False for i in range(0,len(testTags))]
#
#
#print(classification_report(testTags,labelsResults,labels=['Action','Musical','Adventure','Comedy','Crime','Romance','War'],target_names=['Action','Musical','Adventure','Comedy','Crime','Romance','War']))

#clf = sklearn.svm.SVC(kernel='linear', C=1)
#scores = cross_val_score(clf, tfidfscores,docTags, cv=10)