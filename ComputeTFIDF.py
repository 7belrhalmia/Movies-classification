# -*- coding: utf-8
import io
import nltk
import os
import scipy
import numpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import  TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import sklearn.svm
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import csv


stemmedDataPath="C:/Users/Mohamed Ali/Desktop/9raya/Paris Sud/TAL/Projet/Clean Data stemmed/"
TokenizedDataPath="C:/Users/Mohamed Ali/Desktop/9raya/Paris Sud/TAL/Projet/Clean Data/"


actionList=os.listdir(stemmedDataPath+"Action/")
adventureList=os.listdir(stemmedDataPath+"Adventure/")
comedyList=os.listdir(stemmedDataPath+"/Comedy/")
crimeList=os.listdir(stemmedDataPath+"/Crime/")
romanceList=os.listdir(stemmedDataPath+"/Romance/")
musicalList=os.listdir(stemmedDataPath+"/Musical/")
warList=os.listdir(stemmedDataPath+"/War/")
westernList=os.listdir(stemmedDataPath+"/Western/")

listMovies=[]
listMovies.append(actionList)
listMovies.append(adventureList)
listMovies.append(comedyList)
listMovies.append(crimeList)
listMovies.append(romanceList)
listMovies.append(musicalList)
listMovies.append(warList)
listMovies.append(westernList)

directory=("C:/Users/Mohamed Ali/Desktop/9raya/Paris Sud/TAL/Projet/Clean Data2/")


def cleanTheLists(listMovies):
    
    i=0;
    try:
        while(i<len(listMovies)-1):
            listCategory=listMovies[i]
            m=i+1
            while ( m<len(listMovies)):
                k=-1;
                while(k<len(listMovies[m])-1):
                   k=k+1
                   if listMovies[m][k] in listCategory:
                       if (len(listCategory)>len(listMovies[m])):
                           listCategory.remove(listMovies[m][k])
                       else:
                           listMovies[m].remove(listMovies[m][k])
                       k=-1  
                m=m+1    
            i=i+1
    except:
        pass
    return(listMovies)

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
        f=io.open(directory+category+os.path.sep+subList[i],encoding='ASCII',errors='replace')
        strFile=f.read()
        f.close();
        testList.append(strFile)
        testTags.append(category)
    return( docsList,testList,tags,testTags)

def populateKonwledgeBaseRDN(category,directory,subList,trainingRate=0.8):
    docsList=list()
    testList=list()
    testTags=list()
    tags=list()
    for i,item in enumerate(subList):
        f=io.open(directory+category+os.path.sep+subList[i],encoding='ascii',errors='replace')
        
        strFile=(f.read()).encode('ascii','replace')
        print type(strFile)
        f.close();
        if numpy.random.rand()<=0.8:
            docsList.append(strFile)
            tags.append(category)
        else:
             testList.append(strFile)
             testTags.append(category)
            
    return( docsList,testList,tags,testTags)
def saveThefiles(docToSave,labels,name):
    try:
        with open(name+".csv", "wb") as f:
            
            writer = csv.writer(f)
            for i in range(0,len(docToSave)):    
                writer.writerows([[docToSave[i]]] )
    except  Exception,e:
            print "docs"+str(+e)
    try:
        with open(name+"labels.csv","wb") as f:
            writer = csv.writer(f)
            for i in range(0,len(labels)):    
                writer.writerows([[labels[i]]])
    except Exception,e:
        print "labels"+str(e)
def getData(directory=directory):

    docsList=list()
    testData=list()
    testTags=list()
    docTags=list()
    
    docsClass,annex,tags,annexTags=populateKonwledgeBaseRDN("Action",directory,actionList)
    docsList.extend(docsClass)
    testData.extend(annex)
    testTags.extend(annexTags)
    docTags.extend(tags)
    
    print "finished Action"
    
    docsClass,annex,tags,annexTags=populateKonwledgeBaseRDN("Musical",directory,musicalList)
    docsList.extend(docsClass)
    testData.extend(annex)
    testTags.extend(annexTags)
    docTags.extend(tags)
    
    print "finished Musical"
    docsClass,annex,tags,annexTags=populateKonwledgeBaseRDN("Adventure",directory,adventureList)
    docsList.extend(docsClass)
    testData.extend(annex)
    testTags.extend(annexTags)
    docTags.extend(tags)
    print "finished Adventure"
    docsClass,annex,tags,annexTags=populateKonwledgeBaseRDN("Comedy",directory,comedyList)
    docsList.extend(docsClass)
    docTags.extend(tags)
    testData.extend(annex)
    testTags.extend(annexTags)
    
    print "finished Comedy"
    docsClass,annex,tags,annexTags=populateKonwledgeBaseRDN("Crime",directory,crimeList)
    docsList.extend(docsClass)
    testData.extend(annex)
    testTags.extend(annexTags)
    docTags.extend(tags)
    
    print "finished Crime"
    
    docsClass,annex,tags,annexTags=populateKonwledgeBaseRDN("Romance",directory,romanceList)
    docsList.extend(docsClass)
    testData.extend(annex)
    testTags.extend(annexTags)
    docTags.extend(tags)
    
    print "finished Romance"
    
    docsClass,annex,tags,annexTags=populateKonwledgeBaseRDN("War",directory,warList)
    docsList.extend(docsClass)
    testData.extend(annex)
    testTags.extend(annexTags)
    docTags.extend(tags)
    
    print "finished War"
    docsClass,annex,tags,annexTags=populateKonwledgeBaseRDN("Western",directory,westernList)
    docsList.extend(docsClass)
    testData.extend(annex)
    testTags.extend(annexTags)
    docTags.extend(tags)
    
    print "finished Western"
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
    
    print "finished TFIDF for training"
    trainingMatrix=tfidfscores
    trainingTags=docTags
    vectorizer2 = TfidfVectorizer( stop_words='english',ngram_range=(1,1),vocabulary=vocab)
    testMatrix=vectorizer2.fit_transform(testData)
    
    print "finished TFIDF for test"
    
    return trainingMatrix,trainingTags,testMatrix,testTags,docsList,testData,trainingTags,testTags

actionList,adventureList,comedyList,crimeList,romanceList,musicalList,warList,westernList=cleanTheLists(listMovies)



#trainingMatrix,trainingTags,testMatrix,testTags=getData(directory)

trainingMatrixStem,trainingTagsStem,testMatrixStem,testTagsStem,trainingDocsStemmed,testDocsStemmed,trainingLabelsStemmed,testLabelsStemmed=getData(stemmedDataPath)
trainingMatrix,trainingTags,testMatrix,testTags,trainingDocs,testDocs,trainingLabels,testLabels=getData(TokenizedDataPath)




saveThefiles(trainingDocs,trainingLabels,"Training")
print "saved training"
saveThefiles(testDocs,testLabels,"Test")
print "saved test"
scipy.sparse.save_npz("TrainingMatrixStemm.npz",trainingMatrixStem)
print "saved training tfidf"
scipy.sparse.save_npz("TestMatrixStemm.npz",testMatrixStem)
scipy.sparse.save_npz("TestMatrix.npz",testMatrixStem)
scipy.sparse.save_npz("TrainingMatrix.npz",trainingMatrix)
scipy.sparse.save_npz("TestMatrix.npz",testMatrix)
print "saved test tfidf"
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