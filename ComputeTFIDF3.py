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


directory=("C:/Users/Mohamed Ali/Desktop/9raya/Paris Sud/TAL/Projet/Data_Full/")


actionList=os.listdir(directory+"Action/")
adventureList=os.listdir(directory+"Adventure/")
comedyList=os.listdir(directory+"/Comedy/")
crimeList=os.listdir(directory+"/Crime/")
romanceList=os.listdir(directory+"/Romance/")
musicalList=os.listdir(directory+"/Musical/")
warList=os.listdir(directory+"/War/")
westernList=os.listdir(directory+"/Western/")
horrorList=os.listdir(directory+"/Horror/")


listMovies=[]
listMovies.append(actionList)
listMovies.append(adventureList)
listMovies.append(comedyList)
listMovies.append(crimeList)
listMovies.append(romanceList)
listMovies.append(musicalList)
listMovies.append(warList)
listMovies.append(westernList)



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
    valList=list();
    valTags=list();
    for i,item in enumerate(subList):
        f=io.open(directory+category+os.path.sep+subList[i],encoding='ascii',errors='replace')
        
        strFile=(f.read()).encode('ascii','replace')
#        print type(strFile)
        f.close();
        number=numpy.random.rand()
        if number<=0.7:
            docsList.append(strFile)
            tags.append(category)
        elif ((number>0.7) and (number <0.8)):
             testList.append(strFile)
             testTags.append(category)
        else:
              valList.append(strFile)
              valTags.append(category)
            
    return( docsList,testList,valList,tags,testTags,valTags)
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
    testTagsAll=list()
    docTags=list()
    
    validationTags=list()
    validationData=list()
    
    docsClass,test,valData,tags,testTags,valTags=populateKonwledgeBaseRDN("Action",directory,actionList)
    docsList.extend(docsClass)
    testData.extend(test)
    testTagsAll.extend(testTags)
    docTags.extend(tags)
    validationData.extend(valData)
    validationTags.extend(valTags)
    print "finished Action"
    
    
    docsClass,test,valData,tags,testTags,valTags=populateKonwledgeBaseRDN("Horror",directory,horrorList)
    docsList.extend(docsClass)
    testData.extend(test)
    testTagsAll.extend(testTags)
    docTags.extend(tags)
    validationData.extend(valData)
    validationTags.extend(valTags)
    
    
    print "finished Horror"
    
    
    docsClass,test,valData,tags,testTags,valTags=populateKonwledgeBaseRDN("Musical",directory,musicalList)
    docsList.extend(docsClass)
    testData.extend(test)
    testTagsAll.extend(testTags)
    docTags.extend(tags)
    validationData.extend(valData)
    validationTags.extend(valTags)
    
    
    print "finished Musical"
    docsClass,test,valData,tags,testTags,valTags=populateKonwledgeBaseRDN("Adventure",directory,adventureList)
    docsList.extend(docsClass)
    testData.extend(test)
    testTagsAll.extend(testTags)
    docTags.extend(tags)
    validationData.extend(valData)
    validationTags.extend(valTags)
    
    print "finished Adventure"
    docsClass,test,valData,tags,testTags,valTags=populateKonwledgeBaseRDN("Comedy",directory,comedyList)
    docsList.extend(docsClass)
    docTags.extend(tags)
    testData.extend(test)
    testTagsAll.extend(testTags)
    validationData.extend(valData)
    validationTags.extend(valTags)
    
    
    print "finished Comedy"
    docsClass,test,valData,tags,testTags,valTags=populateKonwledgeBaseRDN("Crime",directory,crimeList)
    docsList.extend(docsClass)
    testData.extend(test)
    testTagsAll.extend(testTags)
    docTags.extend(tags)
    validationData.extend(valData)
    validationTags.extend(valTags)
    
    print "finished Crime"
    
    docsClass,test,valData,tags,testTags,valTags=populateKonwledgeBaseRDN("Romance",directory,romanceList)
    docsList.extend(docsClass)
    testData.extend(test)
    testTagsAll.extend(testTags)
    docTags.extend(tags)
    validationData.extend(valData)
    validationTags.extend(valTags)
    
    print "finished Romance"
    
    docsClass,test,valData,tags,testTags,valTags=populateKonwledgeBaseRDN("War",directory,warList)
    docsList.extend(docsClass)
    testData.extend(test)
    testTagsAll.extend(testTags)
    docTags.extend(tags)
    validationData.extend(valData)
    validationTags.extend(valTags)
    
    print "finished War"
    docsClass,test,valData,tags,testTags,valTags=populateKonwledgeBaseRDN("Western",directory,westernList)
    docsList.extend(docsClass)
    testData.extend(test)
    testTagsAll.extend(testTags)
    docTags.extend(tags)
    validationData.extend(valData)
    validationTags.extend(valTags)
    print "finished Western"
#    docTags.extend(testTags)
#    docsList.extend(testData)
#    trainingTags=docTags[0:len(docTags)-len(testTags)]    
    trainingTags=docTags
    
    tags=list()
    vectorizer = TfidfVectorizer( stop_words='english',ngram_range=(1,1))
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
    validationMatrix=vectorizer2.fit_transform(validationData)
    print "finished TFIDF for test"
    
    return trainingMatrix,trainingTags,testMatrix,testTagsAll,validationMatrix,validationTags

actionList,adventureList,comedyList,crimeList,romanceList,musicalList,warList,westernList=cleanTheLists(listMovies)



#trainingMatrix,trainingTags,testMatrix,testTags=getData(directory)

#trainingMatrixStem,trainingTagsStem,testMatrixStem,testTagsStem,trainingDocsStemmed,testDocsStemmed,trainingLabelsStemmed,testLabelsStemmed=getData(stemmedDataPath)
trainingMatrix,trainingTags,testMatrix,testTags,validationMatrix,validationTags=getData(directory)

