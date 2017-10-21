import os
import os.path
import re
import nltk
import string
import shutil
def clean_text(text):
        lowers = text.lower() #lower case for everyone
        #remove the punctuation using the character deletion step of translate
        punct_killer = str.maketrans('', '', string.punctuation)
        no_punctuation = lowers.translate(punct_killer)
        return no_punctuation

def get_tokens(file):
        with open(file, 'r') as d:
            text = d.read()
            tokens = nltk.word_tokenize(clean_text(text))
        return tokens

def sup_noise(Str):
        # return re.sub(r'\d+:\d+:\d+,\d* --> \d+:\d+:\d+','',Str)
        return re.sub(r'\d*','',re.sub(r'\(.+\)','',re.sub(r'\d+:\d+:\d+,\d* --> \d+:\d+:\d+,','',Str)))

dir='/home/belr/Projet_TAL/train_data'
dir_h='/home/belr/Projet_TAL/train_data_postprocessed'
# print os.walk(dir).next()[1]
dict_label={}
i=0
# print os.listdir(".")
for dirnames in os.walk(dir).next()[1]:
        print(dirnames)
        if not os.path.exists(dir_h+os.path.sep+dirnames):
           os.makedirs(dir_h+os.path.sep+dirnames)
        subf= dir+os.path.sep+dirnames
        # print(type(string.punctuation))
        for fil in os.walk(subf).next()[2]:
            file = open(subf+os.path.sep+fil, "r")
            # s=[]
            # for t in file:
            #     s.append(t)
            res= sup_noise(file.read())
            # res=clean_text(res)
            res = ''.join([c for c in res if c not in ('!','.','?',',',':')])
            tokens=nltk.word_tokenize(res)
            # g=open(dir_h+os.path.sep+dirnames+os.path.sep+fil,"w")
            # [g.write(x+' ') for x in tokens]
            print tokens
            # [shutil.rmtree(dir_h+os.path.sep+x) for x in os.walk(dir_h).next()[1]]


        dict_label[dirnames]=i
        i+=1

# print dict_label

