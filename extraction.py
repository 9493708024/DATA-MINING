# ./lexparser.csh temp.txt
import re
import os
import csv
import nltk
import numpy as np
import pandas as pd
import shlex , shutil
import subprocess as sp
from nltk import word_tokenize
from operator import itemgetter

contractions = {r"ain't": r"am not",
                r"aren't": r"are not",
                r"can't": r"cannot",
                r"can't've": r"cannot have",
                r"'cause": r"because",
                r"could've": r"could have",
                r"couldn't": r"could not",
                r"couldn't've": r"could not have",
                r"didn't": r"did not",
                r"doesn't": r"does not",
                r"don't": r"do not",
                r"hadn't": r"had not",
                r"hadn't've": r"had not have",
                r"hasn't": r"has not",
                r"haven't": r"have not",
                r"he'd": r"he would",
                r"he'd've": r"he would have",
                r"he'll": r"he will",
                r"he'll've": r"he will have",
                r"he's": r"he is",
                r"how'd": r"how did",
                r"how'd'y": r"how do you",
                r"how'll": r"how will",
                r"how's": r"how is",
                r"i'd": r"i would",
                r"i'd've": r"i would have",
                r"i'll": r"i will",
                r"i'll've": r"i will have",
                r"i'm": "i am",
                r"i've": "i have",
                r"isn't": "is not",
                r"it'd": "it had",
                r"it'd've": "it would have",
                r"it'll": "it will",
                r"it'll've": "it will have",
                r"it's": "it is",
                r"let's": "let us",
                r"ma'am": "madam",
                r"mayn't": "may not",
                r"might've": "might have",
                r"mightn't": "might not",
                r"mightn't've": "might not have",
                r"must've": "must have",
                r"mustn't": "must not",
                r"mustn't've": "must not have",
                r"needn't": "need not",
                r"needn't've": "need not have",
                r"o'clock": "of the clock",
                r"oughtn't": "ought not",
                r"oughtn't've": "ought not have",
                r"shan't": "shall not",
                r"sha'n't": "shall not",
                r"shan't've": "shall not have",
                r"she'd": "she would",
                r"she'd've": "she would have",
                r"she'll": "she will",
                r"she'll've": "she will have",
                r"she's": "she is",
                r"should've": "should have",
                r"shouldn't": "should not",
                r"shouldn't've": "should not have",
                r"so've": "so have",
                r"so's": "so is",
                r"that'd": "that would",
                r"that'd've": "that would have",
                r"that's": "that is",
                r"there'd": "there had",
                r"there'd've": "there would have",
                r"there's": "there is",
                r"they'd": "they would",
                r"they'd've": "they would have",
                r"they'll": "they will",
                r"they'll've": "they will have",
                r"they're": "they are",
                r"they've": "they have",
                r"to've": "to have",
                r"wasn't": "was not",
                r"we'd": "we had",
                r"we'd've": "we would have",
                r"we'll": "we will",
                r"we'll've": "we will have",
                r"we're": "we are",
                r"we've": "we have",
                r"weren't": "were not",
                r"what'll": "what will",
                r"what'll've": "what will have",
                r"what're": "what are",
                r"what's": "what is",
                r"what've": "what have",
                r"when's": "when is",
                r"when've": "when have",
                r"where'd": "where did",
                r"where's": "where is",
                r"where've": "where have",
                r"who'll": "who will",
                r"who'll've": "who will have",
                r"who's": "who is",
                r"who've": "who have",
                r"why's": "why is",
                r"why've": "why have",
                r"will've": "will have",
                r"won't": "will not",
                r"won't've": "will not have",
                r"would've": "would have",
                r"wouldn't": "would not",
                r"wouldn't've": "would not have",
                r"y'all": "you all",
                r"y'alls": "you alls",
                r"y'all'd": "you all would",
                r"y'all'd've": "you all would have",
                r"y'all're": "you all are",
                r"y'all've": "you all have",
                r"you'd": "you had",
                r"you'd've": "you would have",
                r"you'll": "you you will",
                r"you'll've": "you you will have",
                r"you're": "you are",
                r"you've": "you have"
            }

c_re = re.compile('(%s)' % '|'.join(contractions.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return contractions[match.group(0)]
    return c_re.sub(replace, text)

for fname in os.listdir("customer review data/") :
    reviews = []
    #fname=os.listdir("customer review data/")[4]
    print fname
    labeled_aspects = []
    if fname != "Readme.txt" :
        fp = open('customer review data/'+fname , 'r')
        data = fp.read()
        data = data.split('[t]')
        data = data[1:]
        for rev in data :
            comments = rev.split('\n')
            comments = comments[1:len(comments)-1]
            for each_line in comments :
                temp = each_line.split('##')
                if len(temp) == 2 :
                    l_ , u_ , p_ = [] , [] , []
                    if temp[0].split() == [] :
                        temp = [temp[1]]
                        reviews.append(["","","",temp[0]])
                    elif temp[1].split() == [] :
                        temp = [temp[0]]
                        temp[0] = temp[0].split(',')
                        for j in temp[0] :
                            j = j.split('[')
                            if 'u]' in j :
                                labeled_aspects.append(j[0])
                                u_.append(j[0])
                            elif 'p]' in j :
                                labeled_aspects.append(j[0])
                                p_.append(j[0])
                            else :
                                labeled_aspects.append(j[0])
                                l_.append(j[0])
                        reviews.append([','.join(l_),','.join(u_),','.join(p_),""])
                    else :
                        temp[0] = temp[0].split(',')
                        for j in temp[0] :
                            j = j.split('[')
                            if 'u]' in j :
                                labeled_aspects.append(j[0])
                                u_.append(j[0])
                            elif 'p]' in j :
                                labeled_aspects.append(j[0])
                                p_.append(j[0])
                            else :
                                labeled_aspects.append(j[0])
                                l_.append(j[0])
                        reviews.append([','.join(l_),','.join(u_),','.join(p_),temp[1]])

    for abc in range (len(reviews)) :
        reviews[abc][3] = re.sub('\r' , "" , reviews[abc][3])
        reviews[abc][3] = expandContractions(reviews[abc][3].lower())
        text = word_tokenize(reviews[abc][3])
        grammar = nltk.pos_tag(text)
        grammar = map(list , grammar)
        
        temp_txt = []
        dn_use = []
        for ak in range (len(grammar)-1) :
            if ak not in dn_use :
                if (grammar[ak][1] in ["NN","NNS","NNP","NNPS"]) and (grammar[ak+1][1] in ["NN","NNS","NNP","NNPS"]) :
                    temp_txt.append(grammar[ak][0] + "_" + grammar[ak+1][0])
                    dn_use.append(ak+1)
                else :
                    temp_txt.append(grammar[ak][0])
        if (len(grammar)-1) not in dn_use :
            temp_txt.append(grammar[(len(grammar)-1)][0])

        reviews[abc][3] = ' '.join(temp_txt)

        print reviews[abc][3]
        print ("\n*****************")
        print ('Review number : ' + str(abc))
        print ("*****************\n")

        rf = open('temp.txt' , 'wb')
        rf.write("%s\n" % reviews[abc][3])
        rf.close()

        try :
            args = shlex.split(r'./lexparser.csh /home/rahul/Desktop/sem-6/topics-in-data-mining/project/temp.txt')
            p = sp.Popen(args , cwd = r'/usr/stanford-parser-python-r22186/3rdParty/stanford-parser/stanford-parser-2010-08-20/' , stdout = sp.PIPE).communicate()

            p = p[0].split('\n\n')
            for i in range (len(p)/2) :
                temp = []
                p[(2*i)+1] = p[(2*i)+1].split('\n')
                for dep in p[(2*i)+1] :
                    dep = dep.split('(')
                    temp1 = dep[1].split(',')
                    temp1[1] = temp1[1][1:-1]
                    temp2 = []
                    for wrd in temp1 :
                        wrd = wrd.split('-')
                        wrd = '-'.join(wrd[:-1])
                        temp2.append(wrd)
                    temp.append([dep[0]] + temp2)

            for i in range (len(temp)) :
                temp[i] = '####'.join(temp[i])

            temp = "######".join(temp)
            reviews[abc].append(temp)
        except :
            print "Cannot process review :"
            print "*****" , reviews[abc][3] , "*****"
            print "Continued with next review"

    with open('modified_review_files3/'+fname[:-4]+'.csv' , 'wb') as fp :
        a = csv.writer(fp , delimiter=',')
        a.writerows(reviews)

#print "Precision : " , (len(list(set(extracted_aspects) & set(labeled_aspects))) / float(len(extracted_aspects)))
#print "Recall : " , (len(list(set(extracted_aspects) & set(labeled_aspects))) / float(len(labeled_aspects)))
