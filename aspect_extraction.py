import re
import os
import csv
import nltk
import difflib
import numpy as np
import pandas as pd
import shlex , shutil
import subprocess as sp
from nltk import word_tokenize
from operator import itemgetter

# Collecting seed opinion words from the file
positive_op = [line.rstrip() for line in open("opinions/positive-words.txt")]
negative_op = [line.rstrip() for line in open("opinions/negative-words.txt")]
#print "Total seed opinions : " , len(opinions)

#print "Initial : " , len(opinions)
# Dependency Relations
def checkRule1(dependency) :
    prog = re.compile('.*amod.*|.*prep.*|.*nsubj.*|.*csubj.*|.*xsubj.*|.*dobj.*|.*iobj.* ')
    if prog.match(dependency) :
        return True
    return False

# Dependency Relations
def checkRule2(dependency) :
    prog = re.compile('.*amod.*|.*prep.*|.*nsubj.*|.*csubj.*|.*xsubj.*|.*dobj.*|.*conj.* ')
    if prog.match(dependency) :
        return True
    return False

# returns the matching dependency relation
def checkRule3(dependency) :
    if re.search("amod", dependency , re.IGNORECASE) :
        return "amod"
    elif re.search("prep", dependency , re.IGNORECASE) :
        return "prep"
    elif re.search("nsubj", dependency , re.IGNORECASE) :
        return "nsubj"
    elif re.search("csubj", dependency , re.IGNORECASE) :
        return "csubj"
    elif re.search("xsubj", dependency , re.IGNORECASE) :
        return "xsubj"
    elif re.search("dobj", dependency , re.IGNORECASE) :
        return "dobj"
    elif re.search("amod", dependency , re.IGNORECASE) :
        return "amod"
    elif re.search("iobj", dependency , re.IGNORECASE) :
        return "iobj"
    elif re.search("conj", dependency , re.IGNORECASE) :
        return "conj"

fnames = os.listdir("modified_review_files2")

for fname in fnames :
    """if os.path.exists(r'noun_phrases/'+fname[:-4]) :
        shutil.rmtree(r'noun_phrases/'+fname[:-4])
    os.makedirs(r'noun_phrases/'+fname[:-4])"""

    opinions = positive_op[36:] + negative_op[37:]
    reviews = []
    print "\n" + fname
    df = pd.read_csv("modified_review_files2/"+fname , sep = ',' , header = None)
    data = df.values
    tyur = list(data)
    data = np.asarray(data)
    for i in data :
        temp = []
        for j in i :
            if str(j) == "nan" :
                temp.append([])
            else :
                temp.append(j)
        reviews.append(temp)

    final_rev = []
    for i in range (len(reviews)) :
        if reviews[i][4] != [] :
            final_rev.append(reviews[i])

    # Extracting labeled aspects from the data
    labeled_aspects = []
    training_data = []
    test_data = []
    for i in range (len(final_rev)) :
        temp = []
        if final_rev[i][0] != [] :
            temp1 = final_rev[i][0].split(',')
            for j in temp1 :
                j = j.split()
                temp.append(' '.join(j))
        if final_rev[i][1] != [] :
            temp1 = final_rev[i][1].split(',')
            for j in temp1 :
                j = j.split()
                temp.append(' '.join(j))
        if final_rev[i][2] != [] :
            temp1 = final_rev[i][2].split(',')
            for j in temp1 :
                j = j.split()
                temp.append(' '.join(j))
        if temp != [] :
            labeled_aspects.append(temp)
            training_data.append(final_rev[i])
        else :
            test_data.append(final_rev[i])

    print len(training_data)
    # Pronoun resolution
    for i in range (len(training_data)) :
        temp1 = training_data[i][4].split("######")
        temp2 = []
        for j in temp1 :
            temp2.append(j.split("####"))

        text = word_tokenize(training_data[i][3])
        grammar = nltk.pos_tag(text)
        grammar = map(list , grammar)

        #print i
        for j in grammar :
            if j[1] in ['PRP','PRP$'] :
                if training_data[i][2] != [] :
                    for k in range (len(temp2)) :
                        if len(temp2[k]) == 3 :
                            if j[0].lower() == temp2[k][1].lower() :
                                temp2[k][1] = training_data[i][2].split()
                                temp2[k][1] = ' '.join(temp2[k][1])
                            elif j[0].lower() == temp2[k][2].lower() :
                                temp2[k][2] = training_data[i][2].split()
                                temp2[k][2] = ' '.join(temp2[k][2])
                        if len(temp2[k]) == 2 :
                            if j[0].lower() == temp2[k][1].lower() :
                                temp2[k][1] = training_data[i][2]

        training_data[i][4] = temp2

    def rm_dup(ex_ls) :
        temp_dic = {}
        for i in range (len(ex_ls)) :
            temp_dic[ex_ls[i][0]] = ex_ls[i][1]
        return map(list , temp_dic.items())

    def writing_file(aspects , rule_no) :
        write_to_file = []
        for i in range (len(aspects)) :
            tmp = []
            for j in range (len(aspects[i])) :
                tmp.append("##".join(aspects[i][j]))
            write_to_file.append(tmp)
        """with open('noun_phrases/' + fname[:-4] + '/' + rule_no + r'.csv' , 'wb') as fp :
            a = csv.writer(fp , delimiter=',')
            a.writerows(write_to_file)"""

    # Precision calculation
    def precision_and_recall(aspects) :
        temp_ext_asp = []
        temp_lab_asp = []
        pre_8 = {}
        for i in range (len(aspects)) :
            if aspects[i] != [] :
                for j in range (len(labeled_aspects[i])) :
                    if labeled_aspects[i][j] != "" :
                        labeled_aspects[i][j] = labeled_aspects[i][j].split()
                        if len(labeled_aspects[i][j]) == 2 :
                            labeled_aspects[i][j] = "_".join(labeled_aspects[i][j])
                        else :
                            labeled_aspects[i][j] = labeled_aspects[i][j][0]
                        temp_lab_asp.append(labeled_aspects[i][j])
                #temp_lab_asp += labeled_aspects[i]

        for i in range (len(aspects)) :
            if aspects[i] != [] :
                temp_ls = []
                for j in range (len(aspects[i])) :
                    if (len(aspects[i][j][0]) != 1) and (len(aspects[i][j][0]) != 2) :
                        temp_ls.append(aspects[i][j])
                temp_ext_asp += temp_ls

        temp_lab_asp = list(set(temp_lab_asp))
        temp_ext_asp = map(tuple , temp_ext_asp)
        temp_ext_asp = map(list , set(temp_ext_asp))

        for i in range (len(temp_ext_asp)) :
            if temp_ext_asp[i][1] in pre_8 :
                pre_8[temp_ext_asp[i][1]].append(temp_ext_asp[i][0])
            else :
                pre_8[temp_ext_asp[i][1]] = [temp_ext_asp[i][0]]

        tmp_store = []
        if temp_ext_asp != [] :
            for rel in pre_8 :
                tmp_store.append([rel , pre_8[rel]])
                correct_count = 0
                total_count = 0
                tmp_ls = pre_8[rel]
                for ext in tmp_ls :
                    for lab in temp_lab_asp :
                        if (difflib.SequenceMatcher(None , ext.lower() , lab.lower()).ratio()) > 0.75 :
                            correct_count += 1
                            break ;
                pre_8[rel] = [(correct_count / float(len(tmp_ls))) , (correct_count / float(len(temp_lab_asp)))]
                #print rel + " : Precision = " + str(pre_8[rel][0]) + ", Recall = " + str(pre_8[rel][1])

        return map(list , pre_8.items()) , tmp_store

    extracted_aspects = []  # Complete extracted aspects
    rule_eval_results = []  # storing all precision values
    labels_store = []   # Storing extracted labels of each a every rule

    # Type 1 rules
    def type1_rules(training_data, rule_no) :
        extracted_aspects1, extracted_aspects2 = [], []
        for abc in range (len(training_data)) :
            temp = training_data[abc][4]

            text = word_tokenize(training_data[abc][3])
            grammar = nltk.pos_tag(text)
            grammar = map(list , grammar)
            hashmap1 = {}
            for g in grammar :
                hashmap1[g[0]] = g[1]

            # Dependency Rule 1-1
            temp_aspects = []
            for dep in temp :
                if (checkRule1(dep[0])) and (len(dep) == 3) :
                    if (dep[1] in hashmap1) and (hashmap1[dep[1]] in ["NN" , "NNS" , "NNP" , "NNPS"]) :
                        if (dep[2].lower() in opinions) :
                            if extracted_aspects != [] :
                                if (dep[1] not in temp_aspects) and (dep[1] not in list(np.asarray(extracted_aspects)[:,0])) :
                                    temp_aspects.append([dep[1], checkRule3(dep[0])])
                            else :
                                if (dep[1] not in temp_aspects) :
                                    temp_aspects.append([dep[1], checkRule3(dep[0])])
                    elif (dep[2] in hashmap1) and (hashmap1[dep[2]] in ["NN" , "NNS" , "NNP" , "NNPS"]) :
                        if (dep[1].lower() in opinions) :
                            if extracted_aspects != [] :
                                if (dep[2] not in temp_aspects) and (dep[2] not in list(np.asarray(extracted_aspects)[:,0])) :
                                    temp_aspects.append([dep[2], checkRule3(dep[0])])
                            else :
                                if (dep[2] not in temp_aspects) :
                                    temp_aspects.append([dep[2], checkRule3(dep[0])])
            extracted_aspects1.append(temp_aspects)

            # Dependency Rule 1-2
            temp_aspects = []
            for i in range (len(temp)) :
                if (checkRule1(temp[i][0])) and (len(temp[i]) == 3) :
                    if (temp[i][1].lower() in opinions) :
                        for j in range (len(temp)) :
                            if j != i :
                                if (checkRule1(temp[j][0])) and (len(temp[j]) == 3) :
                                    if (temp[j][1] == temp[i][2]) and (temp[j][2] in hashmap1) and (hashmap1[temp[j][2]] in ["NN" , "NNS" , "NNP" , "NNPS"]) :
                                        if extracted_aspects != [] :
                                            if (temp[j][2] not in temp_aspects) and (temp[j][2] not in list(np.asarray(extracted_aspects)[:,0])) :
                                                temp_aspects.append([temp[j][2], checkRule3(temp[j][0])])
                                                #print temp[i][0] , temp[i][1] , temp[j][0] , temp[j][1] , temp[j][2]
                                                #print training_data[abc][3]
                                        else :
                                            if (temp[j][2] not in temp_aspects) :
                                                temp_aspects.append([temp[j][2], checkRule3(temp[j][0])])
                                                #print training_data[abc][3]
                                    elif (temp[j][2] == temp[i][2]) and (temp[j][1] in hashmap1) and (hashmap1[temp[j][1]] in ["NN" , "NNS" , "NNP" , "NNPS"]) :
                                        if extracted_aspects != [] :
                                            if (temp[j][1] not in temp_aspects) and (temp[j][1] not in list(np.asarray(extracted_aspects)[:,0])) :
                                                temp_aspects.append([temp[j][1], checkRule3(temp[j][0])])
                                                #print training_data[abc][3]
                                        else :
                                            if (temp[j][1] not in temp_aspects) :
                                                temp_aspects.append([temp[j][1], checkRule3(temp[j][0])])
                                                #print training_data[abc][3]
                    elif (temp[i][2].lower() in opinions) :
                        for j in range (len(temp)) :
                            if j != i :
                                if (checkRule1(temp[j][0])) and (len(temp[j]) == 3) :
                                    if (temp[j][1] == temp[i][1]) and (temp[j][2] in hashmap1) and (hashmap1[temp[j][2]] in ["NN" , "NNS" , "NNP" , "NNPS"]) :
                                        if extracted_aspects != [] :
                                            if (temp[j][2] not in temp_aspects) and (temp[j][2] not in list(np.asarray(extracted_aspects)[:,0])) :
                                                temp_aspects.append([temp[j][2], checkRule3(temp[j][0])])
                                                #print training_data[abc][3]
                                        else :
                                            if (temp[j][2] not in temp_aspects) :
                                                temp_aspects.append([temp[j][2], checkRule3(temp[j][0])])
                                                #print training_data[abc][3]
                                    elif (temp[j][2] == temp[i][1]) and (temp[j][1] in hashmap1) and (hashmap1[temp[j][1]] in ["NN" , "NNS" , "NNP" , "NNPS"]) :
                                        if extracted_aspects != [] :
                                            if (temp[j][1] not in temp_aspects) and (temp[j][1] not in list(np.asarray(extracted_aspects)[:,0])) :
                                                temp_aspects.append([temp[j][1], checkRule3(temp[j][0])])
                                                #print training_data[abc][3]
                                        else :
                                            if (temp[j][1] not in temp_aspects) :
                                                temp_aspects.append([temp[j][1], checkRule3(temp[j][0])])
                                                #print training_data[abc][3]
            extracted_aspects2.append(temp_aspects)
        return extracted_aspects1, extracted_aspects2

    extracted_aspects1 , extracted_aspects2 = type1_rules(training_data , 1)

    for i in range (len(extracted_aspects2)) :
        extracted_aspects += extracted_aspects1[i]
        extracted_aspects += extracted_aspects2[i]

    extracted_aspects = rm_dup(extracted_aspects)

    print "Type 1 rules"
    print "R1-1 : "
    rule_eval_results.append(precision_and_recall(extracted_aspects1))
    writing_file(extracted_aspects1 , '1_1')
    print "R1-2 : "
    rule_eval_results.append(precision_and_recall(extracted_aspects2))
    writing_file(extracted_aspects2 , '1_2')

    # Type 2 rules
    def type2_rules(training_data) :
        extracted_aspects3, extracted_aspects4 = [], []
        for abc in range (len(training_data)) :
            temp = training_data[abc][4]

            text = word_tokenize(training_data[abc][3])
            grammar = nltk.pos_tag(text)
            grammar = map(list , grammar)
            hashmap1 = {}
            for g in grammar :
                hashmap1[g[0]] = g[1]

            # Dependency Rule 3-1
            temp_aspects = []
            for dep in temp :
                if (re.search('conj' , dep[0] , re.IGNORECASE)) and (len(dep) == 3) :
                    if (dep[1] in list(np.asarray(extracted_aspects)[:,0])) and (dep[2] in hashmap1) and (hashmap1[dep[2]] in ["NN" , "NNS" , "NNP" , "NNPS"]) :
                        if (dep[2] not in list(np.asarray(extracted_aspects)[:,0])) and (dep[2] not in temp_aspects) :
                            temp_aspects.append([dep[2] , "conj"])
                    elif (dep[2] in list(np.asarray(extracted_aspects)[:,0])) and (dep[1] in hashmap1) and (hashmap1[dep[1]] in ["NN" , "NNS" , "NNP" , "NNPS"]) :
                        if (dep[1] not in list(np.asarray(extracted_aspects)[:,0])) and (dep[2] not in temp_aspects) :
                            temp_aspects.append([dep[1] , "conj"])
            extracted_aspects3.append(temp_aspects)

            # Dependency Rule 3-2
            temp_aspects = []
            for i in range (len(temp)) :
                if (checkRule2(temp[i][0])) and (len(temp[i]) == 3) :
                    if (temp[i][1] in list(np.asarray(extracted_aspects)[:,0])) :
                        for j in range (len(temp)) :
                            if j != i :
                                if (checkRule2(temp[j][0])) and (len(temp[j]) == 3) :
                                    if (temp[j][1] == temp[i][2]) and (temp[j][2] in hashmap1) and (hashmap1[temp[j][2]] in ["NN" , "NNS" , "NNP" , "NNPS"]) :
                                        if temp[j][2] not in list(np.asarray(extracted_aspects)[:,0]) :
                                            temp_aspects.append([temp[j][2] , checkRule3(temp[j][0])])
                                    elif (temp[j][2] == temp[i][2]) and (temp[j][1] in hashmap1) and (hashmap1[temp[j][1]] in ["NN" , "NNS" , "NNP" , "NNPS"]) :
                                        if temp[j][1] not in list(np.asarray(extracted_aspects)[:,0]) :
                                            temp_aspects.append([temp[j][1] , checkRule3(temp[j][0])])
                    elif (temp[i][2] in list(np.asarray(extracted_aspects)[:,0])) :
                        for j in range (len(temp)) :
                            if j != i :
                                if (checkRule2(temp[j][0])) and (len(temp[j]) == 3) :
                                    if (temp[j][1] == temp[i][1]) and (temp[j][2] in hashmap1) and (hashmap1[temp[j][2]] in ["NN" , "NNS" , "NNP" , "NNPS"]) :
                                        if temp[j][2] not in list(np.asarray(extracted_aspects)[:,0]) :
                                            temp_aspects.append([temp[j][2] , checkRule3(temp[j][0])])
                                    elif (temp[j][2] == temp[i][1]) and (temp[j][1] in hashmap1) and (hashmap1[temp[j][1]] in ["NN" , "NNS" , "NNP" , "NNPS"]) :
                                        if temp[j][1] not in list(np.asarray(extracted_aspects)[:,0]) :
                                            temp_aspects.append([temp[j][1] , checkRule3(temp[j][0])])
            extracted_aspects4.append(temp_aspects)
        return extracted_aspects3, extracted_aspects4

    extracted_aspects3 = []
    extracted_aspects4 = []
    extracted_aspects3, extracted_aspects4 = type2_rules(training_data)

    for i in range (len(extracted_aspects3)) :
        extracted_aspects += extracted_aspects3[i]
        extracted_aspects += extracted_aspects4[i]

    extracted_aspects = rm_dup(extracted_aspects)

    print "Type 2 rules"
    print "R3-1"
    rule_eval_results.append(precision_and_recall(extracted_aspects3))
    writing_file(extracted_aspects3 , '3_1')
    print "R3-2"
    rule_eval_results.append(precision_and_recall(extracted_aspects4))
    writing_file(extracted_aspects4 , '3_2')

    # Type 3 rules
    print "Type 3 rules"
    for abc in range (len(training_data)) :
        temp = training_data[abc][4]

        text = word_tokenize(training_data[abc][3])
        grammar = nltk.pos_tag(text)
        grammar = map(list , grammar)
        hashmap1 = {}
        for g in grammar :
            hashmap1[g[0]] = g[1]

        # Dependency Rule 2-1
        for dep in temp :
            if (checkRule1(dep[0])) and (len(dep) == 3) :
                if (dep[1] in list(np.asarray(extracted_aspects)[:,0])) :
                    if (dep[2] in hashmap1) and (hashmap1[dep[2]] == 'JJ') :
                        if dep[2].lower() not in opinions :
                            opinions.append(dep[2].lower())
                elif (dep[2] in list(np.asarray(extracted_aspects)[:,0])) :
                    if (dep[1] in hashmap1) and (hashmap1[dep[1]] == 'JJ') :
                        if dep[1].lower() not in opinions :
                            opinions.append(dep[1].lower())

    extracted_aspects5 , extracted_aspects6 = type1_rules(training_data , 3)
    extracted_aspects7 , extracted_aspects8 = type2_rules(training_data)

    for i in range (len(extracted_aspects5)) :
        extracted_aspects += extracted_aspects5[i]
        extracted_aspects += extracted_aspects6[i]
        extracted_aspects += extracted_aspects7[i]
        extracted_aspects += extracted_aspects8[i]

    extracted_aspects = rm_dup(extracted_aspects)

    temp_asp = []
    for i in range (len(extracted_aspects5)) :
        temp_asp.append(extracted_aspects5[i] + extracted_aspects6[i] + extracted_aspects7[i] + extracted_aspects8[i])
    print "R2-1"
    rule_eval_results.append(precision_and_recall(temp_asp))
    writing_file(temp_asp , '2_1')

    for abc in range (len(training_data)) :
        temp = training_data[abc][4]

        text = word_tokenize(training_data[abc][3])
        grammar = nltk.pos_tag(text)
        grammar = map(list , grammar)
        hashmap1 = {}
        for g in grammar :
            hashmap1[g[0]] = g[1]

        # Dependency Rule 2-2
        for i in range (len(temp)) :
            if (checkRule1(temp[i][0])) and (len(temp[i]) == 3) :
                if (temp[i][1] in list(np.asarray(extracted_aspects)[:,0])) :
                    for j in range (len(temp)) :
                        if j != i :
                            if (checkRule1(temp[j][0])) and (len(temp[j]) == 3) :
                                if (temp[j][1] == temp[i][2]) and (temp[j][2] in hashmap1) and (hashmap1[temp[j][2]] == "JJ") :
                                    if temp[j][2].lower() not in opinions :
                                        opinions.append(temp[j][2].lower())
                                elif (temp[j][2] == temp[i][2]) and (temp[j][1] in hashmap1) and (hashmap1[temp[j][1]] == "JJ") :
                                    if temp[j][1].lower() not in opinions :
                                        opinions.append(temp[j][1].lower())
                elif (temp[i][2] in list(np.asarray(extracted_aspects)[:,0])) :
                    for j in range (len(temp)) :
                        if j != i :
                            if (checkRule1(temp[j][0])) and (len(temp[j]) == 3) :
                                if (temp[j][1] == temp[i][1]) and (temp[j][2] in hashmap1) and (hashmap1[temp[j][2]] == "JJ") :
                                    if temp[j][2].lower() not in opinions :
                                        opinions.append(temp[j][2].lower())
                                elif (temp[j][2] == temp[i][1]) and (temp[j][1] in hashmap1) and (hashmap1[temp[j][1]] == "JJ") :
                                    if temp[j][1].lower() not in opinions :
                                        opinions.append(temp[j][1].lower())

    extracted_aspects5 , extracted_aspects6 = type1_rules(training_data , 3)
    extracted_aspects7 , extracted_aspects8 = type2_rules(training_data)

    for i in range (len(extracted_aspects5)) :
        extracted_aspects += extracted_aspects5[i]
        extracted_aspects += extracted_aspects6[i]
        extracted_aspects += extracted_aspects7[i]
        extracted_aspects += extracted_aspects8[i]

    extracted_aspects = rm_dup(extracted_aspects)

    temp_asp = []
    for i in range (len(extracted_aspects5)) :
        temp_asp.append(extracted_aspects5[i] + extracted_aspects6[i] + extracted_aspects7[i] + extracted_aspects8[i])
    print "R2-2"
    rule_eval_results.append(precision_and_recall(temp_asp))
    writing_file(temp_asp , '2_2')

    for abc in range (len(training_data)) :
        temp = training_data[abc][4]

        text = word_tokenize(training_data[abc][3])
        grammar = nltk.pos_tag(text)
        grammar = map(list , grammar)
        hashmap1 = {}
        for g in grammar :
            hashmap1[g[0]] = g[1]

        # Dependency Rule 4-1
        for dep in temp :
            if (re.search('conj' , dep[0] , re.IGNORECASE)) :
                if (dep[1].lower() in opinions) and (dep[2] in hashmap1) and (hashmap1[dep[2]] == "JJ") :
                    if dep[2].lower() not in opinions :
                        opinions.append(dep[2].lower())
                elif (dep[2].lower() in opinions) and (dep[1] in hashmap1) and (hashmap1[dep[1]] == "JJ") :
                    if dep[1].lower() not in opinions :
                        opinions.append(dep[1].lower())

    extracted_aspects5 , extracted_aspects6 = type1_rules(training_data , 3)
    extracted_aspects7 , extracted_aspects8 = type2_rules(training_data)

    for i in range (len(extracted_aspects5)) :
        extracted_aspects += extracted_aspects5[i]
        extracted_aspects += extracted_aspects6[i]
        extracted_aspects += extracted_aspects7[i]
        extracted_aspects += extracted_aspects8[i]

    extracted_aspects = rm_dup(extracted_aspects)

    temp_asp = []
    for i in range (len(extracted_aspects5)) :
        temp_asp.append(extracted_aspects5[i] + extracted_aspects6[i] + extracted_aspects7[i] + extracted_aspects8[i])
    print "R4-1"
    rule_eval_results.append(precision_and_recall(temp_asp))
    writing_file(temp_asp , '4_1')

    for abc in range (len(training_data)) :
        temp = training_data[abc][4]

        text = word_tokenize(training_data[abc][3])
        grammar = nltk.pos_tag(text)
        grammar = map(list , grammar)
        hashmap1 = {}
        for g in grammar :
            hashmap1[g[0]] = g[1]

        # Dependency Rule 4-2
        for i in range (len(temp)) :
            if (checkRule2(temp[i][0])) and (len(temp[i]) == 3) :
                if (temp[i][1].lower() in opinions) :
                    for j in range (len(temp)) :
                        if j != i :
                            if (checkRule2(temp[j][0])) and (len(temp[j]) == 3) :
                                if (temp[j][1] == temp[i][2]) and (temp[j][2] in hashmap1) and (hashmap1[temp[j][2]] == "JJ") :
                                    if temp[j][2].lower() not in opinions :
                                        opinions.append(temp[j][2].lower())
                                elif (temp[j][2] == temp[i][2]) and (temp[j][1] in hashmap1) and (hashmap1[temp[j][1]] == "JJ") :
                                    if temp[j][1].lower() not in opinions :
                                        opinions.append(temp[j][1].lower())
                elif (temp[i][2].lower() in opinions) :
                    for j in range (len(temp)) :
                        if j != i :
                            if (checkRule2(temp[j][0])) and (len(temp[j]) == 3) :
                                if (temp[j][1] == temp[i][1]) and (temp[j][2] in hashmap1) and (hashmap1[temp[j][2]] == "JJ") :
                                    if temp[j][2].lower() not in opinions :
                                        opinions.append(temp[j][2].lower())
                                elif (temp[j][2] == temp[i][1]) and (temp[j][1] in hashmap1) and (hashmap1[temp[j][1]] == "JJ") :
                                    if temp[j][1].lower() not in opinions :
                                        opinions.append(temp[j][1].lower())

    extracted_aspects5 , extracted_aspects6 = type1_rules(training_data , 3)
    extracted_aspects7 , extracted_aspects8 = type2_rules(training_data)

    for i in range (len(extracted_aspects5)) :
        extracted_aspects += extracted_aspects5[i]
        extracted_aspects += extracted_aspects6[i]
        extracted_aspects += extracted_aspects7[i]
        extracted_aspects += extracted_aspects8[i]

    extracted_aspects = rm_dup(extracted_aspects)

    temp_asp = []
    for i in range (len(extracted_aspects5)) :
        temp_asp.append(extracted_aspects5[i] + extracted_aspects6[i] + extracted_aspects7[i] + extracted_aspects8[i])
    print "R4-2"
    rule_eval_results.append(precision_and_recall(temp_asp))
    writing_file(temp_asp , '4_2')

    #print "Final opinions : " , len(opinions)
    print "Rule evaluation is Completed"

    for i in range (len(rule_eval_results)) :
        labels_store.append(rule_eval_results[i][1])
        rule_eval_results[i] = rule_eval_results[i][0]

    # Rule Ranking Part
    for i in range (len(rule_eval_results)) :
        temp_ls = rule_eval_results[i]
        if temp_ls != [] :
            temp_hash = {}
            for j in temp_ls :
                temp_hash[j[1][0]] = [j[0] , j[1][1]]
            temp_hash = sorted(map(list , temp_hash.items()) , reverse=True)
            temp_ls = []
            for j in range (len(temp_hash)) :
                if temp_hash[j][0] != 0 :
                    temp_ls.append([temp_hash[j][1][0] , [temp_hash[j][0] , temp_hash[j][1][1]]])
            rule_eval_results[i] = temp_ls

    for i in range (len(rule_eval_results)) :
        if rule_eval_results[i] != [] :
            for j in range (1 , len(rule_eval_results[i])-1) :
                for k in range (0,j) :
                    if (rule_eval_results[i][j-k-1][1][0] == rule_eval_results[i][j-k][1][0]) and (rule_eval_results[i][j-k-1][1][1] < rule_eval_results[i][j-k][1][1]) :
                        rule_eval_results[i][j-k-1], rule_eval_results[i][j-k] = rule_eval_results[i][j-k], rule_eval_results[i][j-k-1]
                    else :
                        break
    # Rule Ranking is completed

    # Rule selection method
    S = []
    scores = []
    temp_lab_asp = []
    temp_ext_set = []
    for i in range (len(labeled_aspects)) :
        temp_lab_asp += labeled_aspects[i]
    temp_lab_asp = list(set(temp_lab_asp))

    for i in range (2) :
        temp_hash = {}
        for j in labels_store[i] :
            temp_hash[j[0]] = j[1]

        for j in range (len(rule_eval_results[i])) :
            if rule_eval_results[i][j] != [] :
                S.append(rule_eval_results[i][j])
                temp_ext_set.append(temp_hash[rule_eval_results[i][j][0]])
                tmp_set2 = []
                for k in temp_ext_set :
                    tmp_set2 += k
                tmp_set2 = list(set(tmp_set2))
                correct_count = 0
                for ext in tmp_set2 :
                    for lab in temp_lab_asp :
                        if (difflib.SequenceMatcher(None , ext.lower() , lab.lower()).ratio()) > 0.75 :
                            correct_count += 1
                            break ;
                precision = (correct_count / float(len(tmp_set2)))
                recall = (correct_count / float(len(temp_lab_asp)))
                scores.append((2 * precision * recall) / float(precision + recall))

    S = S[:(scores.index(max(scores))+1)]
    scores = scores[:(scores.index(max(scores))+1)]
    temp_ext_set = temp_ext_set[:(scores.index(max(scores))+1)]

    for i in range (2 , 4) :
        temp_hash = {}
        for j in labels_store[i] :
            temp_hash[j[0]] = j[1]

        for j in range (len(rule_eval_results[i])) :
            if rule_eval_results[i][j] != [] :
                S.append(rule_eval_results[i][j])
                temp_ext_set.append(temp_hash[rule_eval_results[i][j][0]])
                tmp_set2 = []
                for k in temp_ext_set :
                    tmp_set2 += k
                tmp_set2 = list(set(tmp_set2))
                correct_count = 0
                for ext in tmp_set2 :
                    for lab in temp_lab_asp :
                        if (difflib.SequenceMatcher(None , ext.lower() , lab.lower()).ratio()) > 0.75 :
                            correct_count += 1
                            break ;
                precision = (correct_count / float(len(tmp_set2)))
                recall = (correct_count / float(len(temp_lab_asp)))
                scores.append((2 * precision * recall) / float(precision + recall))

    S = S[:(scores.index(max(scores))+1)]
    scores = scores[:(scores.index(max(scores))+1)]
    temp_ext_set = temp_ext_set[:(scores.index(max(scores))+1)]

    maxF = max(scores)

    for i in range (4 , 8) :
        temp_hash = {}
        for j in labels_store[i] :
            temp_hash[j[0]] = j[1]

        for j in range (len(rule_eval_results[i])) :
            if rule_eval_results[i][j] != [] :
                temp_ext_set.append(temp_hash[rule_eval_results[i][j][0]])
                tmp_set2 = []
                for k in temp_ext_set :
                    tmp_set2 += k
                tmp_set2 = list(set(tmp_set2))
                correct_count = 0
                for ext in tmp_set2 :
                    for lab in temp_lab_asp :
                        if (difflib.SequenceMatcher(None , ext.lower() , lab.lower()).ratio()) > 0.75 :
                            correct_count += 1
                            break ;
                precision = (correct_count / float(len(tmp_set2)))
                recall = (correct_count / float(len(temp_lab_asp)))
                f1_score = (2 * precision * recall) / float(precision + recall)
                if f1_score > maxF :
                    S.append(rule_eval_results[i][j])
                    maxF = f1_score
                else :
                    temp_ext_set = temp_ext_set[:-1]

    print "Final scores " , scores
