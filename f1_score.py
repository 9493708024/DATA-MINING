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

fld_names = os.listdir("nouns")

for fld_name in fld_names :
    print fld_name
    reviews = []
    df = pd.read_csv("modified_review_files2/"+fld_name+".csv" , sep = ',' , header = None)
    data = df.values
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
    labeled_aspects = {}
    training_data1 = []
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
            labeled_aspects[final_rev[i][3]] = temp
            training_data1.append(final_rev[i])
        else :
            test_data.append(final_rev[i])

    reviews = []
    df = pd.read_csv("modified_review_files3/"+fld_name+".csv" , sep = ',' , header = None)
    data = df.values
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
    training_data2 = []
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
            if final_rev[i][3] not in labeled_aspects :
                labeled_aspects[final_rev[i][3]] = temp
            else :
                labeled_aspects[final_rev[i][3]] += temp
                labeled_aspects[final_rev[i][3]] = list(set(labeled_aspects[final_rev[i][3]]))
            training_data2.append(final_rev[i])
        else :
            test_data.append(final_rev[i])

    # Precision calculation
    def precision_and_recall(aspects1 , aspects2) :
        temp_ext_asp = []
        temp_lab_asp = []
        pre_8 = {}
        for i in range (len(aspects1)) :
            temp = labeled_aspects[aspects1[i][1]]
            if aspects1[i] != [] :
                for j in range (len(temp)) :
                    if temp[j] != "" :
                        temp[j] = temp[j].split()
                        if len(temp[j]) == 2 :
                            temp[j] = "_".join(temp[j])
                        else :
                            temp[j] = temp[j][0]
                        temp_lab_asp.append(temp[j])
        for i in range (len(aspects2)) :
            temp = labeled_aspects[aspects2[i][1]]
            if aspects2[i] != [] :
                for j in range (len(temp)) :
                    if temp[j] != "" :
                        temp[j] = temp[j].split()
                        if len(temp[j]) == 2 :
                            temp[j] = "_".join(temp[j])
                        else :
                            temp[j] = temp[j][0]
                        temp_lab_asp.append(temp[j])

        for i in range (len(aspects1)) :
            if aspects1[i] != [] :
                temp_ls = []
                for j in range (len(aspects1[i][0])) :
                    if (len(aspects1[i][0][j][0]) != 1) and (len(aspects1[i][0][j][0]) != 2) :
                        temp_ls.append(aspects1[i][0][j])
                temp_ext_asp += temp_ls
        for i in range (len(aspects2)) :
            if aspects2[i] != [] :
                temp_ls = []
                for j in range (len(aspects2[i][0])) :
                    if (len(aspects2[i][0][j][0]) != 1) and (len(aspects2[i][0][j][0]) != 2) :
                        temp_ls.append(aspects2[i][0][j])
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

    rule_eval_results = []
    labels_store = []
    for i in [1,3,2,4] :
        for j in [1,2] :
            noun_aspects = []
            noun_phrase_aspects = []
            counter = 0
            for line in open("nouns/"+fld_name+"/"+str(i)+'_'+str(j)+".csv") :
                csv_row = line.split()
                if csv_row != [] :
                    csv_row = csv_row[0].split(',')
                    for asp in range (len(csv_row)) :
                        csv_row[asp] = csv_row[asp].split('##')
                noun_aspects.append([csv_row,training_data1[counter][3]])
                counter += 1
            counter = 0
            for line in open("noun_phrases/"+fld_name+"/"+str(i)+'_'+str(j)+".csv") :
                csv_row = line.split()
                if csv_row != [] :
                    csv_row = csv_row[0].split(',')
                    for asp in range (len(csv_row)) :
                        csv_row[asp] = csv_row[asp].split('##')
                noun_phrase_aspects.append([csv_row,training_data2[counter][3]])

            rule_eval_results.append(precision_and_recall(noun_aspects , noun_phrase_aspects))

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
    for i in labeled_aspects :
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
                #tmp_set2 = list(set(tmp_set2))
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
    #print "\n", len(S) , S
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
                #tmp_set2 = list(set(tmp_set2))
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
    #print "\n", len(S) , S
    scores = scores[:(scores.index(max(scores))+1)]
    temp_ext_set = temp_ext_set[:(scores.index(max(scores))+1)]

    maxF = max(scores)

    pre , rec = 0 , 0
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
                #tmp_set2 = list(set(tmp_set2))
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
                    pre = precision
                    rec = recall
                else :
                    temp_ext_set = temp_ext_set[:-1]

    #print "\n", len(S) , S
    print scores[-1:][0] * 100
