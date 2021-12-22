# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 23:17:03 2021

@author: KeliDu
"""
import os
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import warnings
warnings.filterwarnings("ignore")

os.chdir(r'/mnt/data/users/kelidu/JCLS/')

dtm_df = pd.read_hdf(r'dtm_90s_absolutefreqs.hd5')
labels = open(r'labels_90s.txt', 'r', encoding='utf-8').read().split('\n')

ns = [5000, 4000, 3000, 2000, 1000, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
clfs = [svm.LinearSVC(), MultinomialNB(), LogisticRegression(), tree.DecisionTreeClassifier()]

f1_all = []
count = 0
while count < 1000:
    for n in ns:
        df_sample = dtm_df.sample(n*4, axis=1)
        for clf in clfs:
            f1 = cross_val_score(clf, df_sample, labels, cv=10, scoring='f1_macro', n_jobs = 5)
            for i in f1:
                f1_all.append((n, count, clf.__class__.__name__, f1.mean(), f1.var(), i))
    count+=1
    
output_df = pd.DataFrame(f1_all, columns=['no_of_random_words/4', 'test_count', 'classifier', 'f1_macro_mean', 'f1_macro_var', 'f1'])

output_df.to_csv(r'random_words_classification_results_90s.csv', sep='\t', index=False)
