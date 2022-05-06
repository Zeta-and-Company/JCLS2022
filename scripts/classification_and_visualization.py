# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:21:25 2021

@author: KeliDu
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import cross_val_score
from os.path import abspath
from os.path import join
from sklearn.feature_extraction.text import CountVectorizer
import random
from scipy import stats
from itertools import product
import warnings
warnings.filterwarnings("ignore")

os.chdir(r'C:\Workstation\Trier\Github\JCLS2022_submission')

parameters_lines = open(r'C:\Workstation\Trier\pydistinto\scripts\parameters.txt', 'r', encoding='utf-8').read().split('\n')
parameters = {}
for line in parameters_lines:
    line_split = line.split('=')
    parameters[line_split[0]] = line_split[1]
    
# You need to adapt these
corpus = parameters['corpus']
workdir = parameters['workdir']

# It is recommended to name your files and folders accordingly
datadir = abspath(os.path.join(corpus, os.pardir))
plaintextfolder = join(datadir, "corpus", "")
metadatafile = join(datadir, "metadata.csv")
stoplistfile = join(datadir, "stoplist.txt")

# It is recommended not to change these
outputdir = join(workdir, "output_" + os.path.basename(datadir))
segmentfolder = join(outputdir, "segments1000", "")
datafolder = join(outputdir, "results", "")
plotfolder = join(outputdir, "plots", "")

meta_df = pd.read_csv(metadatafile, sep='\t')
textnames = sorted([os.path.join(segmentfolder, fn) for fn in os.listdir(segmentfolder) if '.txt' in fn])
labels = []
for text in textnames:
    textname = os.path.basename(text)[:-4]
    subgenre = meta_df.loc[meta_df['idno'] == textname.split('-')[0]]['subgenre'].item()
    labels.append(subgenre)
    #group = textname.split('_')[0]
    #labels.append(group)

vectorizer = CountVectorizer(input="filename")
dtm = vectorizer.fit_transform(textnames)
vocab = vectorizer.get_feature_names()
dtm_df = pd.DataFrame(dtm.toarray(), columns=vocab)

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# classification of ELTeC-collections 

def run_classification (n, clf, measures, df_distinctive_words):
    if measure == 'eta_sg0':
        df_distinctive_words = df_distinctive_words[(df_distinctive_words['relfreqs1'] != 0) & (df_distinctive_words['relfreqs2'] != 0)]
    df_distinctive_words = df_distinctive_words.sort_values(by=measure, ascending=False)
    df_distinctive_words = df_distinctive_words.dropna()
    distinctive_words = pd.concat([df_distinctive_words['words'][:n], df_distinctive_words['words'][-n:]], axis=0)
    distinctive_df = dtm_df[distinctive_words]
    f1 = cross_val_score(clf, distinctive_df, labels, cv=10, scoring='f1_macro', n_jobs = 2)
    print(f1.mean())
    return f1

df_distinctive_words = pd.read_csv(r'C:\Workstation\Trier\Papers_submitted\JCLS_2022\tests_ELTeC\ELTeC-rom\output_ELTeC-rom\results\results_5000-lemmata-all_group_T4_1900-20-T2_1860-79.csv', sep='\t')
df_distinctive_words.rename(columns={'Unnamed: 0':'words'}, inplace=True)

ns = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
clf = svm.LinearSVC()
measures = ['zeta_sd0','zeta_sd2','rrf_dr0','welsh','ranksum','KL_Divergence','chi_square','LLR','tf_idf','eta_sg0']

f1_all = []
for measure in measures:
    for n in ns:
        f1 = run_classification (n, clf, measure, df_distinctive_words)
        for i in f1:
            f1_all.append((n, clf.__class__.__name__, measure, f1.mean(), f1.var(), i))

output_df = pd.DataFrame(f1_all, columns=['N', 'classifier', 'measure', 'f1_macro_mean', 'f1_macro_var', 'F1'])

#a quick check of results by Boxplot-Visualization
sns.set(font_scale=2)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (20,15))
g = sns.boxplot(x='N', y='F1', data=output_df, hue='measure', showfliers=True, palette="colorblind")
g.legend(title='measure', loc='best', bbox_to_anchor=(1, 1))

#save the classification results
output_df.to_csv(r'results\classification_results_rom.csv', sep='\t', index=False)

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#classification of french corpus 80s and 90s
df_scifi = pd.read_csv(r'tests_fr\corpus_fr\90s\results_90s\output_90s\results\results_5000-lemmata-all_blanche_no-yes.csv', sep='\t')
df_scifi.rename(columns={'Unnamed: 0':'words'}, inplace=True)

df_sentimental = pd.read_csv(r'tests_fr\corpus_fr\90s\results_90s\output_90s\results\results_5000-lemmata-all_sentimental_no-yes.csv', sep='\t')
df_sentimental.rename(columns={'Unnamed: 0':'words'}, inplace=True)

df_policier = pd.read_csv(r'tests_fr\corpus_fr\90s\results_90s\output_90s\results\results_5000-lemmata-all_policier_no-yes.csv', sep='\t')
df_policier.rename(columns={'Unnamed: 0':'words'}, inplace=True)

df_blanche = pd.read_csv(r'tests_fr\corpus_fr\90s\results_90s\output_90s\results\results_5000-lemmata-all_blanche_no-yes.csv', sep='\t')
df_blanche.rename(columns={'Unnamed: 0':'words'}, inplace=True)

ns = [5000, 4000, 3000, 2000, 1000, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
clfs = [svm.LinearSVC(), MultinomialNB(), LogisticRegression(), tree.DecisionTreeClassifier()]
measures = ['zeta_sd0','zeta_sd2','rrf_dr0','welsh','ranksum','KL_Divergence','chi_square','LLR','tf_idf','eta_sg0']

def get_all_features_df (measure, n, df_scifi, df_sentimental, df_policier, df_blanche):
    if measure == 'eta_sg0':
        df_scifi = df_scifi[(df_scifi['relfreqs1'] != 0) & (df_scifi['relfreqs2'] != 0)]
        df_sentimental = df_sentimental[(df_sentimental['relfreqs1'] != 0) & (df_sentimental['relfreqs2'] != 0)]
        df_policier = df_policier[(df_policier['relfreqs1'] != 0) & (df_policier['relfreqs2'] != 0)]
        df_blanche = df_blanche[(df_blanche['relfreqs1'] != 0) & (df_blanche['relfreqs2'] != 0)]
            
    df_scifi = df_scifi.sort_values(by=measure, ascending=False)
    df_scifi = df_scifi.dropna()
    scifi_words = pd.concat([df_scifi['words'][:n], df_scifi['words'][-n:]], axis=0)
    scifi_distinctive_df = dtm_df[scifi_words]
    
    df_sentimental = df_sentimental.sort_values(by=measure, ascending=False)
    df_sentimental = df_sentimental.dropna()
    sentimental_words = pd.concat([df_sentimental['words'][:n], df_sentimental['words'][-n:]], axis=0)
    sentimental_distinctive_df = dtm_df[sentimental_words]
    
    df_policier = df_policier.sort_values(by=measure, ascending=False)
    df_policier = df_policier.dropna()
    policier_words = pd.concat([df_policier['words'][:n], df_policier['words'][-n:]], axis=0)
    policier_distinctive_df = dtm_df[policier_words]
    
    df_blanche = df_blanche.sort_values(by=measure, ascending=False)
    df_blanche = df_blanche.dropna()
    blanche_words = pd.concat([df_blanche['words'][:n], df_blanche['words'][-n:]], axis=0)
    blanche_distinctive_df = dtm_df[blanche_words]
    all_features_df = pd.concat([scifi_distinctive_df, sentimental_distinctive_df, policier_distinctive_df, blanche_distinctive_df], axis=1)
    return all_features_df
    
f1_all = []
for measure in measures:
    for n in ns:
        all_features_df = get_all_features_df (measure, n, df_scifi, df_sentimental, df_policier, df_blanche)
        for clf in clfs:
            f1 = cross_val_score(clf, all_features_df, labels, cv=10, scoring='f1_macro', n_jobs = 2)
            for i in f1:
                f1_all.append((n, clf.__class__.__name__, measure, f1.mean(), f1.var(), i))
            print(f1.mean())


output_df = pd.DataFrame(f1_all, columns=['N', 'classifier', 'measure', 'f1_macro_mean', 'f1_macro_var', 'F1'])
output_df.to_csv(r'tests_fr\classification_results\classification_results_fra_90s.csv', sep='\t', index=False)

########################################################################################################################
########################################################################################################################
########################################################################################################################
#Visualizing classification results of four classifiers, french corpus 80s 

output_df = pd.read_csv(r'results\classification_results_fra_80s.csv', sep='\t')
output_df = output_df.loc[output_df['measure'] != 'KL_Divergence']

order = ['RRF', 'χ2', 'LLR', 'Welch', 'Wilcoxon', 'TF-IDF', 'Eta', 'Zeta_orig', 'Zeta_log']
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
g = sns.FacetGrid(output_df, col="classifier", col_wrap=2, height=5, aspect = 1.5)
g.map_dataframe(sns.pointplot, x="N", y="F1", hue="measure", ci=None, palette="colorblind", hue_order = order)
g.add_legend()
g.set_xticklabels(rotation=30)

########################################################################################################################
########################################################################################################################
########################################################################################################################
# Visualizing F1-distributions and baseline
language = 'fra_80s'
output_df = pd.read_csv(r'results\classification_results_' + language + '.csv', sep='\t')
output_df = output_df.loc[output_df['classifier'] == 'LogisticRegression']
output_df = output_df.loc[output_df['measure'] != 'KL_Divergence']
to_visual_random = pd.read_csv(r'results\random_words_classification_results_' + language + '.csv', sep='\t')
to_visual_random = to_visual_random.loc[to_visual_random['classifier'] == 'LogisticRegression']

output_df = output_df.sort_values(by=['f1_macro_mean'])
order = ['RRF', 'χ2', 'LLR', 'Welch', 'Wilcoxon', 'TF-IDF', 'Eta', 'Zeta_orig', 'Zeta_log']

sns.set(font_scale=2)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (15,15))
sns.pointplot(y="N", x="F1_macro_mean", data=to_visual_random, orient='h', color='green')
g = sns.boxplot(y='N', x='F1', data=output_df, hue='measure', orient='h', showfliers=True, palette="colorblind", hue_order = order)
g.legend(title='measure', loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=5)
ax.invert_yaxis()
ax.set_title('F1_distributions_' + language)
ax.set(xlim=(0, 1.02))

########################################################################################################################
########################################################################################################################
########################################################################################################################
#Significance test

ns = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
measures = ['RRF', 'χ2', 'LLR', 'Welch', 'Wilcoxon', 'TF-IDF', 'Eta', 'Zeta_orig', 'Zeta_log']

pairs_measure = list(product(measures, measures))
pairs_n = list(product(ns, ns))

language = 'fra_90s'
output_df = pd.read_csv(r'results\classification_results_' + language + '.csv', sep='\t')
output_df = output_df.loc[output_df['classifier'] == 'LinearSVC']


t_test_results_same_n = []
for n in ns:
    for pair in pairs_measure:
        f1s_1 = output_df.loc[(output_df['measure'] == pair[0]) & (output_df['N'] == n)]
        f1s_2 = output_df.loc[(output_df['measure'] == pair[1]) & (output_df['N'] == n)]
        t_test = stats.ttest_ind(f1s_1['F1'], f1s_2['F1'])
        t_test_results_same_n.append((n, pair, t_test))

t_test_results_same_measure = []
for measure in measures:
    for pair in pairs_n:
        f1s_1 = output_df.loc[(output_df['N'] == pair[0]) & (output_df['measure'] == measure)]
        f1s_2 = output_df.loc[(output_df['N'] == pair[1]) & (output_df['measure'] == measure)]
        t_test = stats.ttest_ind(f1s_1['F1'], f1s_2['F1'])
        t_test_results_same_measure.append((measure, pair, t_test))

#t-test results visualization
language = 'fra_90s'
t_test_results_df_same_n = pd.read_csv(r'results\significant_test_' + language + '_same_n.csv', sep='\t')
t_test_results_df_same_measure = pd.read_csv(r'results\significant_test_' + language + '_same_measure.csv', sep='\t')

sns.set(font_scale=2)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (16,10))
g = sns.boxplot(x="measure", y="pvalue", data=t_test_results_df_same_measure, showfliers=True, palette={"b"})
plt.xticks(rotation=30)
plt.axhline(y=0.05, color='black', linestyle='-')

sns.set(font_scale=2)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (16,10))
g = sns.boxplot(x="N", y="pvalue", data=t_test_results_df_same_n, showfliers=True, palette={"b"})
plt.xticks(rotation=30)
plt.axhline(y=0.05, color='black', linestyle='-')

#fra_80s results visualization N = 10, Figure 4a and 4b

#4a
language = 'fra_80s'
output_df = pd.read_csv(r'results\classification_results_' + language + '.csv', sep='\t')
output_df = output_df.loc[output_df['classifier'] == 'LogisticRegression']
output_df = output_df.loc[output_df['N'] == 10]

order = ['RRF', 'χ2', 'LLR', 'Welch', 'Wilcoxon', 'TF-IDF', 'Eta', 'Zeta_orig', 'Zeta_log']
sns.set(font_scale=2)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (15,15))
g = sns.boxplot(y='N', x='F1', data=output_df, hue='measure', orient='h', showfliers=True, palette="colorblind", hue_order = order)
ax.set_title('4a. F1_distributions_fra_80s N = 10')
ax.set(xlim=(0, 1.02))

#4b
t_test_results_df_n_10 = t_test_results_df_same_n.loc[t_test_results_df_same_n['N'] == 10]
t_test_confusion_matrix = t_test_results_df_n_10.pivot("measure_1", "measure_2", "pvalue")
t_test_confusion_matrix = t_test_confusion_matrix.reindex(index = ['RRF', 'χ2', 'LLR', 'Welch', 'Wilcoxon', 'TF-IDF', 'Eta', 'Zeta_orig', 'Zeta_log'])
t_test_confusion_matrix = t_test_confusion_matrix[['RRF', 'χ2', 'LLR', 'Welch', 'Wilcoxon', 'TF-IDF', 'Eta', 'Zeta_orig', 'Zeta_log']]

sns.set(font_scale=2)
sns.set_style("whitegrid")
f, ax1 = plt.subplots(figsize = (15,10))
sns.heatmap(t_test_confusion_matrix, annot=True, cmap="vlag", vmin=0, vmax=0.05, cbar_kws={'extend': 'max'}, ax=ax1, annot_kws={"fontsize":16})
ax1.set_title('4b. significant_test_fra_80s N = 10')


########################################################################################################################
########################################################################################################################
########################################################################################################################
#Visualization of other Figures which are not in paper but in Github

# Visualizing significant test between measures, N = 10
language = 'fra_90s'
t_test_results_df = pd.read_csv(r'results\significant_test_' + language + '_same_n_LogReg.csv', sep='\t')
t_test_results_df_n_10 = t_test_results_df.loc[t_test_results_df_same_n['N'] == 10]
t_test_confusion_matrix = t_test_results_df_n_10.pivot("measure_1", "measure_2", "pvalue")
t_test_confusion_matrix = t_test_confusion_matrix.reindex(index = ['RRF', 'χ2', 'LLR', 'Welch', 'Wilcoxon', 'TF-IDF', 'Eta', 'Zeta_orig', 'Zeta_log'])
t_test_confusion_matrix = t_test_confusion_matrix[['RRF', 'χ2', 'LLR', 'Welch', 'Wilcoxon', 'TF-IDF', 'Eta', 'Zeta_orig', 'Zeta_log']]

sns.set(font_scale=2)
sns.set_style("whitegrid")
f, ax1 = plt.subplots(figsize = (15,10))
sns.heatmap(t_test_confusion_matrix, annot=True, cmap="vlag", vmin=0, vmax=0.05, cbar_kws={'extend': 'max'}, ax=ax1, annot_kws={"fontsize":16})
ax1.set_title('significant_test_' + language +' N = 10')

# Visualizing F1-distributions and baseline
language = 'rom'
output_df = pd.read_csv(r'results\classification_results_' + language + '.csv', sep='\t')
output_df = output_df.loc[output_df['classifier'] == 'LinearSVC']
output_df = output_df.loc[output_df['measure'] != 'KL_Divergence']
to_visual_random = pd.read_csv(r'results\random_words_classification_results_' + language + '.csv', sep='\t')
to_visual_random = to_visual_random.loc[to_visual_random['classifier'] == 'LinearSVC']

output_df = output_df.sort_values(by=['f1_macro_mean'])
order = ['RRF', 'χ2', 'LLR', 'Welch', 'Wilcoxon', 'TF-IDF', 'Eta', 'Zeta_orig', 'Zeta_log']

sns.set(font_scale=2)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (15,15))
sns.pointplot(y="N", x="F1_macro_mean", data=to_visual_random, orient='h', color='green')
g = sns.boxplot(y='N', x='F1', data=output_df, hue='measure', orient='h', showfliers=True, palette="colorblind", hue_order = order)
g.legend(title='measure', loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=5)
ax.invert_yaxis()
ax.set_title('F1_distributions_' + language)
ax.set(xlim=(0, 1.02))





