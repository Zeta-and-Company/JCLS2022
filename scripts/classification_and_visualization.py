# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:21:25 2021

@author: KeliDu
"""

import os
import pandas as pd
#import numpy as np
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
#import random
import warnings
warnings.filterwarnings("ignore")

os.chdir(r'C:\Workstation\Trier\Papers_submitted\JCLS_2022')

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
    #subgenre = meta_df.loc[meta_df['idno'] == textname.split('-')[0]]['subgenre'].item()
    #labels.append(subgenre)
    group = textname.split('_')[0]
    labels.append(group)
    

with open(outputdir + '\\labels_cze.txt', 'w', encoding='utf-8') as fout:
    for label in labels:
        fout.write(label + '\n')


#vectorizer = CountVectorizer(input="filename", binary=True)
vectorizer = CountVectorizer(input="filename")
dtm = vectorizer.fit_transform(textnames)

vocab = vectorizer.get_feature_names()
dtm_df = pd.DataFrame(dtm.toarray(), columns=vocab)

#save dtm
dtmfile = outputdir + "\\dtm_cze_absolutefreqs.hd5"
dtm_df.to_hdf(dtmfile, key="df")

'''
#for french corpus 80s and 90s
df_scifi = pd.read_csv(r'corpus_fr\90s\results_90s\output_90s\results\results_5000-lemmata-all_scifi_no-yes.csv', sep='\t')
df_scifi.rename(columns={'Unnamed: 0':'words'}, inplace=True)

df_sentimental = pd.read_csv(r'corpus_fr\90s\results_90s\output_90s\results\results_5000-lemmata-all_sentimental_no-yes.csv', sep='\t')
df_sentimental.rename(columns={'Unnamed: 0':'words'}, inplace=True)

df_policier = pd.read_csv(r'corpus_fr\90s\results_90s\output_90s\results\results_5000-lemmata-all_policier_no-yes.csv', sep='\t')
df_policier.rename(columns={'Unnamed: 0':'words'}, inplace=True)

df_blanche = pd.read_csv(r'corpus_fr\90s\results_90s\output_90s\results\results_5000-lemmata-all_blanche_no-yes.csv', sep='\t')
df_blanche.rename(columns={'Unnamed: 0':'words'}, inplace=True)

ns = [5000, 4000, 3000, 2000, 1000, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
clfs = [svm.LinearSVC()]#, MultinomialNB(), LogisticRegression(), tree.DecisionTreeClassifier()]
measures = ['zeta_sd0','zeta_sd2','rrf_dr0','eta_sg0','welsh','ranksum','KL_Divergence','chi_square','LLR','tf_idf']

f1_all = []
for n in ns:
    for measure in measures:
        
        if measure != 'eta_sg0':
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
        else:
            df_scifi = df_scifi[(df_scifi['relfreqs1'] != 0) & (df_scifi['relfreqs2'] != 0)]
            df_scifi = df_scifi.sort_values(by=measure, ascending=False)
            df_scifi = df_scifi.dropna()
            scifi_words = pd.concat([df_scifi['words'][:n], df_scifi['words'][-n:]], axis=0)
            scifi_distinctive_df = dtm_df[scifi_words]
            
            df_sentimental = df_sentimental[(df_sentimental['relfreqs1'] != 0) & (df_sentimental['relfreqs2'] != 0)]
            df_sentimental = df_sentimental.sort_values(by=measure, ascending=False)
            df_sentimental = df_sentimental.dropna()
            sentimental_words = pd.concat([df_sentimental['words'][:n], df_sentimental['words'][-n:]], axis=0)
            sentimental_distinctive_df = dtm_df[sentimental_words]
            
            df_policier_1 = df_policier[(df_policier['relfreqs1'] != 0) & (df_policier['relfreqs2'] != 0)]
            df_policier = df_policier.sort_values(by=measure, ascending=False)
            df_policier = df_policier.dropna()
            policier_words = pd.concat([df_policier['words'][:n], df_policier['words'][-n:]], axis=0)
            policier_distinctive_df = dtm_df[policier_words]
            
            df_blanche = df_blanche[(df_blanche['relfreqs1'] != 0) & (df_blanche['relfreqs2'] != 0)]
            df_blanche = df_blanche.sort_values(by=measure, ascending=False)
            df_blanche = df_blanche.dropna()
            blanche_words = pd.concat([df_blanche['words'][:n], df_blanche['words'][-n:]], axis=0)
            blanche_distinctive_df = dtm_df[blanche_words]
        
        all_features_df = pd.concat([scifi_distinctive_df, sentimental_distinctive_df, policier_distinctive_df, blanche_distinctive_df], axis=1)
        
        for clf in clfs:
            f1 = cross_val_score(clf, all_features_df, labels, cv=10, scoring='f1_macro', n_jobs = 2)
            for i in f1:
                f1_all.append((n, clf.__class__.__name__, measure, f1.mean(), f1.var(), i))
'''

df_distinctive_words = pd.read_csv(r'C:\Workstation\Trier\Papers_submitted\JCLS_2022\tests_ELTeC\ELTeC-cze\output_ELTeC-cze\results\results_5000-lemmata-all_group_T4_1900-20-T2_1860-79.csv', sep='\t')
df_distinctive_words.rename(columns={'Unnamed: 0':'words'}, inplace=True)

ns = [5000, 4000, 3000, 2000, 1000, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
#clfs = [svm.LinearSVC()]#, MultinomialNB(), LogisticRegression(), tree.DecisionTreeClassifier()]
clf = svm.LinearSVC()
measures = ['zeta_sd0','zeta_sd2','rrf_dr0','eta_sg0','welsh','ranksum','KL_Divergence','chi_square','LLR','tf_idf']

f1_all = []
for n in ns:
    for measure in measures:
        if measure != 'eta_sg0':
            df_distinctive_words = df_distinctive_words.sort_values(by=measure, ascending=False)
            df_distinctive_words = df_distinctive_words.dropna()
            distinctive_words = pd.concat([df_distinctive_words['words'][:n], df_distinctive_words['words'][-n:]], axis=0)
            distinctive_df = dtm_df[distinctive_words]
        else:            
            df_distinctive_words = df_distinctive_words[(df_distinctive_words['relfreqs1'] != 0) & (df_distinctive_words['relfreqs2'] != 0)]
            df_distinctive_words = df_distinctive_words.sort_values(by=measure, ascending=False)
            df_distinctive_words = df_distinctive_words.dropna()
            distinctive_words = pd.concat([df_distinctive_words['words'][:n], df_distinctive_words['words'][-n:]], axis=0)
            distinctive_df = dtm_df[distinctive_words]
        #for clf in clfs:
        f1 = cross_val_score(clf, distinctive_df, labels, cv=10, scoring='f1_macro', n_jobs = 2)
        print(f1.mean())
        for i in f1:
            f1_all.append((n, clf.__class__.__name__, measure, f1.mean(), f1.var(), i))
   

output_df = pd.DataFrame(f1_all, columns=['no_of_distinctive_words', 'classifier', 'measure', 'f1_macro_mean', 'f1_macro_var', 'f1'])

output_df.to_csv(r'classification_results_cze.csv', sep='\t', index=False)

output_df1 = pd.read_csv(r'classification_results\fr_90s.csv', sep='\t')
output_df1 = output_df1.loc[output_df1['no_of_distinctive_words'] == 10]

cate1 = ['binary'] * len(output_df)
output_df['type'] = cate1

cate2 = ['absolute'] * len(output_df1)
output_df1['type'] = cate2

to_visual = pd.concat([output_df, output_df1])
to_visual = to_visual.astype({"classifier": str})
to_visual_1 = to_visual.loc[to_visual['classifier'] == 'MultinomialNB()']


df_classification_80s = pd.read_csv(r'tests_fr\classification_results\classification_results_fra_80s.csv', sep='\t')
df_classification_90s = pd.read_csv(r'tests_fr\classification_results\classification_results_fra_90s.csv', sep='\t')

df_random_baseline_80s = pd.read_csv(r'tests_fr\classification_results\random_words_classification_results_80s.csv', sep='\t')
df_random_baseline_90s = pd.read_csv(r'tests_fr\classification_results\random_words_classification_results_90s.csv', sep='\t')

to_visual_80s = df_classification_80s.loc[df_classification_80s['classifier'] == 'LinearSVC']
to_visual_random_80s = df_random_baseline_80s.loc[df_random_baseline_80s['classifier'] == 'LinearSVC']

to_visual_90s = df_classification_90s.loc[df_classification_90s['classifier'] == 'LinearSVC']
to_visual_random_90s = df_random_baseline_90s.loc[df_random_baseline_90s['classifier'] == 'LinearSVC']



'''
Signifikanztest
'''
from scipy import stats
from itertools import product

ns = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
measures = ['zeta_sd0','zeta_sd2','rrf_dr0','eta_sg0','welsh','ranksum','KL_Divergence','chi_square','LLR','tf_idf']

pairs = list(product(measures, ns))
pairs_1 = list(product(pairs, pairs))

pairs_measure = list(product(measures, measures))

pairs_n = list(product(ns, ns))



output_df = pd.read_csv(r'classification_results_deu.csv', sep='\t')
to_visual = output_df
all_t_test_results = []

'''
for pair in pairs_1:
    f1s_1 = to_visual.loc[(to_visual['measure'] == pair[0][0]) & (to_visual['no_of_distinctive_words'] == pair[0][1])]
    f1s_2 = to_visual.loc[(to_visual['measure'] == pair[1][0]) & (to_visual['no_of_distinctive_words'] == pair[1][1])]
    t_test = stats.ttest_ind(f1s_1['f1'], f1s_2['f1'])
    all_t_test_results.append((pair, t_test))
'''    

for n in ns:
    for pair in pairs_measure:
        f1s_1 = to_visual.loc[(to_visual['measure'] == pair[0]) & (to_visual['no_of_distinctive_words'] == n)]
        f1s_2 = to_visual.loc[(to_visual['measure'] == pair[1]) & (to_visual['no_of_distinctive_words'] == n)]
        t_test = stats.ttest_ind(f1s_1['f1'], f1s_2['f1'])
        all_t_test_results.append((n, pair, t_test))

for measure in measures:
    for pair in pairs_n:
        f1s_1 = to_visual.loc[(to_visual['no_of_distinctive_words'] == pair[0]) & (to_visual['measure'] == measure)]
        f1s_2 = to_visual.loc[(to_visual['no_of_distinctive_words'] == pair[1]) & (to_visual['measure'] == measure)]
        t_test = stats.ttest_ind(f1s_1['f1'], f1s_2['f1'])
        all_t_test_results.append((measure, pair, t_test))

    
t_test_results_df = pd.read_csv(r'results\significant_test_fra_90s_same_n.csv', sep='\t')
t_test_results_df = t_test_results_df.loc[(t_test_results_df['measure_1'] != 'KLD') & (t_test_results_df['measure_2'] != 'KLD')]

t_test_results_df = pd.read_csv(r'results\significant_test_fra_90s_same_measure.csv', sep='\t')
t_test_results_df = t_test_results_df.loc[t_test_results_df['measure'] != 'KLD']


'''
visualization
'''

sns.set(font_scale=2)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (16,10))
g = sns.boxplot(x="N", y="pvalue", data=t_test_results_df, showfliers=True, palette={"b"})
#g = sns.boxplot(x="measure", y="pvalue", data=t_test_results_df, showfliers=True, palette={"b"})
plt.xticks(rotation=30)
plt.axhline(y=0.05, color='black', linestyle='-')


t_test_results_df = pd.read_csv(r'results\significant_test_fra_90s_same_n.csv', sep='\t')
t_test_results_df = t_test_results_df.loc[(t_test_results_df['measure_1'] != 'KLD') & (t_test_results_df['measure_2'] != 'KLD')]
to_visual1 = t_test_results_df.loc[t_test_results_df['N'] == 10]
aa1 = to_visual1.pivot("measure_1", "measure_2", "pvalue")
#aa1 = aa1.reindex(index = ['RRF', 'χ2', 'LLR', 'Welch', 'Wilcoxon', 'TF-IDF', 'Eta', 'Zeta_orig', 'Zeta_log'])
#aa1 = aa1[['RRF', 'χ2', 'LLR', 'Welch', 'Wilcoxon', 'TF-IDF', 'Eta', 'Zeta_orig', 'Zeta_log']]

#sns.set(font_scale=1.5)
#sns.set_style("whitegrid")
#sns.clustermap(aa1, annot=True, cmap="YlGnBu_r", vmin=0, vmax=0.05, cbar_kws={'extend': 'max'}, figsize=(15, 15))

sns.set(font_scale=2)
sns.set_style("whitegrid")
f, ax1 = plt.subplots(figsize = (15,10))
sns.heatmap(aa1, annot=True, cmap="YlGnBu_r", vmin=0, vmax=0.05, cbar_kws={'extend': 'max'}, ax=ax1, annot_kws={"fontsize":16})
ax1.set_title('significant_test_fra_90s N = 10')
#ax1.set_title('4b. significant_test_fra_80s N = 10')

t_test_results_df = pd.read_csv(r'significant_test_eng_same_n.csv', sep='\t')
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(30, 20), sharex=False)
to_visual1 = t_test_results_df.loc[t_test_results_df['no_of_distinctive_words'] == 10]
aa1 = to_visual1.pivot("measure_1", "measure_2", "pvalue")
sns.heatmap(aa1, annot=True, cmap="YlGnBu", vmin=0, vmax=0.05, cbar_kws={'extend': 'max'}, ax=ax1)
ax1.set_title('no_of_distinctive_words = 10')

to_visual2 = t_test_results_df.loc[t_test_results_df['no_of_distinctive_words'] == 50]
aa2 = to_visual2.pivot("measure_1", "measure_2", "pvalue")
sns.heatmap(aa2, annot=True, cmap="YlGnBu", vmin=0, vmax=0.05, cbar_kws={'extend': 'max'}, ax=ax2)
ax2.set_title('no_of_distinctive_words = 50')

to_visual3 = t_test_results_df.loc[t_test_results_df['no_of_distinctive_words'] == 500]
aa3 = to_visual3.pivot("measure_1", "measure_2", "pvalue")
sns.heatmap(aa3, annot=True, cmap="YlGnBu", vmin=0, vmax=0.05, cbar_kws={'extend': 'max'}, ax=ax3)
ax3.set_title('no_of_distinctive_words = 500')

to_visual4 = t_test_results_df.loc[t_test_results_df['no_of_distinctive_words'] == 1000]
aa4 = to_visual4.pivot("measure_1", "measure_2", "pvalue")
sns.heatmap(aa4, annot=True, cmap="YlGnBu", vmin=0, vmax=0.05, cbar_kws={'extend': 'max'}, ax=ax4)
ax4.set_title('no_of_distinctive_words = 5000')

########################################################################################################################
########################################################################################################################
########################################################################################################################

# Visualizing F1-distributions and baseline
language = 'fra_90s'
output_df = pd.read_csv(r'results\classification_results_' + language + '.csv', sep='\t')
output_df = output_df.loc[output_df['classifier'] == 'LinearSVC']
output_df = output_df.loc[output_df['N'] == 10]
output_df = output_df.loc[output_df['measure'] != 'KLD']
to_visual_random = pd.read_csv(r'results\random_words_classification_results_' + language + '.csv', sep='\t')
to_visual_random = to_visual_random.loc[to_visual_random['classifier'] == 'LinearSVC']

output_df = output_df.sort_values(by=['F1_macro_mean'])
order = ['RRF', 'χ2', 'LLR', 'Welch', 'Wilcoxon', 'TF-IDF', 'Eta', 'Zeta_orig', 'Zeta_log']

sns.set(font_scale=2)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (15,15))
sns.pointplot(y="N", x="F1_macro_mean", data=to_visual_random, orient='h', color='green')
g = sns.boxplot(y='N', x='F1', data=output_df, hue='measure', orient='h', showfliers=True, palette="hls")#, hue_order = order)
#g = sns.pointplot(y='N', x='F1', data=output_df, hue='measure', ci=None, palette="hls")
#g.legend(title='measure', loc='best', bbox_to_anchor=(1, 1))#legend(title='OptimizeInterval', loc='lower left')
g.legend(title='measure', loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=5)
ax.invert_yaxis()
#ax.set_title('4a. F1_distributions_' + language + ' N = 10')
ax.set_title('F1_distributions_' + language)
ax.set(xlim=(0.2, 1.02))

########################################################################################################################
########################################################################################################################
########################################################################################################################

#Visualizing F1_mean
sns.set(font_scale=2)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (15,15))
#sns.pointplot(y="N", x="F1_macro_mean", data=to_visual_random, orient='h', color='green')
g = sns.pointplot(y='N', x='F1_macro_mean', data=output_df, hue='measure', orient='h', ci=None, palette="hls")
g.legend(title='measure', loc='lower center', bbox_to_anchor=(0.5, -0.21), ncol=5)
ax.invert_yaxis()
ax.set_title('F1_macro_mean_' + language)
ax.set(xlim=(0.2, 1.02))

########################################################################################################################
########################################################################################################################
########################################################################################################################

#Visualizing F1 for N = 10
to_visual = output_df.loc[output_df['N'] == 10]
sns.set(font_scale=2)
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize = (15,10))
g = sns.boxplot(y='N', x='F1', data=to_visual, hue='measure', orient='h', palette="hls")
g.legend(title='measure', loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=5)
ax.set_title('F1_distributions_' + language +' N = 10')
ax.set(xlim=(0.2, 1.02))

########################################################################################################################
########################################################################################################################
########################################################################################################################

sns.set(font_scale=1.5)
sns.set_style("whitegrid")
g = sns.factorplot(x="N", y="f1_macro_mean", hue="type", col="measure",
               height=5, aspect=1.5, hue_order=measures, col_wrap=2,
               data=output_df)
g.set_xticklabels(rotation=30)

sns.set(font_scale=1.5)
sns.set_style("whitegrid")
g = sns.catplot(x="no_of_distinctive_words", y="f1_macro_mean", hue="type", col="measure", data=to_visual_1, kind="point", dodge=True, height=5, aspect=1.5);
g.set_xticklabels(rotation=30)


sns.set(font_scale=1.5)
sns.set_style("whitegrid")
g = sns.catplot(x="no_of_distinctive_words", y="f1", hue="type", col="measure", col_wrap=3, data=to_visual_1, kind="box", height=5, aspect=1.5)
g.set_xticklabels(rotation=30)

########################################################################################################################
########################################################################################################################
########################################################################################################################

#Visualizing classification results of seven languages

CORPUS_PATH = r'C:\Workstation\Trier\Papers_submitted\JCLS_2022\results\eltec'
filenames = sorted([os.path.join(CORPUS_PATH, fn) for fn in os.listdir(CORPUS_PATH)])

all_dfs = []
for file in filenames:
    df = pd.read_csv(file, sep='\t')
    df['language'] = os.path.basename(file)[-7:-4]
    all_dfs.append(df)
    
all_eltec_df = pd.concat(all_dfs)    

sns.set(font_scale=1.5)
sns.set_style("whitegrid")
g = sns.FacetGrid(all_eltec_df, col="language", col_wrap=2, height=5, aspect = 1.5)
g.map_dataframe(sns.pointplot, x="N", y="F1", hue="measure", ci=None, palette="hls")
g.add_legend()
g.set_xticklabels(rotation=30)


########################################################################################################################
########################################################################################################################
########################################################################################################################
#Visualizing classification results of four classifiers /

output_df = pd.read_csv(r'results\classification_results_fra_90s.csv', sep='\t')
output_df = output_df.loc[output_df['measure'] != 'KLD']


sns.set(font_scale=1.5)
sns.set_style("whitegrid")
g = sns.FacetGrid(output_df, col="classifier", col_wrap=2, height=5, aspect = 1.5)
g.map_dataframe(sns.pointplot, x="N", y="F1", hue="measure", ci=None, palette="hls")
g.add_legend()
g.set_xticklabels(rotation=30)






