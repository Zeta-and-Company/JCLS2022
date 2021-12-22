# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 18:07:55 2021

@author: KeliDu
"""

import os
from shutil import copyfile
import pandas as pd
import random


lists = ['ELTeC-cze', 'ELTeC-deu', 'ELTeC-eng', 'ELTeC-fra', 'ELTeC-hun', 'ELTeC-por', 'ELTeC-rom']

sets_path = r'C:\Workstation\Trier\Github\ELTeC-Sets'
ELTec_sets = sorted([os.path.join(sets_path, fn) for fn in os.listdir(sets_path)])
test_sets = []
for name in ELTec_sets:
    if os.path.basename(name) in lists:
        test_sets.append(name)
    
output_path = r'C:\Workstation\Trier\Papers_submitted\JCLS_2022\ELTec_corpora'

for corpus in test_sets:
    metadata = pd.read_csv(corpus + '\\' + os.path.basename(corpus) + '_metadata.csv', sep=',')
    
    texts = sorted([os.path.join(corpus+'\\plain', fn) for fn in os.listdir(corpus+'\\plain')])
    
    group_1_meta = metadata.loc[metadata['time-slot'] == 'T2']
    group_2_meta = metadata.loc[metadata['time-slot'] == 'T4']
    
    filename_1 = random.sample(list(group_1_meta['filename']), 20)
    filename_2 = random.sample(list(group_2_meta['filename']), 20)
    
    directory = output_path + '\\' + os.path.basename(corpus) + '\\corpus'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    meta_output = []
    
    for text in filename_1:
        dst = directory + '\\1_' + os.path.basename(text) + '.txt'
        copyfile(os.path.join(corpus+'\\plain', text + '.txt'), dst)
        meta_output.append((os.path.basename(text), 'T2_1860-79'))
        
    for text in filename_2:
        dst = directory + '\\2_' + os.path.basename(text) + '.txt'
        copyfile(os.path.join(corpus+'\\plain', text + '.txt'), dst)
        meta_output.append((os.path.basename(text), 'T4_1900-20'))
        
    meta_df = pd.DataFrame(data=meta_output, columns=['idno', 'group'])
    meta_df.to_csv(os.path.dirname(directory)+'\\metadata.csv', sep='\t', index=False)
    
    
