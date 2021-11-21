#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
import warnings
warnings.simplefilter("ignore")


# In[2]:


corpus = pd.read_csv('data/train', names=["index", "type", "pos"], sep='\t', error_bad_lines=False, warn_bad_lines=False)


# In[3]:


# corpus['type'] = corpus['type'].str.lower()
corpus


# ### 1. Vocabulary Creation

# In[4]:


counts = pd.DataFrame(corpus['type'].value_counts())
# counts_df = pd.DataFrame({'type': counts.index, 'idx' :range(1, len(vocab) + 1), 'count': counts['type']})
counts.reset_index(inplace=True)
counts.columns = ['type', 'count']
counts.head()


# In[5]:


unknown_count = len(counts[counts['count']<=1])

vocab = {'type': ['<unk>'], 'index': [0], 'count': [unknown_count]}
cnt = 1

for _, row in counts.iterrows():
    if row['count'] > 1:
        vocab['type'].append(row['type'])
        vocab['index'].append(cnt)
        vocab['count'].append(row['count'])
        cnt+=1

vocab = pd.DataFrame(vocab)
vocab
# vocab = vocab[vocab.count!=1]
# drop_indexes = vocab[vocab['count']==1]].index
# vocab.drop(drop_indexes, inplace=True)


# In[6]:


vocab.to_csv('vocab.txt', sep='\t')


# ### I have selected the threshold as 1 to identify a word as unknown, my vocabulary size is 23183 and total occurences of 'unk' is 20011.

# ### 2. Model Learning

# In[7]:


vocab_set = set(vocab['type'])
vocab_set.remove('<unk>')


# In[8]:


emission_prob = {}
transition_prob = {}
pos_count = {}

for _, row in corpus.iterrows():
    print(f'\r{_}', end='')
    if row['type'] in vocab_set:
        tp = row['type']
    else:
        tp = '<unk>'
    pos_count[row['pos']] = pos_count.get(row['pos'], 0) + 1
    if tp not in emission_prob:
        emission_prob[tp] = {}
    emission_prob[tp][row['pos']] = emission_prob[tp].get(row['pos'], 0) + 1
    if row['index'] == 1:
        if '.' not in transition_prob:
            transition_prob['.'] = {}
        transition_prob['.'][row['pos']] = transition_prob['.'].get(row['pos'], 0) + 1
    else:
        if corpus.iloc[_-1]['pos'] not in transition_prob:
            transition_prob[corpus.iloc[_-1]['pos']] = {}
        transition_prob[corpus.iloc[_-1]['pos']][row['pos']] = transition_prob[corpus.iloc[_-1]['pos']].get(row['pos'], 0) + 1


# In[9]:


for k1 in emission_prob.keys():
    for k2 in emission_prob[k1].keys():
        emission_prob[k1][k2] /= pos_count[k2]
emission_prob


# In[10]:


for k1 in transition_prob.keys():
    for k2 in transition_prob[k1].keys():
        transition_prob[k1][k2] /= pos_count[k1]
transition_prob


# In[98]:


emission = {}
transition = {}

for _, row in corpus.iterrows():
    print(f'\r{_}', end='')
    if row['type'] in vocab_set:
        emission[(row['type'], row['pos'])] = emission.get((row['type'], row['pos']), 0) + 1
        if row['index'] == 1:
            transition[('.', row['pos'])] = transition.get(('.', row['pos']), 0) + 1
        else:
            transition[(row['pos'], corpus.iloc[_-1]['pos'])] = transition.get((row['pos'], corpus.iloc[_-1]['pos']), 0) + 1


# In[99]:


for k1 in emission.keys():
    emission[k1]/=pos_count[k1[1]]
    
for k1 in transition.keys():
    transition[k1]/=pos_count[k1[0]]


# In[108]:


emissions = {}
transitions = {}

for k in emission.keys():
    emissions[str((k))] = emission[k]

for k in transition.keys():
    transitions[str((k))] = transition[k]
hmm = {'emission': emissions, 'transition': transitions}

import json

with open('hmm.json', 'w') as f:
    json.dump(hmm, f)


# In[109]:


hmm


# In[ ]:


print(f'The number of emission and transition parameters are {len(hmm['emission'])} and {len(hmm['transition'])} respectively')


# ### 3. Greedy Decoding with HMM

# In[13]:


transition_prob['NNP']['VBZ']


# In[14]:


dev_corpus = pd.read_csv('data/dev', names=["index", "type", "pos"], sep='\t', error_bad_lines=False, warn_bad_lines=False)


# In[15]:


dev_corpus.head()


# In[16]:


len(dev_corpus)


# In[17]:


total = 0
correct = 0
prev = '.'

for idx, row in dev_corpus.iterrows():
    print(f'\r{idx}', end='')
    lst = []
    true_pos = row['pos']
    for k, v in transition_prob[prev].items():
        if row['type'] not in vocab_set:
            tp = '<unk>'
        else:
            tp = row['type']
#         if k not in emission_prob[tp]:
#             prob = 1 / (pos_count[k] + 1365)
#             num, den = 0, 0
#             for k1 in emission_prob.keys():
#                 for k2 in emission_prob[k1].keys():
#                     if k2 == k:
#                         num+=emission_prob[k1][k2]
#                         den+=1
#             ep = num / den
#             ep = unk_dist['unknown'][k]
#             lst.append((k, v*ep))
        if tp in emission_prob and k in emission_prob[tp]:
            lst.append((k, v*emission_prob[tp][k]))
        else:
            lst.append((k, 0))
    lst.sort(key=lambda x: x[1], reverse=True)
    if lst[0][0] == 0:
        tags = transition_prob[prev]
        vals = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        tag = vals[0][0]
    else:
        tag = lst[0][0]
    prev = tag
    total+=1
    if tag == true_pos:
        correct+=1


# In[18]:


print('Accuracy with Greedy HMM algorithm is {:.2f}'.format(correct/total*100))


# In[19]:


sum([len(v) for k, v in transition_prob.items()])


# <h4> Evaluating on test data </h5>

# In[20]:


test_corpus = pd.read_csv('data/test', names=["index", "type", "pos"], sep='\t', error_bad_lines=False, warn_bad_lines=False)

total = 0
correct = 0
prev = '.'

f = open('greedy.out', 'w')

for idx, row in test_corpus.iterrows():
    print(f'\r{idx}', end='')
    lst = []
#     true_pos = row['pos']
    for k, v in transition_prob[prev].items():
        if row['type'] not in vocab_set:
            tp = '<unk>'
        else:
            tp = row['type']
#         if k not in emission_prob[tp]:
#             prob = 1 / (pos_count[k] + 1365)
#             num, den = 0, 0
#             for k1 in emission_prob.keys():
#                 for k2 in emission_prob[k1].keys():
#                     if k2 == k:
#                         num+=emission_prob[k1][k2]
#                         den+=1
#             ep = num / den
#             ep = unk_dist['unknown'][k]
#             lst.append((k, v*ep))
        if tp in emission_prob and k in emission_prob[tp]:
            lst.append((k, v*emission_prob[tp][k]))
        else:
            lst.append((k, 0))
    lst.sort(key=lambda x: x[1], reverse=True)
    if lst[0][0] == 0:
        tags = transition_prob[prev]
        vals = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        tag = vals[0][0]
    else:
        tag = lst[0][0]
    prev = tag
    total+=1
    f.write(str(str(row['index']) + '\t' + row['type'] + '\t' + tag))
    f.write('\n')
f.close()
#     if tag == true_pos:
#         correct+=1


# In[ ]:





# In[ ]:





# ### 4. Viterbi Decoding with HMM

# In[76]:


dev_corpus.head()


# In[82]:


hmmDecode = {}
prevs = ['.']
true_pos = []
pred_pos = []
total = 0
correct = 0
cntr = 0

for idx, row in dev_corpus.iterrows():
    print(f'\r{idx}', end='')
    hmmDecode[cntr] = {}  
    true_pos.append(row['pos'])  
    curr_prevs = set()
    for prev in prevs:
        for k, v in transition_prob[prev].items():
            if row['type'] not in vocab_set:
                tp = '<unk>'
            else:
                tp = row['type']
            if k not in emission_prob[tp]:
                et = v / (pos_count[k] + 1365)
            else:
                et = v*emission_prob[tp][k]
            curr_prevs.add(k)
            if cntr == 0:
                hmmDecode[cntr][k] = [et, [k]]
            elif k not in hmmDecode[cntr]:
                prevProb = hmmDecode[cntr - 1][prev][0]
                currProb = et
                path = hmmDecode[cntr - 1][prev][1] + [k]
                hmmDecode[cntr][k] = [prevProb*currProb, path]
            else:
                prevProb = hmmDecode[cntr][k][0]
                currProb = hmmDecode[cntr - 1][prev][0]*et
                if currProb > prevProb:
                    path = hmmDecode[cntr - 1][prev][1] + [k]
                    hmmDecode[cntr][k] = [currProb, path]
    cntr+=1
    prevs = tuple(curr_prevs)
    if idx == len(dev_corpus) - 1:
        pred_pos = list(hmmDecode[cntr-1].values())[0][1]
        for i in range(len(true_pos)):
            total+=1
            if true_pos[i] == pred_pos[i]:
                correct+=1
        break
    if idx != 0 and dev_corpus.iloc[idx + 1]['index'] == 1:
        pred_pos = list(hmmDecode[cntr-1].values())[0][1]
        for i in range(len(true_pos) - 1):
            total+=1
#             if true_pos[i] == '.':
#                 correct+=1
            if true_pos[i] == pred_pos[i]:
                correct+=1
        hmmDecode = {}
        true_pos = []
        pred_pos = []
        prevs = ['.']
        cntr = 0


# In[83]:


print('Accuracy with Viterbi HMM algorithm is {:.2f}'.format(correct/total*100))


# In[84]:


# pos_list = {}
# for idx, row in dev_corpus.iterrows():
#     print(f'\r{idx}', end='')
#     if row['type'] not in pos_list:
#         pos_list[row['type']] = {row['pos']}
#     else:
#         pos_list[row['type']].add(row['pos'])
# pos_list


# In[85]:


test_corpus = pd.read_csv('data/test', names=["index", "type", "pos"], sep='\t', error_bad_lines=False, warn_bad_lines=False)

hmmDecode = {}
prevs = ['.']
pred_pos = []
final_preds = []
total = 0
correct = 0
cntr = 0
for idx, row in test_corpus.iterrows():
    print(f'\r{idx}', end='')
    hmmDecode[cntr] = {}  
    curr_prevs = set()
    for prev in prevs:
        for k, v in transition_prob[prev].items():
            if row['type'] not in vocab_set:
                tp = '<unk>'
            else:
                tp = row['type']
            if k not in emission_prob[tp]:
                et = v / (pos_count[k] + 1365)
            else:
                et = v*emission_prob[tp][k]
            curr_prevs.add(k)
            if cntr == 0:
                hmmDecode[cntr][k] = [et, [k]]
            elif k not in hmmDecode[cntr]:
                prevProb = hmmDecode[cntr - 1][prev][0]
                currProb = et
                path = hmmDecode[cntr - 1][prev][1] + [k]
                hmmDecode[cntr][k] = [prevProb*currProb, path]
            else:
                prevProb = hmmDecode[cntr][k][0]
                currProb = hmmDecode[cntr - 1][prev][0]*et
                if currProb > prevProb:
                    path = hmmDecode[cntr - 1][prev][1] + [k]
                    hmmDecode[cntr][k] = [currProb, path]

    cntr+=1
    prevs = tuple(curr_prevs)
    if idx == len(test_corpus) - 1:
        pred_pos = list(hmmDecode[cntr-1].values())[0][1]
        final_preds = final_preds + pred_pos
        break
    if idx != 0 and test_corpus.iloc[idx + 1]['index'] == 1:
        pred_pos = list(hmmDecode[cntr-1].values())[0][1]   
        if not cntr == len(pred_pos): 
            print(cntr == len(pred_pos))
        final_preds = final_preds + pred_pos
        hmmDecode = {}
        pred_pos = []
        prevs = ['.']
        cntr = 0


# In[86]:


len(final_preds)


# In[87]:


f = open('viterbi.out', 'w')
cnt = 0
for idx, row in test_corpus.iterrows():
    f.write(str(str(row['index']) + '\t' + row['type'] + '\t' + final_preds[idx]))
    f.write('\n')
f.close()


# In[57]:


# emission_dist = {}

# for k1 in emission_prob.keys():
#     for k2 in emission_prob[k1].keys():
#         if k2 not in emission_dist:
#             emission_dist[k2] = [0, 0]
#         emission_dist[k2][0]+=emission_prob[k1][k2]
#         emission_dist[k2][1]+=1

# # for k1 in unk_dist.keys():
# #     for k2 in unk_dist[k1].keys():
# #         unk_dist[k1][k2] /= pos_count[k2]

# emission_dist


# In[58]:


# transition_dist = {}

# for k1 in transition_prob.keys():
#     for k2 in transition_prob[k1].keys():
#         if k1 not in transition_dist:
#             transition_dist[k1] = [0, 0]
#         transition_dist[k1][0]+=transition_prob[k1][k2]
#         transition_dist[k1][1]+=1

# # for k1 in unk_dist.keys():
# #     for k2 in unk_dist[k1].keys():
# #         unk_dist[k1][k2] /= pos_count[k2]

# transition_dist


# In[59]:


# emission_dist = {}
# total_tags = 0
# for idx, row in corpus.iterrows():
#     print(f'\r{idx}', end='')
#     total_tags+=1
#     emission_dist[row['pos']] = emission_dist.get(row['pos'], 0) + 1
    
# for k1 in emission_dist.keys():
#     emission_dist[k1] /= total_tags

# emission_dist, total_tags


# In[ ]:





# In[ ]:




