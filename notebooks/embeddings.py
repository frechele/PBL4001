#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np


# In[4]:


import torch
from bbert.model.bbert import BBERT
from bbert.data.instruction import Vocabulary, InstructionMapping

imap = InstructionMapping()
vmap = Vocabulary(imap)

model = BBERT(vmap).cuda()
model.load_state_dict(torch.load('bbert.pth'))
model.eval()


# In[5]:


token_emb = model.bert.embedding.token


# In[6]:


tokens = list(range(imap.size))
token_str = [imap.get_inst(tok) for tok in tokens]


# In[7]:


tokens_t = torch.tensor(tokens).long().cuda()


# In[8]:


tokens_t.shape


# In[9]:


token_embeddings = token_emb(tokens_t).cpu().detach().numpy()


# In[26]:


tsne = TSNE(learning_rate='auto', init='pca', n_components=2, n_jobs=-1)
transformed = tsne.fit_transform(token_embeddings)


# In[31]:


plt.figure(figsize=(12, 10))
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title('token embedding t-SNE', fontsize=20)
plt.grid(True)

xs = transformed[:, 0]
ys = transformed[:, 1]
plt.scatter(xs, ys, s=1)

insts = ['jz', 'mul', 'sub', 'div']
insts_idx = [token_str.index(inst) for inst in insts]
plt.scatter(xs[insts_idx], ys[insts_idx], c=['r', 'g', 'b', 'y'])


# # 여기부턴 BB Embedding

# In[2]:


import glob
import pickle


# In[3]:


bb_list = glob.glob('/data/pbl/data/pkl2/*.pkl')
print('size:', len(bb_list))


# In[4]:


A = np.arange(10000)
B = np.sin((A + 1) * np.pi / (2 * 10000 + 1))
C = np.where(A > 10000 * 0.01, 1, np.sin((A + 1) * np.pi / (2 * 10000 * 0.01 + 1)))
D1 = (1 - np.cos(np.pi * (A / 10000) / 0.48)) / 2.
D2 = (3 - np.cos(np.pi * (A / 10000) / 0.48 / (0.52 / 0.5))) / 4
D = np.where(A / 10000 > 0.48, D2, D1)
#plt.plot(A, C, label=r'$p=0.01$')
#plt.plot(A, B, c='orange', label=r'$p=1$')
plt.plot(A, D, label='Ours')
#plt.legend(loc='best', fontsize=15)
plt.xlabel(r'$i$', fontsize=15)
plt.ylabel(r'$w_i$', fontsize=15)


# In[5]:


import ray
import os


# In[6]:


ckpt = np.load('bb_stat.npy')
std_mu, std_std = ckpt[0]
mean_mu, mean_std = ckpt[1]


# In[ ]:


from scipy.stats import norm

datalen = len(bb_list)
bb_type = np.zeros(datalen)
bbs = np.zeros((datalen, 768))

futures = []

ray.shutdown()
ray.init(num_cpus=28, dashboard_host='0.0.0.0', dashboard_port=40000)

@ray.remote
def proc(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    bb_type = data['type']

    total_length = len(data['bbs'])
    
    basename = os.path.basename(filename)
    with open(os.path.join('/data/pbl/data/pkl', basename), 'rb') as f:
        data2 = pickle.load(f)

    bb_lengths = np.array([len(bb) for bb in data2['bbs']])
    bb_mean, bb_std = bb_lengths.mean(), bb_lengths.std()
    
    std_z = (bb_std - std_mu) / std_std
    mean_z = (bb_mean - mean_mu) / mean_std
    
    std_score = 2 * (1 - norm.cdf(np.abs(std_z)))
    mean_score = 2 * (1 - norm.cdf(np.abs(mean_z)))
    score = 1 - (std_score ** 0.44404637) * (mean_score ** 0.33140396)
    
    if total_length > 1:
        idxs = np.arange(total_length) / (total_length - 1)

        # weights = np.sin((idxs + 1) * np.pi / (total_length + 2))
        weights1 = (1 - np.cos(np.pi * idxs / 0.48)) / 2.
        weights2 = (3 - np.cos(np.pi * idxs / 0.48 / (0.52 / 0.5))) / 4
        weights = np.where(idxs > 0.48, weights2, weights1)
        weight_sum = np.sum(weights)
        bb = np.array(data['bbs'])
        bb_sum = np.sum(bb * weights[:, np.newaxis], axis=0)
        if weight_sum < 1e-5:
            bb_sum = np.sum(np.array(data['bbs']), axis=0)
            bbs = bb_sum
        else:
            bbs = bb_sum / weight_sum
    else:
        bb_sum = np.sum(np.array(data['bbs']), axis=0)
        bbs = bb_sum
        
    return bb_type, bbs, score
            
for i, bb_filename in enumerate(bb_list):
    inp = ray.put(bb_filename)
    futures.append(proc.remote(inp))
    
bb_type, bbs, scores = zip(*ray.get(futures))
bb_type = np.fromiter(bb_type, dtype=np.int32)
bbs = np.fromiter(bbs, dtype=np.float32)
scores = np.fromiter(scores, dtype=np.float32)

ray.shutdown()


# In[9]:


scores = np.array(scores)


# In[10]:


print(len(bb_type), len(bbs))
print(bbs[0].shape)
print(scores.shape)
print(np.unique(bb_type, return_counts=True))


# In[107]:


bb_tsne = TSNE(learning_rate='auto', n_components=2, n_jobs=-1)
bb_transformed = bb_tsne.fit_transform(np.stack(bbs))

xs = bb_transformed[:, 0]
ys = bb_transformed[:, 1]

plt.figure(figsize=(12, 10))
plt.scatter(xs, ys, c=['r' if tp else 'b' for tp in bb_type])


# In[108]:


bb_tsne2 = TSNE(learning_rate='auto', n_components=3, n_jobs=-1)
bb_transformed2 = bb_tsne2.fit_transform(np.stack(bbs))


# In[109]:


fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('BB embedding(weighted-avg) t-SNE', fontsize=20)

xs = bb_transformed2[:, 0]
ys = bb_transformed2[:, 1]
zs = bb_transformed2[:, 2]

ax.scatter(xs, ys, zs, c=['r' if tp else 'b' for tp in bb_type])


# In[10]:


with open('t', 'wt') as f, open('t2', 'wt') as f2:
    for (x, y, z), t in zip(bb_transformed2[:], bb_type):
        if t == 0:
            f.write('{} {} {}\n'.format(x, y, z))
        else:
            f2.write('{} {} {}\n'.format(x, y, z))


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, f1_score
import itertools


# In[17]:


train_X, train_y, train_sc = [], [], []
test_X, test_y, test_sc = [], [], []

test_normal, test_malware = 0, 0

for pe, tp, sc in zip(bbs, bb_type, scores):
    if tp == 0:
        if test_normal < 100:
            test_X.append(pe)
            test_y.append(tp)
            test_normal += 1
            test_sc.append(sc)
        else:
            train_X.append(pe)
            train_y.append(tp)
            train_sc.append(sc)
    elif tp == 1:
        if test_malware < 100:
            test_X.append(pe)
            test_y.append(tp)
            test_malware += 1
            test_sc.append(sc)
        else:
            train_X.append(pe)
            train_y.append(tp)
            train_sc.append(sc)
            
train_X = np.stack(train_X)
test_X = np.stack(test_X)
train_y = np.array(train_y)
test_y = np.array(test_y)
train_sc = np.array(train_sc)
test_sc = np.array(test_sc)


# In[18]:


import xgboost as xgb


# In[19]:


#model = LogisticRegression(n_jobs=-1).fit(train_X, train_y)
#model = SVC(kernel='linear', verbose=True).fit(train_X, train_y)
model = xgb.XGBClassifier().fit(train_X, train_y)


# In[20]:


pred = model.predict(train_X)


# In[21]:


np.mean((pred == train_y).astype(np.float32)), f1_score(train_y, pred)


# In[22]:


print(pred[:30])
print(train_y[:30])


# In[23]:


cm = confusion_matrix(train_y, pred)
cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]


# In[24]:


plt.title('Confusion matrix on train set', fontsize=20)
plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))

tick_marks = np.arange(2)
ticks = ['normal', 'malware']
plt.xticks(tick_marks, ticks, fontsize=15)
plt.yticks(tick_marks, ticks, fontsize=15)
plt.xlabel('Prediction', fontsize=20)
plt.ylabel('Ground truth', fontsize=20)

threshold = cm.max() / 1.5
for i, j in itertools.product(range(2), range(2)):
    plt.text(j, i, '{:0.4f}'.format(cm[i, j]), horizontalalignment='center',
             color='white' if cm[i, j] > threshold else 'black', fontsize=15)


# In[25]:


pred = model.predict(test_X)


# In[26]:


print(pred[:30])
print(test_y[:30])


# In[27]:


np.mean((pred == test_y).astype(np.float32)), f1_score(test_y, pred)


# In[28]:


cm = confusion_matrix(test_y, pred)
cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]


# In[29]:


plt.title('Confusion matrix on test set', fontsize=20)
plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))

tick_marks = np.arange(2)
ticks = ['normal', 'malware']
plt.xticks(tick_marks, ticks, fontsize=15)
plt.yticks(tick_marks, ticks, fontsize=15)
plt.xlabel('Prediction', fontsize=20)
plt.ylabel('Ground truth', fontsize=20)

threshold = cm.max() / 1.5
for i, j in itertools.product(range(2), range(2)):
    plt.text(j, i, '{:0.4f}'.format(cm[i, j]), horizontalalignment='center',
             color='white' if cm[i, j] > threshold else 'black', fontsize=15)


# In[54]:


predict_score = model.predict_proba(train_X).max(axis=-1)

features = np.stack((predict_score, train_sc), axis=-1)
print(features.shape)


# In[99]:


df = pd.DataFrame(features)
df['target'] = train_y
df['predict'] = model.predict(train_X)
df[0] = 1-df[0]
print(df)


# In[100]:


df['sum'] = df[0] + df[1]


# In[101]:


sns.histplot(data=df, x='sum', hue='target')


# In[102]:


sns.kdeplot(data=df, x=0, hue='target')


# In[103]:


sns.histplot(data=df, x=1, hue='target')


# In[132]:


sns.histplot(data=df[(df['target'] == 1) & (df['predict'] == 0)], x=1, hue='target')


# In[118]:


sns.histplot(data=df[(df['target'] == df['predict'])], x=1, hue='target')


# In[105]:


model2 = xgb.XGBClassifier().fit(features, train_y)


# In[133]:


predicts = model.predict_proba(test_X)
pred_scores = predicts.max(axis=-1)
pred = predicts.argmax(axis=-1)
pred = np.where((pred_scores < 0.6) & (pred == 0), test_sc < 0.9, pred)
pred = test_sc < 0.9
np.mean((pred == test_y).astype(np.float32)), f1_score(test_y, pred)


# In[134]:


cm = confusion_matrix(test_y, pred)
cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]


# In[135]:


plt.title('Confusion matrix on test set', fontsize=20)
plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))

tick_marks = np.arange(2)
ticks = ['normal', 'malware']
plt.xticks(tick_marks, ticks, fontsize=15)
plt.yticks(tick_marks, ticks, fontsize=15)
plt.xlabel('Prediction', fontsize=20)
plt.ylabel('Ground truth', fontsize=20)

threshold = cm.max() / 1.5
for i, j in itertools.product(range(2), range(2)):
    plt.text(j, i, '{:0.4f}'.format(cm[i, j]), horizontalalignment='center',
             color='white' if cm[i, j] > threshold else 'black', fontsize=15)


# In[151]:


plt.figure(figsize=(8,5))
labels = ['Ours', 'Opt. DL', 'NoOpt. DL', 'Stats']
values = [0.862559241706161, 0.8557692307692307, 0.804, 0.506]
plt.bar(labels, values, color=['#B68DC0', '#66C8CD', '#F38BA8', '#70D0F6'])
plt.ylabel('F1 score', fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15, rotation=45)


# In[ ]:




