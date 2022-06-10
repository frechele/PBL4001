#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import glob
import pickle
from scipy.stats import norm


# In[3]:


file_lst = glob.glob('/data/pbl/data/pkl/*.pkl')


# In[4]:


bb_len_mean = []
bb_len_std = []
labels = []

total_bb = 0
malware_bb = 0
normal_bb = 0

for filename in file_lst:
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        
    labels.append(data['type'])
    bb_lengths = np.array([len(bb) for bb in data['bbs']])
    
    bb_len_mean.append(bb_lengths.mean())
    bb_len_std.append(bb_lengths.std())
    
    total_bb += len(data['bbs'])
    if data['type'] == 0:
        normal_bb += len(data['bbs'])
    else:
        malware_bb += len(data['bbs'])


# In[5]:


total_bb, normal_bb, malware_bb


# In[6]:


bb_len_mean = np.array(bb_len_mean)
bb_len_std = np.array(bb_len_std)
labels = np.array(labels)


# In[7]:


normal_bb_len_mean = bb_len_mean[np.where(labels == 0)]
normal_bb_len_std = bb_len_std[np.where(labels == 0)]


# In[8]:


std_mu, std_std = normal_bb_len_std.mean(), normal_bb_len_std.std()
print(std_mu, std_std)
mean_mu, mean_std = normal_bb_len_mean.mean(), normal_bb_len_mean.std()
print(mean_mu, mean_std)


# In[9]:


q1, q3 = np.quantile(normal_bb_len_mean, 0.25), np.quantile(normal_bb_len_mean, 0.75)
iqr = q3 - q1
s1 = set(np.where((normal_bb_len_mean <= q3 + 1.5 * iqr) & (normal_bb_len_mean >= q1 - 1.5 * iqr))[0].tolist())
q1, q3 = np.quantile(normal_bb_len_std, 0.25), np.quantile(normal_bb_len_std, 0.75)
iqr = q3 - q1
s2 = set(np.where((normal_bb_len_std <= q3 + 1.5 * iqr) & (normal_bb_len_std >= q1 - 1.5 * iqr))[0].tolist())


# In[10]:


s = s1.intersection(s2)


# In[11]:


normal_bb_len_mean = normal_bb_len_mean[list(s)]
normal_bb_len_std = normal_bb_len_std[list(s)]


# In[12]:


plt.figure(figsize=(8, 6))
plt.hist(normal_bb_len_mean, bins=100)
plt.grid()
plt.title('histogram of BB length mean', fontsize=20)
plt.xlabel('bb length mean', fontsize=15)


# In[13]:


plt.figure(figsize=(8, 6))
plt.hist(normal_bb_len_std, bins=100)
plt.grid()
plt.title('histogram of BB length std.', fontsize=20)
plt.xlabel('bb length std.', fontsize=15)


# In[14]:


std_mu, std_std = normal_bb_len_std.mean(), normal_bb_len_std.std()
print(std_mu, std_std)


# In[15]:


mean_mu, mean_std = normal_bb_len_mean.mean(), normal_bb_len_mean.std()
print(mean_mu, mean_std)


# In[16]:


np.save('bb_stat.npy', np.array([[std_mu, std_std], [mean_mu, mean_std]]))


# In[17]:


def cal_prob(mean, std, w=0.5):
    mean_zscore = (mean - mean_mu) / mean_std
    std_zscore = (std - std_mu) / std_std
    return ((2 * (1 - norm.cdf(abs(mean_zscore)))) ** 0.33140396) * ((2 * (1 - norm.cdf(abs(std_zscore)))) ** 0.44404637)


# In[18]:


score_normal = []
score_malware = []

for mean, std, label in zip(bb_len_mean, bb_len_std, labels):
    score = cal_prob(mean, std, 0.5155099670057157)
    if label == 0:
        score_normal.append(score)
    else:
        score_malware.append(score)
        
score_normal = np.array(score_normal)
score_malware = np.array(score_malware)


# In[19]:


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.ylim(0, 0.4)
plt.hist(score_normal, weights=np.ones(len(score_normal)) / len(score_normal))
plt.title('score histogram for normal', fontsize=20)
plt.xlabel('score', fontsize=15)

plt.subplot(1, 2, 2)
plt.ylim(0, 0.4)
plt.hist(score_malware, weights=np.ones(len(score_malware)) / len(score_malware))
plt.title('score histogram for malware', fontsize=20)
plt.xlabel('score', fontsize=15)


# In[20]:


score_mean, score_std = [], []
classes = []

for mean, std, label in zip(bb_len_mean, bb_len_std, labels):
    score_mean.append(2 * (1 - norm.cdf(abs((mean - mean_mu) / mean_std))))
    score_std.append(2 * (1 - norm.cdf(abs((std - std_mu) / std_std))))
    classes.append(label)
        
score_mean = np.array(score_mean)
score_std = np.array(score_std)
classes = np.array(classes)


# In[21]:


from scipy.optimize import fminbound


# In[22]:


def f(w):
    w1, w2 = w
    score = 1 - (score_mean ** w1) * (score_std ** w2)
    return -np.corrcoef(score, classes)[0, 1]


# In[23]:


X = np.linspace(0, 1, 100)
Y = np.linspace(0, 1, 100)
z = []

for x in X:
    for y in Y:
        z.append(f([x, y]))
z = np.array(z)


# In[24]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

z = np.where(np.isnan(z), 0, z)

ax.plot_surface(X, Y, z.reshape((100, 100)), cmap='jet')

ax.set_xlabel('w1', fontsize=15)
ax.set_ylabel('w2', fontsize=15)
ax.set_zlabel('cost', fontsize=15)


# In[25]:


fmin(f, np.array([1., 1.]))


# In[ ]:


-f([0.33140396, 0.44404637]), -f([1., 1.])


# In[ ]:




