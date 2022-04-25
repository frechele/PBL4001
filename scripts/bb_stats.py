from pickletools import optimize
from statistics import mean
import numpy as np
import glob
import pickle
from scipy.stats import norm
from scipy.optimize import fmin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

std_mu = 0
std_std =0
mean_mu =0
mean_std = 0

def blockSplit(file_lst):
    bb_len_mean = []
    bb_len_std = []
    labels = []

    for filename in file_lst:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        labels.append(data['type'])
        bb_lengths = np.array([len(bb) for bb in data['bbs']])
        
        bb_len_mean.append(bb_lengths.mean())
        bb_len_std.append(bb_lengths.std())
       
    return np.array(labels), np.array(bb_len_mean), np.array(bb_len_std)
    

def zscore(x, mean, std):
    return (x - mean)/std


def cal_prob(mean, std, mean_weight, std_weight):
    mean_zscore = zscore(mean, mean_mu, mean_std)
    std_zscore = zscore(std, std_mu, std_std)
    return 2 * ((1 - norm.cdf(abs(mean_zscore))) ** mean_weight) *  2 * ((1 - norm.cdf(abs(std_zscore)) ** std_weight)) 

def func(x, labels, bb_means, bb_stds):
    score_normal = []
    score_malware = []

    for mean, std, label in zip(bb_means, bb_stds, labels):
        score = cal_prob(mean, std, x, 1/x)
        if label == 0:
            score_normal.append(score)
        else:
            score_malware.append(score)

    score_normal = np.array(score_normal)
    score_malware = np.array(score_malware)
    
    TP = len(score_malware[score_malware < 0.5])
    FN = len(score_malware[score_malware >= 0.5])
    FP = len(score_normal[score_normal < 0.5])
    TN = len(score_normal[score_normal >= 0.5])
    precision = TP / (TP + FP)
    recall = TP / (TP+FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    print (x, TP, FN, FP, TN, F1)
    return 1 / F1

def setStats(labels, bb_means, bb_stds):
    global std_mu, std_std, mean_mu, mean_std
    normal_bb_means = bb_means[np.where(labels == 0)]
    normal_bb_stds = bb_stds[np.where(labels == 0)]
    std_mu, std_std = normal_bb_stds.mean(), normal_bb_stds.std()
    mean_mu, mean_std = normal_bb_means.mean(), normal_bb_means.std()

if __name__ == "__main__" :
    file_lst = glob.glob('data/pkl/*.pkl')

    labels, bb_means, bb_stds = blockSplit(file_lst)
    setStats(labels, bb_means, bb_stds)
    
    print("run")

    mininum = fmin(func, 1, (labels, bb_means, bb_stds))
    print(mininum[0])
    print(1/func(mininum[0], labels, bb_means, bb_stds))
    """
    x = np.array(list(zip(bb_means, bb_stds)))
    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size= 0.2, random_state= 0)
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    predictions = logreg.predict(x_test)
    score = logreg.score(x_test, y_test)
    print(predictions)
    print(score)
    """