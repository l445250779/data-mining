# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 14:26:04 2018

@author: Xudong Li
"""

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

##### Question 1 ################################

ratings = pd.read_csv('ratings.csv')
userID_total = ratings.iloc[-1,0] # num of users
movieID_total = len(set(ratings.iloc[:,1].values)) # num of unique id's
sparsity = len(ratings.iloc[:,2].values)/(userID_total * movieID_total)
print('sparsity = ' , sparsity) # sparsity = 0.016439


##### Question 2 ################################
scores = ratings.iloc[:,2].values
s = Counter(scores)
labels, values = zip(*s.items())
indexes = np.arange(0.5,5.5,0.5)
plt.bar(indexes, list(values), 0.3)
plt.xticks(indexes)
plt.xlabel('Ratings', fontsize = 12)
plt.ylabel('Frequency', fontsize = 12)
plt.show() # The shape is similar to normal distribution, but with many values of 0.5  


##### Question 3 ################################
import seaborn as sns
rating_freq = Counter(ratings.iloc[:,1].values)
sorted_rating_freq = rating_freq.most_common()
sns.countplot(x='movieId',data=ratings, order = ratings['movieId'].value_counts().index)
plt.show()

##### Question 4 ################################
rating_freq = Counter(ratings.iloc[:,0].values)
sorted_rating_freq = rating_freq.most_common()
sns.countplot(x='userId',data=ratings, order = ratings['userId'].value_counts().index)
plt.show()


##### Question 5 ################################
# See report

##### Question 6 ################################
D = {}
movie_rating = ratings.iloc[:,1:3].values
for i in movie_rating:
    D.setdefault(i[0], []).append(i[1])
variances = {}
for k, v in D.items():
    variances[k] = np.var(v)
plt.hist(list(variances.values()),bins = 10, rwidth = 1)
plt.xlabel('Variances', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.show() 

##### Question 7,8,9  ################################
# See report

##### Question 10 ################################
# import data from surprise
from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
from surprise.prediction_algorithms.knns import KNNWithMeans
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []
mae_all = []

sim = {'name':'pearson', 'user_based': True}
for k in range(2,102,2):
    rmse = []
    mae = []
    knn_with_means = KNNWithMeans(k = k, sim_options = sim)
    
    for trainset, testset in k_fold.split(dataset):
        knn_with_means.fit(trainset)
        pred = knn_with_means.test(testset)
        rmse.append(accuracy.rmse(pred))
        mae.append(accuracy.mae(pred))
    
    rmse_all.append(np.mean(rmse))
    mae_all.append(np.mean(mae))
    
k = np.arange(2,102,2)
plt.plot(k, rmse_all, label='RMSE')
plt.plot(k, mae_all, label = 'MSE')
plt.xlabel('k')
plt.ylabel('RMSE/MAE')
plt.legend()
plt.show()

##### Question 11 ################################
# See report

##### Question 12 ################################
from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
from surprise.prediction_algorithms.knns import KNNWithMeans
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []

# find movieID(keys) that need to be deleted ######
ratings = pd.read_csv('ratings.csv')
rating_freq = dict(Counter(ratings.iloc[:,1].values))

###########################################

def trim(testset):
    for i in testset:
        if (rating_freq[int(i[1])] <= 2): #if movieID of a testset is in delete_keys
            testset.remove(i) #remove it
    return testset

sim = {'name':'pearson', 'user_based': True}
for k in range(2,102,2):
    rmse = []
    knn_with_means = KNNWithMeans(k = k, sim_options = sim)
    
    for trainset, testset in k_fold.split(dataset):
        knn_with_means.fit(trainset)
        testset = trim(testset)
        pred = knn_with_means.test(testset)
        rmse.append(accuracy.rmse(pred))
    
    rmse_all.append(np.mean(rmse))
    
k = np.arange(2,102,2)
plt.plot(k, rmse_all, label='RMSE')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.title('KNN with Popular Movie Trimmed')
plt.legend()
plt.show()


##### Question 13 ################################

from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
from surprise.prediction_algorithms.knns import KNNWithMeans
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []

# find movieID(keys) that need to be deleted ######
ratings = pd.read_csv('ratings.csv')
rating_freq = dict(Counter(ratings.iloc[:,1].values))

###########################################

def trim(testset):
    for i in testset:
        if (rating_freq[int(i[1])] > 2): #if movieID of a testset is in delete_keys
            testset.remove(i) #remove it
    return testset

sim = {'name':'pearson', 'user_based': True}
for k in range(2,102,2):
    rmse = []
    knn_with_means = KNNWithMeans(k = k, sim_options = sim)
    
    for trainset, testset in k_fold.split(dataset):
        knn_with_means.fit(trainset)
        testset = trim(testset)
        pred = knn_with_means.test(testset)
        rmse.append(accuracy.rmse(pred))
    
    rmse_all.append(np.mean(rmse))
    
k = np.arange(2,102,2)
plt.plot(k, rmse_all, label='RMSE')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.title('KNN with Unpopular Movie Trimmed')
plt.legend()
plt.show()


##### Question 14 ################################

from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
from surprise.prediction_algorithms.knns import KNNWithMeans
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []

# find movieID(keys) that need to be deleted ######
ratings = pd.read_csv('ratings.csv')
rating_freq = dict(Counter(ratings.iloc[:,1].values))

D = {}
movie_rating = ratings.iloc[:,1:3].values
for i in movie_rating:
    D.setdefault(i[0], []).append(i[1])
variances = {}
for k, v in D.items():
    variances[k] = np.var(v)
    
###########################################

def trim(testset):
    for i in testset:
        if (rating_freq[int(i[1])] < 5 or variances[int(i[1])] < 2): #if movieID of a testset is in delete_keys
            testset.remove(i) #remove it
    return testset

sim = {'name':'pearson', 'user_based': True}
for k in range(2,102,2):
    rmse = []
    knn_with_means = KNNWithMeans(k = k, sim_options = sim)
    
    for trainset, testset in k_fold.split(dataset):
        knn_with_means.fit(trainset)
        testset = trim(testset)
        pred = knn_with_means.test(testset)
        rmse.append(accuracy.rmse(pred))
    
    rmse_all.append(np.mean(rmse))
    
k = np.arange(2,102,2)
plt.plot(k, rmse_all, label='RMSE')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.title('KNN with High Variance Movie Trimmed')
plt.legend()
plt.show()


##### Question 15 ################################
from surprise.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from surprise import Dataset
from surprise import Reader
import os

from surprise.prediction_algorithms.knns import KNNWithMeans

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

k = 20
thresholds = [2.5, 3, 3.5, 4]
sim = {'name':'pearson', 'user_based': True}

trainset, testset = train_test_split(dataset, test_size = 0.1)
knn_with_means = KNNWithMeans(k = k, sim_options = sim)
knn_with_means.fit(trainset)
pred = knn_with_means.test(testset)

true_label = [i[2] for i in pred]
pred_label = [i[3] for i in pred]


def plot_roc(fpr, tpr, t):
    fig, ax = plt.subplots()

    roc_auc = auc(fpr,tpr)

    ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)
    
    plt.title('ROC Curve with threshold = ' + str(t))
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate',fontsize=15)
    ax.set_ylabel('True Positive Rate',fontsize=15)

    ax.legend(loc="lower right")

    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(15)
    

for t in thresholds:
    threshold_true = list(map(lambda x:0 if x < t else 1, true_label))
    fpr, tpr, _ = roc_curve(threshold_true, pred_label)
    plot_roc(fpr, tpr, t)


##### Question 16 ################################
# see report


##### Question 17 ################################
from surprise import NMF
from surprise.model_selection import cross_validate

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
data = Dataset.load_from_file(file_path, reader=reader)
ks = np.arange(2,52,2)

RMSES = []
MAES = []
for k in ks:
    print('Now staring k =',k)
    nmf = NMF(n_factors=int(k))
    RMSES.append(cross_validate(nmf, data, measures=['RMSE'], cv=10))
    MAES.append(cross_validate(nmf, data, measures=[ 'MAE'], cv=10))

avg_RMSE = []
avg_MAE = []
for RMSE in RMSES:
    avg_RMSE.append(np.mean(RMSE['test_rmse']))
for MAE in MAES:
    avg_MAE.append(np.mean(MAE['test_mae']))

plt.figure(1)
plt.plot(ks, avg_RMSE)
plt.title('RMSE vs Ks')
plt.xlabel('Ks')
plt.ylabel('RMSE')

plt.figure(2)
plt.plot(ks, avg_MAE)
plt.title('MAE vs Ks')
plt.xlabel('Ks')
plt.ylabel('MAE')


##### Question 18 ################################
# Find the min value for RMSE
list_sort = np.vstack((ks,avg_RMSE)).T
min_sort = sorted(list_sort, key=lambda rank: rank[1], reverse=False)
min_index = min_sort[0][0]
print("The optimal number for RSMS is", min_index)
print("The minimun average RSMS is", min_sort[0][1])

# Find the min value for MAE
list_sort = np.vstack((ks,avg_MAE)).T
min_sort = sorted(list_sort, key=lambda rank: rank[1], reverse=False)
min_index = min_sort[0][0]
print("The optimal number for MAE is", min_index)
print("The minimun average MAE is", min_sort[0][1])


##### Question 19 ################################
from surprise import NMF
from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []

# find movieID(keys) that need to be deleted ######
ratings = pd.read_csv('ratings.csv')
rating_freq = dict(Counter(ratings.iloc[:,1].values))

data = Dataset.load_builtin('ml-100k')
k_fold = KFold(n_splits=10)

def trim(testset):
    for i in testset:
        if (rating_freq[int(i[1])] <= 2): #if movieID of a testset is in delete_keys
            testset.remove(i) #remove it
    return testset

sim = {'name':'pearson', 'user_based': True}
for k in range(2,52,2):
    rmse = []
    nmf = NMF(n_factors=int(k))
    print('Now starting k =', k)
    
    for trainset, testset in k_fold.split(dataset):
        nmf.fit(trainset)
        testset = trim(testset)
        pred = nmf.test(testset)
        rmse.append(accuracy.rmse(pred))
    
    rmse_all.append(np.mean(rmse))

    
ks = np.arange(2,52,2)

list_sort = np.vstack((ks, rmse_all)).T
min_sort = sorted(list_sort, key=lambda rank: rank[1], reverse=False)
min_index = min_sort[0][0]
min_RMSE = min_sort[0][1]
print("The optimal number for RSMS is", min_index)
print("The miinimum average RMSE is", min_RMSE)

plt.plot(ks, rmse_all, label='RMSE')
plt.xlabel('ks')
plt.ylabel('RMSE')
plt.title('NNMF with Popular Movie Trimmed')
plt.legend()
plt.show()


##### Question 20 ################################
from surprise import NMF
from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []

# find movieID(keys) that need to be deleted ######
ratings = pd.read_csv('ratings.csv')
rating_freq = dict(Counter(ratings.iloc[:,1].values))

###########################################

def trim(testset):
    for i in testset:
        if (rating_freq[int(i[1])] > 2): #if movieID of a testset is in delete_keys
            testset.remove(i) #remove it
    return testset

sim = {'name':'pearson', 'user_based': True}
for k in range(2,52,2):
    rmse = []
    nmf = NMF(n_factors=int(k))
    print('Now starting k =', k)
    
    for trainset, testset in k_fold.split(dataset):
        nmf.fit(trainset)
        testset = trim(testset)
        pred = nmf.test(testset)
        rmse.append(accuracy.rmse(pred))
    
    rmse_all.append(np.mean(rmse))

    
ks = np.arange(2,52,2)

list_sort = np.vstack((ks, rmse_all)).T
min_sort = sorted(list_sort, key=lambda rank: rank[1], reverse=False)
min_index = min_sort[0][0]
min_RMSE = min_sort[0][1]
print("The optimal number for RSMS is", min_index)
print("The miinimum average RMSE is", min_RMSE)

plt.plot(ks, rmse_all, label='RMSE')
plt.xlabel('ks')
plt.ylabel('RMSE')
plt.title('NNMF with Unpopular Movie Trimmed')
plt.legend()
plt.show()


##### Question 21 ################################

from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []

# find movieID(keys) that need to be deleted ######
ratings = pd.read_csv('ratings.csv')
rating_freq = dict(Counter(ratings.iloc[:,1].values))

D = {}
movie_rating = ratings.iloc[:,1:3].values
for i in movie_rating:
    D.setdefault(i[0], []).append(i[1])
variances = {}
for k, v in D.items():
    variances[k] = np.var(v)
    
###########################################

def trim(testset):
    for i in testset:
        if (rating_freq[int(i[1])] < 5 or variances[int(i[1])] < 2): #if movieID of a testset is in delete_keys
            testset.remove(i) #remove it
    return testset

sim = {'name':'pearson', 'user_based': True}
for k in range(2,52,2):
    rmse = []
    nmf = NMF(n_factors=int(k))
    print('Now starting k =', k)
    
    for trainset, testset in k_fold.split(dataset):
        nmf.fit(trainset)
        testset = trim(testset)
        pred = nmf.test(testset)
        rmse.append(accuracy.rmse(pred))
    
    rmse_all.append(np.mean(rmse))

    
ks = np.arange(2,52,2)

list_sort = np.vstack((ks, rmse_all)).T
min_sort = sorted(list_sort, key=lambda rank: rank[1], reverse=False)
min_index = min_sort[0][0]
min_RMSE = min_sort[0][1]
print("The optimal number for RSMS is", min_index)
print("The miinimum average RMSE is", min_RMSE)

plt.plot(ks, rmse_all, label='RMSE')
plt.xlabel('ks')
plt.ylabel('RMSE')
plt.title('NNMF with High Variance Movie Trimmed')
plt.legend()
plt.show()


##### Question 22 ################################
from surprise.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from surprise import Dataset
from surprise import Reader
import os

from surprise import NMF

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

k = 20
thresholds = [2.5, 3, 3.5, 4]
sim = {'name':'pearson', 'user_based': True}

trainset, testset = train_test_split(dataset, test_size = 0.1)
nmf = NMF(n_factors=20)
nmf.fit(trainset)
pred = nmf.test(testset)

true_label = [i[2] for i in pred]
pred_label = [i[3] for i in pred]


def plot_roc(fpr, tpr, t):
    fig, ax = plt.subplots()

    roc_auc = auc(fpr,tpr)

    ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)
    
    plt.title('ROC Curve with threshold = ' + str(t))
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate',fontsize=15)
    ax.set_ylabel('True Positive Rate',fontsize=15)

    ax.legend(loc="lower right")

    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(15)
    

for t in thresholds:
    threshold_true = list(map(lambda x:0 if x < t else 1, true_label))
    fpr, tpr, _ = roc_curve(threshold_true, pred_label)
    plot_roc(fpr, tpr, t)
    
    
##### Question 23 ################################
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from surprise import Dataset
from surprise import Reader
import os
import csv
from surprise.prediction_algorithms.matrix_factorization import NMF


file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

trainset= dataset.build_full_trainset()
    
nmf = NMF(20, random_state=0)
nmf.fit(trainset)
V = nmf.qi
print(V.shape)
Vt = np.transpose(V)
movie_index = []
Vsort = np.argsort(Vt[5])
b = np.array(Vsort[:10])

for j in b:
        movie_index.append(trainset.to_raw_iid(j))

print('Movie ID: ',movie_index) 

indexx = []
col1 = []
col2 = []
op = open('ml-latest-small/movies.csv',encoding="utf8") 
movie = csv.reader(op)
for row in movie:
    indexx.append(row[0])
    col1.append(row[1])   
    col2.append(row[2])
op.close()
index_list = []

for value in indexx:
    for movie_id in movie_index:
        if value == movie_id:
            a = np.array(indexx).tolist()
            index_list.append(a.index(value))

print ('Movie Index: ', index_list)
genre = []
for i in index_list:
    genre.append(col2[i])
print('Top 10 Movie Genres: ',genre)

##### Question 24 ################################
from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
from surprise.prediction_algorithms.matrix_factorization import SVD
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []

# find movieID(keys) that need to be deleted ######
ratings = pd.read_csv('ratings.csv')
rating_freq = dict(Counter(ratings.iloc[:,1].values))

###########################################

mae_all = []



for k in range(2,52,2):
    rmse = []
    mae = []
    svd = SVD(n_factors = k)

    for trainset, testset in k_fold.split(dataset):
        svd.fit(trainset)
        pred = svd.test(testset)
        rmse.append(accuracy.rmse(pred))
        mae.append(accuracy.mae(pred))

    rmse_all.append(np.mean(rmse))
    mae_all.append(np.mean(mae))

k = np.arange(2,52,2)
plt.plot(k, rmse_all, label='RMSE')
plt.plot(k, mae_all, label = 'MSE')
plt.xlabel('k')
plt.ylabel('RMSE/MAE')
plt.legend()
plt.show()

count = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50]
a = np.array(rmse_all).tolist()
mink = count[a.index(min(a))]

print()
print('The  Optimal number of latent factors is :', mink)
print()
print('The minimum average RMSE is: ', min(rmse_all))
print('The minimum average MAE is: ', min(mae_all))
print()

##### Question 25 ################################
# See Report

##### Question 26 ################################
from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
from surprise.prediction_algorithms.knns import KNNWithMeans
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []

# find movieID(keys) that need to be deleted ######
ratings = pd.read_csv('ratings.csv')
rating_freq = dict(Counter(ratings.iloc[:,1].values))

###########################################

def trim(testset):
    for i in testset:
        if (rating_freq[int(i[1])] <= 2): #if movieID of a testset is in delete_keys
            testset.remove(i) #remove it
    return testset

sim = {'name':'pearson', 'user_based': True}
for k in range(2,52,2):
    rmse = []
    svd = SVD(n_factors = k)

    for trainset, testset in k_fold.split(dataset):
        svd.fit(trainset)
        testset = trim(testset)
        pred = svd.test(testset)
        rmse.append(accuracy.rmse(pred))

    rmse_all.append(np.mean(rmse))

k = np.arange(2,52,2)
plt.plot(k, rmse_all, label='RMSE')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.legend()
plt.show()

print()
print('The  minimum average RMSE is :', min(rmse_all))
print()
##### Question 27 ################################

from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
from surprise.prediction_algorithms.knns import KNNWithMeans
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []

# find movieID(keys) that need to be deleted ######
ratings = pd.read_csv('ratings.csv')
rating_freq = dict(Counter(ratings.iloc[:,1].values))

###########################################

def trim(testset):
    for i in testset:
        if (rating_freq[int(i[1])] > 2): #if movieID of a testset is in delete_keys
            testset.remove(i) #remove it
    return testset

sim = {'name':'pearson', 'user_based': True}
for k in range(2,52,2):
    rmse = []
    svd = SVD(n_factors = k)

    for trainset, testset in k_fold.split(dataset):
        svd.fit(trainset)
        testset = trim(testset)
        pred = svd.test(testset)
        rmse.append(accuracy.rmse(pred))

    rmse_all.append(np.mean(rmse))

k = np.arange(2,52,2)
plt.plot(k, rmse_all, label='RMSE')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.legend()
plt.show()

print()
print('The  minimum average RMSE is :', min(rmse_all))
print()
##### Question 28 ################################

from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
from surprise.prediction_algorithms.knns import KNNWithMeans
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []

# find movieID(keys) that need to be deleted ######
ratings = pd.read_csv('ratings.csv')
rating_freq = dict(Counter(ratings.iloc[:,1].values))

D = {}
movie_rating = ratings.iloc[:,1:3].values
for i in movie_rating:
    D.setdefault(i[0], []).append(i[1])
variances = {}
for k, v in D.items():
    variances[k] = np.var(v)

###########################################

def trim(testset):
    for i in testset:
        if (rating_freq[int(i[1])] < 5 or variances[int(i[1])] < 2): #if movieID of a testset is in delete_keys
            testset.remove(i) #remove it
    return testset

sim = {'name':'pearson', 'user_based': True}
for k in range(2,52,2):
    rmse = []
    svd = SVD(n_factors = k)

    for trainset, testset in k_fold.split(dataset):
        svd.fit(trainset)
        testset = trim(testset)
        pred = svd.test(testset)
        rmse.append(accuracy.rmse(pred))

    rmse_all.append(np.mean(rmse))

k = np.arange(2,52,2)
plt.plot(k, rmse_all, label='RMSE')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.legend()
plt.show()

print()
print('The  minimum average RMSE is :', min(rmse_all))
print()

##### Question 29 ################################
from surprise.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import Dataset
from surprise import Reader
import os



file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

k = 14
thresholds = [2.5, 3, 3.5, 4]

trainset, testset = train_test_split(dataset, test_size = 0.1)
svd = SVD(n_factors=k)
svd.fit(trainset)
pred = svd.test(testset)

true_label = [i[2] for i in pred]
pred_label = [i[3] for i in pred]


def plot_roc(fpr, tpr, t):
    fig, ax = plt.subplots()

    roc_auc = auc(fpr,tpr)

    ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)

    plt.title('ROC Curve with threshold = ' + str(t))
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate',fontsize=15)
    ax.set_ylabel('True Positive Rate',fontsize=15)

    ax.legend(loc="lower right")

    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(15)


for t in thresholds:
    threshold_true = list(map(lambda x:0 if x < t else 1, true_label))
    fpr, tpr, _ = roc_curve(threshold_true, pred_label)
    plot_roc(fpr, tpr, t)
    plt.show()
    
    
##### Question 30 ################################
# import data from surprise
from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
from surprise.prediction_algorithms.knns import KNNWithMeans
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []

D = {}
for i in dataset.raw_ratings:
    D.setdefault(i[0], []).append(i[2])
average_user_rating = {}
for k, v in D.items():
    average_user_rating[k] = np.average(v)


from surprise.prediction_algorithms.predictions import Prediction
    
sim = {'name':'pearson', 'user_based': True}
for k in range(2,102,2):
    rmse = []
    knn_with_means = KNNWithMeans(k = k, sim_options = sim)
    
    for trainset, testset in k_fold.split(dataset):
        knn_with_means.fit(trainset)
        pred = knn_with_means.test(testset)
        # The idea is that to replace the predicted result from KNN to average user rating
        # The reason we use KNN is just to get the format of prediction, so accuracy.rmse() can be run
        # The model and training right here does not matter.
        new_pred = []
        for i in pred:
            new_pred.append(Prediction(i[0], i[1], i[2], average_user_rating[i[0]], i[4]))
        
        rmse.append(accuracy.rmse(new_pred))
    
    rmse_all.append(np.mean(rmse))
    
k = np.arange(2,102,2)
plt.plot(k, rmse_all, label='RMSE')
plt.xlabel('iteration')
plt.ylabel('RMSE')
axes = plt.gca()
axes.set_ylim([1.0, 1.5])
plt.legend()
plt.show()

##### Question 31 ################################
# import data from surprise
from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
from surprise.prediction_algorithms.knns import KNNWithMeans
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []

D = {}
for i in dataset.raw_ratings:
    D.setdefault(i[0], []).append(i[2])
average_user_rating = {}
for k, v in D.items():
    average_user_rating[k] = np.average(v)


from surprise.prediction_algorithms.predictions import Prediction

# find movieID(keys) that need to be deleted ######
ratings = pd.read_csv('ratings.csv')
rating_freq = dict(Counter(ratings.iloc[:,1].values))

###########################################

def trim(testset):
    for i in testset:
        if (rating_freq[int(i[1])] <= 2): #if movieID of a testset is in delete_keys
            testset.remove(i) #remove it
    return testset

sim = {'name':'pearson', 'user_based': True}
for k in range(2,102,2):
    rmse = []
    knn_with_means = KNNWithMeans(k = k, sim_options = sim)
    
    for trainset, testset in k_fold.split(dataset):
        knn_with_means.fit(trainset)
        testset = trim(testset)
        pred = knn_with_means.test(testset)
        # The idea is that to replace the predicted result from KNN to average user rating
        # The reason we use KNN is just to get the format of prediction, so accuracy.rmse() can be run
        # The model and training right here does not matter.
        new_pred = []
        for i in pred:
            new_pred.append(Prediction(i[0], i[1], i[2], average_user_rating[i[0]], i[4]))
        
        rmse.append(accuracy.rmse(new_pred))
    
    rmse_all.append(np.mean(rmse))
    
k = np.arange(2,102,2)
plt.plot(k, rmse_all, label='RMSE')
plt.xlabel('iteration')
plt.ylabel('RMSE')
axes = plt.gca()
axes.set_ylim([1.0, 1.5])
plt.legend()
plt.show()

##### Question 32 ################################
# import data from surprise
from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
from surprise.prediction_algorithms.knns import KNNWithMeans
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []

D = {}
for i in dataset.raw_ratings:
    D.setdefault(i[0], []).append(i[2])
average_user_rating = {}
for k, v in D.items():
    average_user_rating[k] = np.average(v)


from surprise.prediction_algorithms.predictions import Prediction

# find movieID(keys) that need to be deleted ######
ratings = pd.read_csv('ratings.csv')
rating_freq = dict(Counter(ratings.iloc[:,1].values))

###########################################

def trim(testset):
    for i in testset:
        if (rating_freq[int(i[1])] > 2): #if movieID of a testset is in delete_keys
            testset.remove(i) #remove it
    return testset

sim = {'name':'pearson', 'user_based': True}
for k in range(2,102,2):
    rmse = []
    knn_with_means = KNNWithMeans(k = k, sim_options = sim)
    
    for trainset, testset in k_fold.split(dataset):
        knn_with_means.fit(trainset)
        testset = trim(testset)
        pred = knn_with_means.test(testset)
        # The idea is that to replace the predicted result from KNN to average user rating
        # The reason we use KNN is just to get the format of prediction, so accuracy.rmse() can be run
        # The model and training right here does not matter.
        new_pred = []
        for i in pred:
            new_pred.append(Prediction(i[0], i[1], i[2], average_user_rating[i[0]], i[4]))
        
        rmse.append(accuracy.rmse(new_pred))
    
    rmse_all.append(np.mean(rmse))
    
k = np.arange(2,102,2)
plt.plot(k, rmse_all, label='RMSE')
plt.xlabel('iteration')
plt.ylabel('RMSE')
axes = plt.gca()
axes.set_ylim([1.0, 1.5])
plt.legend()
plt.show()


##### Question 33 ################################
# import data from surprise
from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
from surprise.prediction_algorithms.knns import KNNWithMeans
k_fold = KFold(n_splits=10)

from surprise import accuracy

rmse_all = []

D = {}
for i in dataset.raw_ratings:
    D.setdefault(i[0], []).append(i[2])
average_user_rating = {}
for k, v in D.items():
    average_user_rating[k] = np.average(v)


from surprise.prediction_algorithms.predictions import Prediction

# find movieID(keys) that need to be deleted ######
ratings = pd.read_csv('ratings.csv')
rating_freq = dict(Counter(ratings.iloc[:,1].values))

D = {}
movie_rating = ratings.iloc[:,1:3].values
for i in movie_rating:
    D.setdefault(i[0], []).append(i[1])
variances = {}
for k, v in D.items():
    variances[k] = np.var(v)
    
###########################################

def trim(testset):
    for i in testset:
        if (rating_freq[int(i[1])] < 5 or variances[int(i[1])] < 2): #if movieID of a testset is in delete_keys
            testset.remove(i) #remove it
    return testset

sim = {'name':'pearson', 'user_based': True}
for k in range(2,102,2):
    rmse = []
    knn_with_means = KNNWithMeans(k = k, sim_options = sim)
    
    for trainset, testset in k_fold.split(dataset):
        knn_with_means.fit(trainset)
        testset = trim(testset)
        pred = knn_with_means.test(testset)
        # The idea is that to replace the predicted result from KNN to average user rating
        # The reason we use KNN is just to get the format of prediction, so accuracy.rmse() can be run
        # The model and training right here does not matter.
        new_pred = []
        for i in pred:
            new_pred.append(Prediction(i[0], i[1], i[2], average_user_rating[i[0]], i[4]))
        
        rmse.append(accuracy.rmse(new_pred))
    
    rmse_all.append(np.mean(rmse))
    
k = np.arange(2,102,2)
plt.plot(k, rmse_all, label='RMSE')
plt.xlabel('iteration')
plt.ylabel('RMSE')
axes = plt.gca()
axes.set_ylim([1.0, 1.5])
plt.legend()
plt.show()

##### Question 34 ################################
from surprise.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from surprise import Dataset
from surprise import Reader
import os

from surprise import NMF
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.matrix_factorization import SVD

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)


thresholds = [3]
sim = {'name':'pearson', 'user_based': True}

trainset, testset = train_test_split(dataset, test_size = 0.1)

nmf = NMF(n_factors=20)
nmf.fit(trainset)
pred_nmf = nmf.test(testset)

svd = SVD(n_factors = 16)
svd.fit(trainset)
pred_svd = svd.test(testset)

knn_with_means = KNNWithMeans(k = 20, sim_options = sim)
knn_with_means.fit(trainset)
pred_knn_with_means = knn_with_means.test(testset)

true_label_nmf = [i[2] for i in pred_nmf]
pred_label_nmf = [i[3] for i in pred_nmf]

true_label_svd = [i[2] for i in pred_svd]
pred_label_svd = [i[3] for i in pred_svd]

true_label_knn = [i[2] for i in pred_knn_with_means]
pred_label_knn = [i[3] for i in pred_knn_with_means]


def plot_roc(fpr, tpr, t, algo):
    

    roc_auc = auc(fpr,tpr)

    plt.plot(fpr, tpr, lw=2, label= str(algo) + ' area under curve = %0.4f' % roc_auc)

    plt.grid(color='0.7', linestyle='--', linewidth=1)
    
    plt.title('ROC Curve with threshold = ' + str(t))
    plt.xlim([-0.1, 1.1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=15)
    plt.ylabel('True Positive Rate',fontsize=15)

    plt.legend(loc="lower right")
    plt.show()

for t in thresholds:
    threshold_true_nmf = list(map(lambda x:0 if x < t else 1, true_label_nmf))
    fpr, tpr, _ = roc_curve(threshold_true_nmf, pred_label_nmf)
    plot_roc(fpr, tpr, t, 'nmf')
    threshold_true_svd = list(map(lambda x:0 if x < t else 1, true_label_svd))
    fpr, tpr, _ = roc_curve(threshold_true_svd, pred_label_svd)
    plot_roc(fpr, tpr, t, 'svd')
    threshold_true_knn = list(map(lambda x:0 if x < t else 1, true_label_knn))
    fpr, tpr, _ = roc_curve(threshold_true_knn, pred_label_knn)
    plot_roc(fpr, tpr, t, 'knn')

##### Question 35 ################################
# See report

##### Question 36 ################################
# import data from surprise
from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
from surprise.prediction_algorithms.knns import KNNWithMeans
k_fold = KFold(n_splits=10)

sim = {'name':'pearson', 'user_based': True}

from collections import defaultdict
def calculate_precision(pred, t):
    # Build G by selecting all the movies that has threshold > 3
    # The key is userId, and value is movieID, true rating, and predicted rating
    G = defaultdict(list)
    for prediction in pred:
        if (prediction[2] > 3):
            G[prediction[0]].append((prediction[1], prediction[2], prediction[3]))
    for k, v in G.items():
        # if the length of list is 0, delete the user
        if (len(G[k]) == 0):
            del G[k]
    
    S = defaultdict(list)
    for prediction in pred:
        # get top predicted item S
        S[prediction[0]].append((prediction[1], prediction[2], prediction[3]))
        S[prediction[0]] = sorted(S[prediction[0]], key=lambda x: x[2], reverse=True)[:t]
    
    precision_all_users = []
    for k, v in S.items():
        if (len(S[k]) < t):
            continue
        count = 0
        for item in S[k]:
            if (item in G[k]):
                count = count + 1
        precision_each_user = count / len(S[k])
        precision_all_users.append(precision_each_user)
    
    precision_each_fold = np.mean(precision_all_users)
    
    return precision_each_fold
    
def calculate_recall(pred, t):
    # Build G by selecting all the movies that has threshold > 3
    # The key is userId, and value is movieID, true rating, and predicted rating
    G = defaultdict(list)
    for prediction in pred:
        if (prediction[2] > 3):
            G[prediction[0]].append((prediction[1], prediction[2], prediction[3]))
    for k, v in G.items():
        # if the length of list is 0, delete the user
        if (len(G[k]) == 0):
            del G[k]
    
    S = defaultdict(list)
    for prediction in pred:
        # get top predicted item S
        S[prediction[0]].append((prediction[1], prediction[2], prediction[3]))
        S[prediction[0]] = sorted(S[prediction[0]], key=lambda x: x[2], reverse=True)[:t]
    
    recall_all_users = []
    for k, v in S.items():
        if (len(S[k]) < t):
            continue
        count = 0
        for item in S[k]:
            if (item in G[k]):
                count = count + 1
        if (len(G[k]) != 0):
            recall_each_user = count / len(G[k])
            recall_all_users.append(recall_each_user)
    
    recall_each_fold = np.mean(recall_all_users)
    
    return recall_each_fold
    
precision_all_KNN = []
recall_all_KNN = []
for t in range(1,26,1):
    precision = []
    recall = []
    for trainset, testset in k_fold.split(dataset):    
        knn_with_means = KNNWithMeans(k = 20, sim_options = sim)
        knn_with_means.fit(trainset)
        pred = knn_with_means.test(testset)
        precision_each_fold = calculate_precision(pred, t)
        recall_each_fold = calculate_recall(pred, t)
        precision.append(precision_each_fold)
        recall.append(recall_each_fold)
    precision_all_KNN.append(np.mean(precision))
    recall_all_KNN.append(np.mean(recall))    
        


k = np.arange(1,26,1)
plt.plot(k, precision_all_KNN, label='Precision')
plt.xlabel('t')
plt.ylabel('Precision')
plt.title('KNN Precision')
plt.legend()
plt.show()


plt.plot(k, recall_all_KNN, label='Recall')
plt.xlabel('t')
plt.ylabel('Recall')
plt.title('KNN Recall')
plt.legend()
plt.show()


plt.plot(recall_all_KNN, precision_all_KNN, label='Precision vs. Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('KNN Precision vs. Recall')
plt.legend()
plt.show()


##### Question 37 ################################
# import data from surprise
from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
k_fold = KFold(n_splits=10)

sim = {'name':'pearson', 'user_based': True}

from collections import defaultdict
def calculate_precision(pred, t):
    # Build G by selecting all the movies that has threshold > 3
    # The key is userId, and value is movieID, true rating, and predicted rating
    G = defaultdict(list)
    for prediction in pred:
        if (prediction[2] > 3):
            G[prediction[0]].append((prediction[1], prediction[2], prediction[3]))
    for k, v in G.items():
        # if the length of list is 0, delete the user
        if (len(G[k]) == 0):
            del G[k]
    
    S = defaultdict(list)
    for prediction in pred:
        # get top predicted item S
        S[prediction[0]].append((prediction[1], prediction[2], prediction[3]))
        S[prediction[0]] = sorted(S[prediction[0]], key=lambda x: x[2], reverse=True)[:t]
    
    precision_all_users = []
    for k, v in S.items():
        if (len(S[k]) < t):
            continue
        count = 0
        for item in S[k]:
            if (item in G[k]):
                count = count + 1
        precision_each_user = count / len(S[k])
        precision_all_users.append(precision_each_user)
    
    precision_each_fold = np.mean(precision_all_users)
    
    return precision_each_fold
    
def calculate_recall(pred, t):
    # Build G by selecting all the movies that has threshold > 3
    # The key is userId, and value is movieID, true rating, and predicted rating
    G = defaultdict(list)
    for prediction in pred:
        if (prediction[2] > 3):
            G[prediction[0]].append((prediction[1], prediction[2], prediction[3]))
    for k, v in G.items():
        # if the length of list is 0, delete the user
        if (len(G[k]) == 0):
            del G[k]
    
    S = defaultdict(list)
    for prediction in pred:
        # get top predicted item S
        S[prediction[0]].append((prediction[1], prediction[2], prediction[3]))
        S[prediction[0]] = sorted(S[prediction[0]], key=lambda x: x[2], reverse=True)[:t]
    
    recall_all_users = []
    for k, v in S.items():
        if (len(S[k]) < t):
            continue
        count = 0
        for item in S[k]:
            if (item in G[k]):
                count = count + 1
        if (len(G[k]) != 0):
            recall_each_user = count / len(G[k])
            recall_all_users.append(recall_each_user)
    
    recall_each_fold = np.mean(recall_all_users)
    
    return recall_each_fold

from surprise import NMF
nmf = NMF(n_factors = 20)

    
precision_all_NNMF = []
recall_all_NNMF = []
for t in range(1,26,1):
    precision = []
    recall = []
    for trainset, testset in k_fold.split(dataset):
        nmf.fit(trainset)
        pred = nmf.test(testset)
        precision_each_fold = calculate_precision(pred, t)
        recall_each_fold = calculate_recall(pred, t)
        precision.append(precision_each_fold)
        recall.append(recall_each_fold)
    precision_all_NNMF.append(np.mean(precision))
    recall_all_NNMF.append(np.mean(recall))    
        


k = np.arange(1,26,1)
plt.plot(k, precision_all_NNMF, label='Precision')
plt.xlabel('t')
plt.ylabel('Precision')
plt.title('NNMF Precision')
plt.legend()
plt.show()


plt.plot(k, recall_all_NNMF, label='Recall')
plt.xlabel('t')
plt.ylabel('Recall')
plt.title('NNMF Recall')
plt.legend()
plt.show()


plt.plot(recall_all_NNMF, precision_all_NNMF, label='Precision vs. Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('NNMF Precision vs. Recall')
plt.legend()
plt.show()


##### Question 38 ################################
# import data from surprise
from surprise import Dataset
from surprise import Reader
import os

file_path = os.path.expanduser('ratings.csv')
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1, rating_scale= (0,5)) # IMPORTANT!!!!!
dataset = Dataset.load_from_file(file_path, reader=reader)

from surprise.model_selection import KFold
k_fold = KFold(n_splits=10)

sim = {'name':'pearson', 'user_based': True}

from collections import defaultdict
def calculate_precision(pred, t):
    # Build G by selecting all the movies that has threshold > 3
    # The key is userId, and value is movieID, true rating, and predicted rating
    G = defaultdict(list)
    for prediction in pred:
        if (prediction[2] > 3):
            G[prediction[0]].append((prediction[1], prediction[2], prediction[3]))
    for k, v in G.items():
        # if the length of list is 0, delete the user
        if (len(G[k]) == 0):
            del G[k]
    
    S = defaultdict(list)
    for prediction in pred:
        # get top predicted item S
        S[prediction[0]].append((prediction[1], prediction[2], prediction[3]))
        S[prediction[0]] = sorted(S[prediction[0]], key=lambda x: x[2], reverse=True)[:t]
    
    precision_all_users = []
    for k, v in S.items():
        if (len(S[k]) < t):
            continue
        count = 0
        for item in S[k]:
            if (item in G[k]):
                count = count + 1
        precision_each_user = count / len(S[k])
        precision_all_users.append(precision_each_user)
    
    precision_each_fold = np.mean(precision_all_users)
    
    return precision_each_fold
    
def calculate_recall(pred, t):
    # Build G by selecting all the movies that has threshold > 3
    # The key is userId, and value is movieID, true rating, and predicted rating
    G = defaultdict(list)
    for prediction in pred:
        if (prediction[2] > 3):
            G[prediction[0]].append((prediction[1], prediction[2], prediction[3]))
    for k, v in G.items():
        # if the length of list is 0, delete the user
        if (len(G[k]) == 0):
            del G[k]
    
    S = defaultdict(list)
    for prediction in pred:
        # get top predicted item S
        S[prediction[0]].append((prediction[1], prediction[2], prediction[3]))
        S[prediction[0]] = sorted(S[prediction[0]], key=lambda x: x[2], reverse=True)[:t]
    
    recall_all_users = []
    for k, v in S.items():
        if (len(S[k]) < t):
            continue
        count = 0
        for item in S[k]:
            if (item in G[k]):
                count = count + 1
        if (len(G[k]) != 0):
            recall_each_user = count / len(G[k])
            recall_all_users.append(recall_each_user)
    
    recall_each_fold = np.mean(recall_all_users)
    
    return recall_each_fold

from surprise.prediction_algorithms.matrix_factorization import SVD
svd = SVD(n_factors = 8)

    
precision_all_MF = []
recall_all_MF = []
for t in range(1,26,1):
    precision = []
    recall = []
    for trainset, testset in k_fold.split(dataset):
        svd.fit(trainset)
        pred = svd.test(testset)
        precision_each_fold = calculate_precision(pred, t)
        recall_each_fold = calculate_recall(pred, t)
        precision.append(precision_each_fold)
        recall.append(recall_each_fold)
    precision_all_MF.append(np.mean(precision))
    recall_all_MF.append(np.mean(recall))    
        


k = np.arange(1,26,1)
plt.plot(k, precision_all_MF, label='Precision')
plt.xlabel('t')
plt.ylabel('Precision')
plt.title('MF Precision')
plt.legend()
plt.show()


plt.plot(k, recall_all_MF, label='Recall')
plt.xlabel('t')
plt.ylabel('Recall')
plt.title('MF Recall')
plt.legend()
plt.show()


plt.plot(recall_all_MF, precision_all_MF, label='Precision vs. Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('MF Precision vs. Recall')
plt.legend()
plt.show()


##### Question 39 ################################
plt.plot(recall_all_MF, precision_all_MF, label='MF')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs. Recall')
plt.legend()


plt.plot(recall_all_NNMF, precision_all_NNMF, label='NNMF')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs. Recall')
plt.legend()

plt.plot(recall_all_KNN, precision_all_KNN, label='KNN')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs. Recall')
plt.legend()
plt.show()
