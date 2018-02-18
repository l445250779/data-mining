# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:55:36 2018

@author: Xudong Li, Zeyu Jin
"""

from sklearn.datasets import fetch_20newsgroups
comp_categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
rec_categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
eight_categories = fetch_20newsgroups(subset='all', categories=comp_categories+rec_categories, shuffle=True, random_state=42)


#create a mapping of 0 and 1 for Computer tech vs. Recrational Act
for i in range(0, len(eight_categories.target)):
    if (eight_categories.target[i] < 4):
        eight_categories.target[i] = 0
    else:
        eight_categories.target[i] = 1




#### Part 1 Build TF-IDF matrix ##########################
    
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 3)
X = vectorizer.fit_transform(eight_categories.data)
vectorizer.get_feature_names()
X.toarray()
X.shape

# 7882x27768



#### Part 2 K-means clustering ##########################

from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=2, init='k-means++',max_iter = 300, n_init=10)
predicted = k_means.fit(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true = eight_categories.target, y_pred = predicted.labels_)

from sklearn import metrics
homogenity = metrics.homogeneity_score(eight_categories.target, predicted.labels_)
completeness = metrics.completeness_score(eight_categories.target, predicted.labels_)
V_measure = metrics.v_measure_score(eight_categories.target, predicted.labels_)
adjusted_rand = metrics.adjusted_rand_score(eight_categories.target, predicted.labels_)
adjusted_mutual_info = metrics.adjusted_mutual_info_score(eight_categories.target, predicted.labels_)



#### Part 3 K-means clustering ##########################

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
D = X.transpose()
A = csc_matrix(D, dtype=float)
u, s, vt = svds(A, k=1000)
UT = u.transpose()
D_ndarray = D.toarray()
Dk = np.matmul(UT, D_ndarray)  
Dk_transpose = Dk.transpose()
Dk_transpose.shape        # 7882x1000


### ratio = var_retained/total_variance
### var_retained = tr(Xr^T * Xr) = sum of S from 1 to r
### total_variance = tr(X^T * X) = tr(X * X^T)

r = np.arange(1,1001,1)
s_large2small = np.flip(s, 0)
total_var = np.trace(np.matmul(X.toarray(),X.toarray().transpose()))
ratio = [];
for i in range(1, 1001):
    ratio.append(np.sum(s_large2small[:i])/total_var)
    
import matplotlib.pyplot as plt
plt.plot(r, ratio)
plt.xlabel('r', fontsize=18)
plt.ylabel('% of variance retained', fontsize=18)
    

###================for LSI===================

from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.cluster import contingency_matrix
def get_scores(y, yhat):
    cont_matrix = confusion_matrix(y_true = y, y_pred = yhat)

    homogeneity = metrics.homogeneity_score(y, yhat)
    completeness_score = metrics.completeness_score(y, yhat)
    v_measure_score = metrics.v_measure_score(y, yhat)
    adjusted_rand_score = metrics.adjusted_rand_score(y, yhat)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(y, yhat)
    return cont_matrix, homogeneity, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score

r_vals = [1,2,3,5,10,20,50,100,300]
contingency_matrices = []
h_vals = []
c_vals = []
v_vals = []
rand_vals = []
mutual_vals = []

print("for LSI")
for r in r_vals:
    svd = TruncatedSVD(n_components=r)
    lsi = svd.fit_transform(X)
    kmeans = KMeans(n_clusters=2, init='k-means++',max_iter = 300, n_init=10)
    predicted = kmeans.fit(lsi)
    contin, h, c, v, rand, mutual = get_scores(eight_categories.target, predicted.labels_ )
    print("contingency matrix for r=", r)

    print(contin)
    h_vals.append(h)
    c_vals.append(c)
    v_vals.append(v)
    rand_vals.append(rand)
    mutual_vals.append(mutual)
    
for r, h, c, v, rand, mutual in zip(r_vals, h_vals, c_vals, v_vals, rand_vals, mutual_vals):
    print("r: %i \n homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" % (r, h, c, v, rand, mutual))


res_list = [np.argmax(h_vals), np.argmax(c_vals), np.argmax(v_vals), np.argmax(rand_vals), np.argmax(mutual_vals)]
print(res_list)
# get index for each element's max value
best_index = np.bincount(res_list).argmax() # most frequent index
print(best_index)
    
print("Best r-value is: ", r_vals[best_index])
plt.plot(r_vals, h_vals, 'r', label="homogenity")
plt.plot(r_vals, c_vals, 'g', label="completeness score")
plt.plot(r_vals, v_vals, 'b', label="v measure")
plt.plot(r_vals, rand_vals, 'pink', label="rand index")
plt.plot(r_vals, mutual_vals, 'y', label="adjusted mutual information")
plt.grid()


###================for NMF===================
contingency_matrices = []
h_vals = []
c_vals = []
v_vals = []
rand_vals = []
mutual_vals = []

print("for NMF")
for r in r_vals:
    nmf_reduction = NMF(n_components=r)
    nmf = nmf_reduction.fit_transform(X)
    
    kmeans = KMeans(n_clusters=2, init='k-means++',max_iter = 300, n_init=10)
    predicted = kmeans.fit(nmf)
    contin, h, c, v, rand, mutual = get_scores(eight_categories.target, predicted.labels_ )
    print("contingency matrix for r=", r)
    
    print(contin)
    h_vals.append(h)
    c_vals.append(c)
    v_vals.append(v)
    rand_vals.append(rand)
    mutual_vals.append(mutual)
    
    
for r, h, c, v, rand, mutual in zip(r_vals, h_vals, c_vals, v_vals, rand_vals, mutual_vals):
    print("r: %i \n homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" % (r, h, c, v, rand, mutual))
    
res_list = [np.argmax(h_vals), np.argmax(c_vals), np.argmax(v_vals), np.argmax(rand_vals), np.argmax(mutual_vals)]
print(res_list)
# get index for each element's max value
best_index = np.bincount(res_list).argmax() # most frequent index
print(best_index)
    
print("Best r-value is: ", r_vals[best_index])
plt.plot(r_vals, h_vals, 'r', label="homogenity")
plt.plot(r_vals, c_vals, 'g', label="completeness score")
plt.plot(r_vals, v_vals, 'b', label="v measure")
plt.plot(r_vals, rand_vals, 'pink', label="rand index")
plt.plot(r_vals, mutual_vals, 'y', label="adjusted mutual information")
plt.grid()

##====Q&A=============================
# How do you explain the non-monotonic behavior of the measures
# as r increases?

#Answer: As r grows, the variance of data increases, which means more and more features are included. However, when r
# is bigger, the Euclidean distance performs worse, because the distances between data points tends to be almost 
# the same. Thus, the (r=2) get better results than bigger rs.


##=============Problem 4(a)================
from sklearn.preprocessing import scale

#LSI
svd = TruncatedSVD(n_components=2)
lsi = svd.fit_transform(X)
kmeans_lsi = KMeans(n_clusters=2, init='k-means++',max_iter = 300, n_init=10)
predicted_lsi = kmeans_lsi.fit(lsi)

#NMF
nmf_reduction = NMF(n_components=2)
nmf = nmf_reduction.fit_transform(X)
kmeans_nmf = KMeans(n_clusters=2, init='k-means++',max_iter = 300, n_init=10)
predicted_nmf = kmeans_nmf.fit(nmf)

def plot_clusters(X2 ,predicted):
    
    ##red and blue for different class
    x1 = X2[predicted.labels_ == 0][:, 0]
    y1 = X2[predicted.labels_ == 0][:, 1]
    plt.plot(x1,y1,'r+')
    x2 = X2[predicted.labels_ == 1][:, 0]
    y2 = X2[predicted.labels_ == 1][:, 1]
    plt.plot(x2, y2, 'b+')
    ##white X for center of clusters
    centroids = predicted.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3,color='w', zorder=10)
    plt.show()
    
plot_clusters(X2=lsi, predicted = predicted_lsi)

plot_clusters(X2=nmf, predicted = predicted_nmf)

##=============Problem 4(b)================

# Normalizing features after SVD
from sklearn.preprocessing import normalize
lsi_norm = normalize(lsi)
kmeans_lsi_norm = KMeans(n_clusters=2, init='k-means++',max_iter = 300, n_init=10)
predicted_lsi_norm = kmeans_lsi_norm.fit(lsi_norm)
plot_clusters(X2=lsi_norm, predicted = predicted_lsi_norm)
contin, h, c, v, rand, mutual = get_scores(eight_categories.target, predicted_lsi_norm.labels_ )
print("contingency matrix:")
print(contin)
print(" homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" 
      % (h, c, v, rand, mutual))


# Normalizing features after NMF
nmf_norm = normalize(nmf)
kmeans_nmf_norm = KMeans(n_clusters=2, init='k-means++',max_iter = 300, n_init=10)
predicted_nmf_norm = kmeans_nmf_norm.fit(nmf_norm)
plot_clusters(X2=nmf_norm, predicted = predicted_nmf_norm)
contin, h, c, v, rand, mutual = get_scores(eight_categories.target, predicted_nmf_norm.labels_ )
print("contingency matrix:")
print(contin)
print(" homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" 
      % (h, c, v, rand, mutual))

# logarithmic transformation after NMF
from sklearn.preprocessing import FunctionTransformer
logtransformer = FunctionTransformer(np.log1p)
nmf_log = logtransformer.transform(nmf)
kmeans_nmf_log = KMeans(n_clusters=2, init='k-means++',max_iter = 300, n_init=10)
predicted_nmf_log = kmeans_nmf_log.fit(nmf_log)
plot_clusters(X2=nmf_log, predicted = predicted_nmf_log)
contin, h, c, v, rand, mutual = get_scores(eight_categories.target, kmeans_nmf_log.labels_ )
print("contingency matrix:")
print(contin)
print(" homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" 
      % (h, c, v, rand, mutual))

##=============Problem 4(c)================

# norm + log after NMF
nmf_norm = normalize(nmf)
nmf_norm_log = logtransformer.transform(nmf_norm)
kmeans_nmf_norm_log = KMeans(n_clusters=2, init='k-means++',max_iter = 300, n_init=10)
predicted_nmf_norm_log = kmeans_nmf_norm_log.fit(nmf_norm_log)
plot_clusters(X2=nmf_norm_log, predicted = predicted_nmf_norm_log)
contin, h, c, v, rand, mutual = get_scores(eight_categories.target, predicted_nmf_norm_log.labels_ )
print("contingency matrix:")
print(contin)
print(" homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" 
      % (h, c, v, rand, mutual))


# log + norm after NMF
nmf_log = logtransformer.transform(nmf)
nmf_log_norm = normalize(nmf_log)
kmeans_nmf_log_norm = KMeans(n_clusters=2, init='k-means++',max_iter = 300, n_init=10)
predicted_nmf_log_norm = kmeans_nmf_log_norm.fit(nmf_log_norm)
plot_clusters(X2=nmf_log_norm, predicted = predicted_nmf_log_norm)
contin, h, c, v, rand, mutual = get_scores(eight_categories.target, predicted_nmf_log_norm.labels_ )
print("contingency matrix:")
print(contin)
print(" homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" 
      % (h, c, v, rand, mutual))

##====Q&A=============================
# Can you justify why logarithm transformation may increase the
# clustering results?

#Answer: logarithm transformation does not help...

##=============Problem 5================
from sklearn.datasets import fetch_20newsgroups

all_data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42,  remove=('headers','footers','quotes'))
vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 3)
all_data_tfidf = vectorizer.fit_transform(all_data.data)

def get_scores(y, yhat):
    cont_matrix = confusion_matrix(y_true = y, y_pred = yhat)
    #plot_mat(mat=cont_matrix)
    homogeneity = metrics.homogeneity_score(y, yhat)
    completeness_score = metrics.completeness_score(y, yhat)
    v_measure_score = metrics.v_measure_score(y, yhat)
    adjusted_rand_score = metrics.adjusted_rand_score(y, yhat)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(y, yhat)
    return cont_matrix, homogeneity, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score


###------- plot Confusion matrix for the best R-------

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
def plot_mat(mat, xticklabels = None, yticklabels = None, pic_fname = None, size=(-1,-1), if_show_values = True,
             colorbar = True, grid = 'k', xlabel = None, ylabel = None):
    if size == (-1, -1):
        size = (mat.shape[1] / 3, mat.shape[0] / 3)

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(1,1,1)

    # im = ax.imshow(mat, cmap=plt.cm.Blues)
    im = ax.pcolor(mat, cmap=plt.cm.Blues, linestyle='-', linewidth=0.5, edgecolor=grid)
    
    if colorbar:
        plt.colorbar(im,fraction=0.046, pad=0.06)
    # tick_marks = np.arange(len(classes))
    # Ticks
    lda_num_topics = mat.shape[0]
    nmf_num_topics = mat.shape[1]
    yticks = np.arange(lda_num_topics)
    xticks = np.arange(nmf_num_topics)
    ax.set_xticks(xticks + 0.5)
    ax.set_yticks(yticks + 0.5)
    if not xticklabels:
        xticklabels = [str(i) for i in xticks]
    if not yticklabels:
        yticklabels = [str(i) for i in yticks]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)

    # Minor ticks
    # ax.set_xticks(xticks, minor=True);
    # ax.set_yticks(yticks, minor=True);
    # ax.set_xticklabels([], minor=True)
    # ax.set_yticklabels([], minor=True)

    # ax.grid(which='minor', color='k', linestyle='-', linewidth=0.5)

    # tick labels on all four sides
    ax.tick_params(labelright = True, labeltop = True)

    if ylabel:
        plt.ylabel(ylabel, fontsize=15)
    if xlabel:
        plt.xlabel(xlabel, fontsize=15)

    # im = ax.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)
    ax.invert_yaxis()

    # thresh = mat.max() / 2

    def show_values(pc, fmt="%.3f", **kw):
        pc.update_scalarmappable()
        ax = pc.axes
        for p, color, value in itertools.zip_longest(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            if np.all(color[:3] > 0.5):
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw, fontsize=4)

    if if_show_values:
        show_values(im)
    # for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
    #     ax.text(j, i, "{:.2f}".format(mat[i, j]), fontsize = 4,
    #              horizontalalignment="center",
    #              color="white" if mat[i, j] > thresh else "black")

    if pic_fname:
        plt.savefig(pic_fname, dpi=300, transparent=True)
        
#------------------------------end of function-----------------------

# ================= all data without dimension reduction ===================
km = KMeans(n_clusters=20, init='k-means++',max_iter = 300 )
predicted = km.fit(all_data_tfidf)

contin, h, c, v, rand, mutual = get_scores(all_data.target, predicted.labels_ )
plot_mat(contin)
print(" homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" 
      % (h, c, v, rand, mutual))


###================for LSI===================

from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.cluster import contingency_matrix
def get_scores(y, yhat):
    cont_matrix = confusion_matrix(y_true = y, y_pred = yhat)

    homogeneity = metrics.homogeneity_score(y, yhat)
    completeness_score = metrics.completeness_score(y, yhat)
    v_measure_score = metrics.v_measure_score(y, yhat)
    adjusted_rand_score = metrics.adjusted_rand_score(y, yhat)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(y, yhat)
    return cont_matrix, homogeneity, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score

r_vals = [1,2,3,5,10,20,50,100,300]
contingency_matrices = []
h_vals = []
c_vals = []
v_vals = []
rand_vals = []
mutual_vals = []

print("for LSI")
for r in r_vals:
    svd = TruncatedSVD(n_components=r)
    lsi = svd.fit_transform(all_data_tfidf)
    kmeans = KMeans(n_clusters=20, init='k-means++',max_iter = 300, n_init=30)
    predicted = kmeans.fit(lsi)
    contin, h, c, v, rand, mutual = get_scores(all_data.target, predicted.labels_ )
    print("contingency matrix for r=", r)

    print(contin)
    h_vals.append(h)
    c_vals.append(c)
    v_vals.append(v)
    rand_vals.append(rand)
    mutual_vals.append(mutual)
    
for r, h, c, v, rand, mutual in zip(r_vals, h_vals, c_vals, v_vals, rand_vals, mutual_vals):
    print("r: %i \n homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" % (r, h, c, v, rand, mutual))


res_list = [np.argmax(h_vals), np.argmax(c_vals), np.argmax(v_vals), np.argmax(rand_vals), np.argmax(mutual_vals)]
print(res_list)
# get index for each element's max value
best_index = np.bincount(res_list).argmax() # most frequent index
print(best_index)
    
print("Best r-value is: ", r_vals[best_index])
plt.plot(r_vals, h_vals, 'r', label="homogenity")
plt.plot(r_vals, c_vals, 'g', label="completeness score")
plt.plot(r_vals, v_vals, 'b', label="v measure")
plt.plot(r_vals, rand_vals, 'pink', label="rand index")
plt.plot(r_vals, mutual_vals, 'y', label="adjusted mutual information")
plt.legend()
plt.grid()


best = 100
svd_best = TruncatedSVD(n_components=best)
lsi_best = svd.fit_transform(all_data_tfidf)
kmeans_best = KMeans(n_clusters=20, init='k-means++',max_iter = 300, n_init=30)
predicted_best = kmeans.fit(lsi_best)
contin, h, c, v, rand, mutual = get_scores(all_data.target, predicted_best.labels_ )
plot_mat(contin)




###================for NMF===================
contingency_matrices = []
h_vals = []
c_vals = []
v_vals = []
rand_vals = []
mutual_vals = []

print("for NMF")
for r in r_vals:
    nmf_reduction = NMF(n_components=r)
    nmf = nmf_reduction.fit_transform(all_data_tfidf)
    
    kmeans = KMeans(n_clusters=20, init='k-means++',max_iter = 300, n_init=30)
    predicted = kmeans.fit(nmf)
    contin, h, c, v, rand, mutual = get_scores(all_data.target, predicted.labels_ )
    print("contingency matrix for r=", r)
    
    print(contin)
    h_vals.append(h)
    c_vals.append(c)
    v_vals.append(v)
    rand_vals.append(rand)
    mutual_vals.append(mutual)
    
    
for r, h, c, v, rand, mutual in zip(r_vals, h_vals, c_vals, v_vals, rand_vals, mutual_vals):
    print("r: %i \n homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" % (r, h, c, v, rand, mutual))
    
res_list = [np.argmax(h_vals), np.argmax(c_vals), np.argmax(v_vals), np.argmax(rand_vals), np.argmax(mutual_vals)]
print(res_list)
# get index for each element's max value
best_index = np.bincount(res_list).argmax() # most frequent index
print(best_index)
    
print("Best r-value is: ", r_vals[best_index])
plt.plot(r_vals, h_vals, 'r', label="homogenity")
plt.plot(r_vals, c_vals, 'g', label="completeness score")
plt.plot(r_vals, v_vals, 'b', label="v measure")
plt.plot(r_vals, rand_vals, 'pink', label="rand index")
plt.plot(r_vals, mutual_vals, 'y', label="adjusted mutual information")
plt.legend()
plt.grid()

best = 10
nmf_reduction = NMF(n_components=best)
nmf = nmf_reduction.fit_transform(all_data_tfidf)
kmeans = KMeans(n_clusters=20, init='k-means++',max_iter = 300, n_init=30)
predicted = kmeans.fit(nmf)
contin, h, c, v, rand, mutual = get_scores(all_data.target, predicted.labels_ )
plot_mat(contin)


##=============Projection (similar to 4a) ================
from sklearn.preprocessing import scale

#LSI
svd = TruncatedSVD(n_components=100)
lsi = svd.fit_transform(all_data_tfidf)
kmeans_lsi = KMeans(n_clusters=20, init='k-means++',max_iter = 300, n_init=10)
predicted_lsi = kmeans_lsi.fit(lsi)

#NMF
nmf_reduction = NMF(n_components=10)
nmf = nmf_reduction.fit_transform(all_data_tfidf)
kmeans_nmf = KMeans(n_clusters=20, init='k-means++',max_iter = 300, n_init=10)
predicted_nmf = kmeans_nmf.fit(nmf)

def plot_clusters(X2 ,predicted):
    colors_20 = ['b+', 'g+', 'r+', 'c+', 'm+', 'y+', 'k+', '#5aa5c1+','#8c766b+', '#b4adb4+', '#46de8b+','#46f1f2+'
                 '#18232d+','#d0ca2d+','#d0eb2d+','#d0ebe3+','#808bf7+','#903a3a+','#efa53a+','#efa5c1+']
    for i in range(0,20):
        plt.plot(X2[predicted.labels_ == i][:, 0], X2[predicted.labels_ == i][:, 1], colors_20[i])
     
    ##white X for center of clusters
    centroids = predicted.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],marker='x', s=169, linewidths=3,color='w', zorder=10)
    plt.show()
    
plot_clusters(X2=lsi, predicted = predicted_lsi)

plot_clusters(X2=nmf, predicted = predicted_nmf)


# Normalizing features after SVD
from sklearn.preprocessing import normalize
lsi_norm = normalize(lsi)
kmeans_lsi_norm = KMeans(n_clusters=20, init='k-means++',max_iter = 300, n_init=10)
predicted_lsi_norm = kmeans_lsi_norm.fit(lsi_norm)
plot_clusters(X2=lsi_norm, predicted = predicted_lsi_norm)
contin, h, c, v, rand, mutual = get_scores(all_data.target, predicted_lsi_norm.labels_)
plot_mat(mat=contin)
print(" homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" 
      % (h, c, v, rand, mutual))

# Normalizing features after NMF
nmf_norm = normalize(nmf)
kmeans_nmf_norm = KMeans(n_clusters=20, init='k-means++',max_iter = 300, n_init=10)
predicted_nmf_norm = kmeans_nmf_norm.fit(nmf_norm)
plot_clusters(X2=nmf_norm, predicted = predicted_nmf_norm)
contin, h, c, v, rand, mutual = get_scores(all_data.target, predicted_nmf_norm.labels_)
plot_mat(mat=contin)
print(" homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" 
      % (h, c, v, rand, mutual))

# logarithmic transformation after NMF
from sklearn.preprocessing import FunctionTransformer
logtransformer = FunctionTransformer(np.log1p)
nmf_log = logtransformer.transform(nmf)
kmeans_nmf_log = KMeans(n_clusters=20, init='k-means++',max_iter = 300, n_init=10)
predicted_nmf_log = kmeans_nmf_log.fit(nmf_log)
plot_clusters(X2=nmf_log, predicted = predicted_nmf_log)
contin, h, c, v, rand, mutual = get_scores(all_data.target, kmeans_nmf_log.labels_)
plot_mat(mat=contin)
print(" homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" 
      % (h, c, v, rand, mutual))

# norm + log after NMF
nmf_norm = normalize(nmf)
nmf_norm_log = logtransformer.transform(nmf_norm)
kmeans_nmf_norm_log = KMeans(n_clusters=20, init='k-means++',max_iter = 300, n_init=10)
predicted_nmf_norm_log = kmeans_nmf_norm_log.fit(nmf_norm_log)
plot_clusters(X2=nmf_norm_log, predicted = predicted_nmf_norm_log)
contin, h, c, v, rand, mutual = get_scores(all_data.target, predicted_nmf_norm_log.labels_ )
plot_mat(mat=contin)
print(" homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" 
      % (h, c, v, rand, mutual))

# log + norm after NMF
nmf_log = logtransformer.transform(nmf)
nmf_log_norm = normalize(nmf_log)
kmeans_nmf_log_norm = KMeans(n_clusters=20, init='k-means++',max_iter = 300, n_init=10)
predicted_nmf_log_norm = kmeans_nmf_log_norm.fit(nmf_log_norm)
plot_clusters(X2=nmf_log_norm, predicted = predicted_nmf_log_norm)
contin, h, c, v, rand, mutual = get_scores(all_data.target, predicted_nmf_log_norm.labels_ )
plot_mat(mat=contin)
print(" homogeneity=%f \n completeness=%f \n v-measure=%f \n adj rand index=%f \n adj mutual info=%f \n-----------------" 
      % (h, c, v, rand, mutual))
