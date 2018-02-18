# Large Scale Data Mining

This repository contains six projects for EE219 Large-Scale Data Mining: Models and Algorithms at UCLA.

### 1. 20 Newsgroups

In this project, we work with "20 Newsgroups" dataset. It is a collection of approximately 20,000 newsgroup documents, partitioned evenly across 20 different newsgroups, each corresponding to a different topic. 

We used Latent Semantic Indexing (LSI) and Non-Negative Matrix Factorization (NMF) for dimension reduction, and implemented machine learning models such as Logestic Regression, Naive Bayes and SVM to classify these newsgroups. The performance of each model is evaluated by confusion matrix, cross validation and receiver operating characteristic (ROC) curve.


### 2. Clustering

In this project, we still work with the "20 Newsgroups" dataset, but use K-means clustering to find groups of data points that have similar representations in a proper space. 

We used TF-IDF matrix to transform the data, and then measures confusion matrix, homogeneity score, completeness score, V-measure, adjusted Rand score and the adjusted mutual info score. The first part of the project is to group the dataset by using only two clusters (Computer vs. Recreation), and the second part is group the dataset with 20 clusters corresponding to the 20 newsgroups.
