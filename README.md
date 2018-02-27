# Large Scale Data Mining

This repository contains six projects for EE219 Large-Scale Data Mining: Models and Algorithms at UCLA.

### 1. 20 Newsgroups

In this project, we work with "20 Newsgroups" dataset. It is a collection of approximately 20,000 newsgroup documents, partitioned evenly across 20 different newsgroups, each corresponding to a different topic. 

We used Latent Semantic Indexing (LSI) and Non-Negative Matrix Factorization (NMF) for dimension reduction, and implemented machine learning models such as Logestic Regression, Naive Bayes and SVM to classify these newsgroups. The performance of each model is evaluated by confusion matrix, cross validation and receiver operating characteristic (ROC) curve.


### 2. Clustering

In this project, we still work with the "20 Newsgroups" dataset, but use K-means clustering to find groups of data points that have similar representations in a proper space. 

We used TF-IDF matrix to transform the data, and then measures confusion matrix, homogeneity score, completeness score, V-measure, adjusted Rand score and the adjusted mutual info score. The first part of the project is to group the dataset by using only two clusters (Computer vs. Recreation), and the second part is group the dataset with 20 clusters corresponding to the 20 newsgroups.


### 3. Collaborative Filtering

The increasing importance of the web as a medium for electronic and business transactions has served as a driving force for the development of recommender systems technology. An important catalyst in this regard is the ease with which the web enables users to provide feedback about their likes or dislikes. The basic idea of recommender systems is to utilize these user data to infer customer interests. The entity to which the recommendation is provided is referred to as the user, and the product being recommended is referred to as an item. 

There are two basic models for recommender systems: a user-item interactions such as ratings, and attribute information about the users and items such as textual profiles or relevant keywords. In this project, we choose the user-item interactions referred as collaborative filtering method, and implement user-based collaborative filtering by using Person-Correlation Coefficient, K-Nearest Neighborhood, Non-Negative Matrix Factorization, and Matrix Factorization with bias through SVD. 
