# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:18:43 2018

@author: lixud
"""

#(a)
# import the training and testing dataset
from sklearn.datasets import fetch_20newsgroups
categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware',
              'rec.autos', 'rec.motorcycles','rec.sport.baseball','rec.sport.hockey']
eight_categories_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state = 42)
eight_categories_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state = 42)

# plot the histogram
import matplotlib.pyplot as plt
import numpy as np

targets = eight_categories_train.target.tolist()
y = [targets.count(0),targets.count(1),targets.count(2),targets.count(3),targets.count(4),targets.count(5)
,targets.count(6),targets.count(7)]

 # enlarge the graph if you cannot see the x-label
x = np.arange(8)
plt.bar(x,y)
plt.xticks(x, categories,rotation='vertical')
plt.xlabel('Class Names', fontsize=18)
plt.ylabel('Training Documents', fontsize=18)




#(b)
from nltk.stem import PorterStemmer
ps = PorterStemmer()

import nltk
from sklearn.feature_extraction import text
stop_words_skt = text.ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
nltk.download('stopwords' )
stop_words_en = stopwords.words('english')
from string import punctuation
combined_stopwords = set.union(set(stop_words_en),set(punctuation),set(stop_words_skt))

from sklearn.feature_extraction.text import CountVectorizer
analyzer = CountVectorizer().build_analyzer()

from nltk import pos_tag
nltk.download('punkt')#, if you need "tokenizers/punkt/english.pickle", choose it
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
wnl = nltk.wordnet.WordNetLemmatizer()
walking_tagged = pos_tag(nltk.word_tokenize('He is walking to school'))
def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'
def lemmatize_sent(list_word): 
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(list_word)]

def stemmed_words(doc):
    return (ps.stem(w) for w in analyzer(doc))

def stem_rmv_punc(doc):
    return (word for word in lemmatize_sent(analyzer(doc)) if word not in combined_stopwords and not word.isdigit())
    
count_vect= CountVectorizer(stop_words = 'english', analyzer=stemmed_words, min_df = 2)
#count_vect= CountVectorizer(stop_words = 'english', analyzer=stem_rmv_punc, min_df = 2)
X_train_counts = count_vect.fit_transform(eight_categories_train.data)
count_vect.get_feature_names()
X_train_counts.toarray()
X_train_counts.shape
X_test_counts = count_vect.transform(eight_categories_test.data)
X_test_counts.shape
### train: 4732,24848 for min_df = 2
### test: 3150,24848
### train: 4732,10396 for min_df = 5
### test:  3150,10396

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)



#(c)
from operator import itemgetter
analyzer = CountVectorizer().build_analyzer()

from nltk.stem import PorterStemmer
ps = PorterStemmer()

newsgroup_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
len(newsgroup_train.target_names)
target_name2idx = {}

for idx, target_name in enumerate(newsgroup_train.target_names):
    target_name2idx[target_name] = idx
    
data_by_class = [""] * len(newsgroup_train.target_names)
for idx, data in enumerate(newsgroup_train.data):
    fname = newsgroup_train.filenames[idx]
    class_name = fname.split('\\')[-2] ##THIS IS DIFFERENT FOR MAC OS
    class_idx = target_name2idx[class_name]

    data_by_class[class_idx] += data + "\n"  
##seperate the data by class, in the data_by_class: all the data from dcuments combined 

count_vect = CountVectorizer(min_df=5,analyzer=stem_rmv_punc)
Xc_train_counts = count_vect.fit_transform(data_by_class)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
Xc_train_tficf = tfidf_transformer.fit_transform(Xc_train_counts)
Xc_train_tficf.shape

reverse_vocab_dict = {}
for term in count_vect.vocabulary_:
    term_idx = count_vect.vocabulary_[term]
    reverse_vocab_dict[term_idx] = term

target_classes = ['comp.sys.ibm.pc.hardware' , 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
for class_name in target_classes:
    Xc_train_tficf_array = Xc_train_tficf.toarray()
    class_idx = target_name2idx[class_name]
    sig_arr = [(idx, val) for idx, val in enumerate(Xc_train_tficf_array[class_idx])]
    top10 = sorted(sig_arr, key=itemgetter(1), reverse=True)[:10]
    #print top10
    
    print("Top 10 significant terms in class %s:" % class_name)
    for idx, val in enumerate(top10):
        term_idx, sig_val = val
        print("%-16s(significance = %f)" % (reverse_vocab_dict[term_idx], sig_val))

    print("") # new line for every target class


#(d)
# SVD for traning data
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
D_train = X_train_tfidf.transpose()
A_train = csc_matrix(D_train, dtype=float)
u_train, s_train, vt_train = svds(A_train, k=50)
UT_train = u_train.transpose()
D_ndarray_train = D_train.toarray()
Dk_train = np.matmul(UT_train, D_ndarray_train)  #50x4732 matrix
Dk_train_transpose = Dk_train.transpose()        #4732x50 matrix

# SVD for testing data
D_test = X_test_tfidf.transpose()
A_test = csc_matrix(D_test, dtype=float)
u_test, s_test, vt_test = svds(A_test, k=50)
UT_test = u_test.transpose()
D_ndarray_test = D_test.toarray()
Dk_test = np.matmul(UT_test, D_ndarray_test)  #50x3150 matrix
Dk_test_transpose = Dk_test.transpose()       #3150x50 matrix

# NMF for training data
from sklearn.decomposition import NMF
model = NMF(n_components = 50)
W_train = model.fit_transform(X_train_tfidf)
H_train = model.components_

# NMF for testing data
model = NMF(n_components = 50)
W_test = model.fit_transform(X_test_tfidf)
H_test = model.components_


# (e)
#================================================= Load Below Data First for part e==========================
# load the datasets Computer tech vs. Recrational Act
from sklearn.datasets import fetch_20newsgroups
comp_categories = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
rec_categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
eight_categories_train = fetch_20newsgroups(subset='train', categories=comp_categories+rec_categories, shuffle=True, random_state=42,)
eight_categories_test = fetch_20newsgroups(subset='test', categories=comp_categories+rec_categories, shuffle=True, random_state=42,)

#create a mapping of 0 and 1 for Computer tech vs. Recrational Act
for i in range(0, len(eight_categories_train.target)):
    if (eight_categories_train.target[i] < 4):
        eight_categories_train.target[i] = 0
    else:
        eight_categories_train.target[i] = 1

for i in range(0, len(eight_categories_test.target)):
    if (eight_categories_test.target[i] < 4):
        eight_categories_test.target[i] = 0
    else:
        eight_categories_test.target[i] = 1

#================================================= Load Above Data First for part e==========================
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

def plot_roc(fpr, tpr):
    fig, ax = plt.subplots()

    roc_auc = auc(fpr,tpr)

    ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate',fontsize=15)
    ax.set_ylabel('True Positive Rate',fontsize=15)

    ax.legend(loc="lower right")

    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(15)

# hard margin svc
hard_margin_svc = SVC(kernel = 'linear', C = 1000, probability=True, random_state = 42)
clf_hard = hard_margin_svc.fit(Dk_train_transpose, eight_categories_train.target)
predicted_hard = clf_hard.predict(Dk_test_transpose)
predicted_hard_proba = clf_hard.predict_proba(Dk_test_transpose)

# ROC hard margin
fpr_hard, tpr_hard, _ = roc_curve(eight_categories_test.target, predicted_hard_proba[:,1])
plot_roc(fpr_hard, tpr_hard)
#plot_roc(tpr_hard, fpr_hard)

# confusion matrix for hard margin
cm_hard = confusion_matrix(y_true = eight_categories_test.target, y_pred = predicted_hard)

# accuracy, recall and precision for hard margin
acc_score_hard = accuracy_score(eight_categories_test.target, predicted_hard)
pre_score_hard = precision_score(eight_categories_test.target, predicted_hard)
rec_score_hard = recall_score(eight_categories_test.target, predicted_hard)

# soft margin svc
soft_margin_svc = SVC(kernel = 'linear', C = 0.001, probability=True, random_state = 42)
clf_soft = soft_margin_svc.fit(Dk_train_transpose, eight_categories_train.target)
predicted_soft = clf_soft.predict(Dk_test_transpose)
predicted_soft_proba = clf_soft.predict_proba(Dk_test_transpose)

# confusion matrix for soft margin
cm_soft = confusion_matrix(y_true = eight_categories_test.target, y_pred = predicted_soft)

# ROC soft margin
fpr_soft, tpr_soft, _ = roc_curve(eight_categories_test.target, predicted_soft_proba[:,1])
plot_roc(fpr_soft, tpr_soft)
#plot_roc(tpr_soft, fpr_soft)

# accuracy, recall and precision for soft margin
acc_score_soft = accuracy_score(eight_categories_test.target, predicted_soft)
pre_score_soft = precision_score(eight_categories_test.target, predicted_soft)
rec_score_soft = recall_score(eight_categories_test.target, predicted_soft)



# (f)
# k-fold cross validation
from sklearn.model_selection import cross_val_score
best_margin_svc = SVC(kernel = 'linear', C = 90, probability=True)
clf_best = best_margin_svc.fit(Dk_train_transpose, eight_categories_train.target)
cross_score = cross_val_score(estimator = best_margin_svc, X = Dk_train_transpose, y = eight_categories_train.target, cv = 5)
cross_score.mean()

predicted_best = clf_best.predict(Dk_test_transpose)
predicted_best_proba = clf_best.predict_proba(Dk_test_transpose)

# confusion matrix for best margin
cm_best = confusion_matrix(y_true = eight_categories_test.target, y_pred = predicted_best)

# ROC best margin
fpr_best, tpr_best, _ = roc_curve(eight_categories_test.target, predicted_best_proba[:,1])
plot_roc(fpr_best, tpr_best)

# accuracy, recall and precision for best margin
acc_score_best = accuracy_score(eight_categories_test.target, predicted_best)
pre_score_best = precision_score(eight_categories_test.target, predicted_best)
rec_score_best = recall_score(eight_categories_test.target, predicted_best)

# (g)
from sklearn.naive_bayes import MultinomialNB 
NB_classifier = MultinomialNB()
NB_clf = NB_classifier.fit(Dk_train_transpose, eight_categories_train.target)
cross_score_NB = cross_val_score(estimator = NB_classifier, X = Dk_train_transpose, y = eight_categories_train.target, cv = 5)
cross_score_NB.mean()

predicted_best_NB = NB_clf.predict(Dk_test_transpose)
predicted_best_proba_NB = NB_clf.predict_proba(Dk_test_transpose)

# confusion matrix for best margin
cm_best_NB = confusion_matrix(y_true = eight_categories_test.target, y_pred = predicted_best_NB)

# ROC best margin
fpr_best_NB, tpr_best_NB, _ = roc_curve(eight_categories_test.target, predicted_best_proba_NB[:,1])
plot_roc(fpr_best_NB, tpr_best_NB)

# accuracy, recall and precision for best margin
acc_score_best_NB = accuracy_score(eight_categories_test.target, predicted_best_NB)
pre_score_best_NB = precision_score(eight_categories_test.target, predicted_best_NB)
rec_score_best_NB = recall_score(eight_categories_test.target, predicted_best_NB)

# (h)
from sklearn.linear_model import LogisticRegression 
LR_classifier = LogisticRegression()
LR_clf = LR_classifier.fit(Dk_train_transpose, eight_categories_train.target)
cross_score_LR = cross_val_score(estimator = LR_classifier, X = Dk_train_transpose, y = eight_categories_train.target, cv = 5)
cross_score_LR.mean()

predicted_best_LR = LR_clf.predict(Dk_test_transpose)
predicted_best_proba_LR = LR_clf.predict_proba(Dk_test_transpose)

# confusion matrix for best margin
cm_best_LR = confusion_matrix(y_true = eight_categories_test.target, y_pred = predicted_best_LR)

# ROC best margin
fpr_best_LR, tpr_best_LR, _ = roc_curve(eight_categories_test.target, predicted_best_proba_LR[:,1])
plot_roc(fpr_best_LR, tpr_best_LR)

# accuracy, recall and precision for best margin
acc_score_best_LR = accuracy_score(eight_categories_test.target, predicted_best_LR)
pre_score_best_LR = precision_score(eight_categories_test.target, predicted_best_LR)
rec_score_best_LR = recall_score(eight_categories_test.target, predicted_best_LR)


# (i)
def plot_roc_with_params(penalty, C, fpr, tpr):
    print("")
    print("penalty is "+penalty)
    print("C = 10**%d" % C)
    fig, ax = plt.subplots()

    roc_auc = auc(fpr,tpr)

    ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate',fontsize=15)
    ax.set_ylabel('True Positive Rate',fontsize=15)

    ax.legend(loc="lower right")

    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(15)
        

from sklearn.linear_model import LogisticRegression
penaltys = ['l1','l2']
c_ranges = range(-4,7,2)
for penalty in penaltys:
    for C in c_ranges:
        LR_l1_classifier = LogisticRegression(penalty=penalty, C=10**C)
        LR_l1_clf = LR_l1_classifier.fit(Dk_train_transpose, eight_categories_train.target)
        cross_score_LR_l1 = cross_val_score(estimator = LR_l1_classifier, X = Dk_train_transpose, y = eight_categories_train.target, cv = 5)
        cross_score_LR_l1.mean()

        predicted_best_LR_l1 = LR_l1_clf.predict(Dk_test_transpose)
        predicted_best_proba_LR_l1 = LR_l1_clf.predict_proba(Dk_test_transpose)

        # confusion matrix for best margin
        cm_best_LR_l1 = confusion_matrix(y_true = eight_categories_test.target, y_pred = predicted_best_LR_l1)

        # ROC best margin
        fpr_best_LR_l1, tpr_best_LR_l1, _ = roc_curve(eight_categories_test.target, predicted_best_proba_LR_l1[:,1])
        plot_roc_with_params(penalty, C, fpr_best_LR_l1, tpr_best_LR_l1)

        # accuracy, recall and precision for best margin
        acc_score_best_LR_l1 = accuracy_score(eight_categories_test.target, predicted_best_LR_l1)
        pre_score_best_LR_l1 = precision_score(eight_categories_test.target, predicted_best_LR_l1)
        rec_score_best_LR_l1 = recall_score(eight_categories_test.target, predicted_best_LR_l1)
        print("acc = %f" %acc_score_best_LR_l1)
        print("pre = %f" %pre_score_best_LR_l1)
        print("rec = %f" %rec_score_best_LR_l1)
        print(cm_best_LR_l1)
        print("==============================")
        print("")

# (i++, multiclass classification)
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import re
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk import SnowballStemmer
import numpy as np
from sklearn import metrics

categories = [
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'misc.forsale',
    'soc.religion.christian'
]
training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state = 42)
testing_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state = 42)
all_data = training_data.data+testing_data.data

stemmer = SnowballStemmer("english")
punctuations = '[! \" # $ % \& \' \( \) \* + , \- \. \/ : ; < = > ? @ \[ \\ \] ^ _ ` { \| } ~]'    
def preprocess_data(data_list):
    for i in range(len(data_list)):
        data_list[i] = " ".join([stemmer.stem(data) for data in re.split(punctuations, data_list[i])])
        data_list[i] = data_list[i].replace('\n','').replace('\t','').replace('\r','')
    
preprocess_data(all_data)

# using CountVectorizer and TFxIDF Transformer
count_vect = CountVectorizer(min_df=2, stop_words ='english')
X_counts = count_vect.fit_transform(all_data)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# apply LSI to TDxIDF matrices
svd = TruncatedSVD(n_components = 50, n_iter = 10,random_state = 42)
svd.fit(X_tfidf)
LSI = svd.transform(X_tfidf)

from sklearn.decomposition import NMF
model = NMF(n_components = 50)
W = model.fit_transform(X_tfidf)

W_train = W[0:len(training_data.data)]
W_test = W[len(training_data.data):]

LSI_train = LSI[0:len(training_data.data)]
LSI_test = LSI[len(training_data.data):]

def calculate_statistics(target, predicted):
    
    print('\n                       Classification Report:')
    print('==================================================================')
    print(metrics.classification_report(target, predicted, target_names=['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']),)
    print("acuracy is %f" %metrics.accuracy_score(target,predicted))
    print('==================================================================\n')
    
    print('Confusion Matrix:')
    print('===================')
    print(metrics.confusion_matrix(target, predicted))
    print('===================\n')
    
    print('Total Accuracy: ')
    print(np.mean(target == predicted))
    
    
clf_list = [OneVsOneClassifier(GaussianNB()), OneVsOneClassifier(svm.LinearSVC()), OneVsRestClassifier(GaussianNB()), OneVsRestClassifier(svm.LinearSVC())]
clf_name = ['One vs One Classifier - Naive Bayes', 'One vs One Classifier - SVM','One vs Rest Classifier - Naive Bayes', 'One vs Rest Classifier - SVM']


# perform classification
for clf,clf_n in zip(clf_list,clf_name):
    pound_sign = ''
    spaces = ''
    for i in range(len(clf_n)+2):
        pound_sign += '#'
        spaces += ' '
    print('\n\n\n\n')
    print('#' + pound_sign + '#')
    print('#' + spaces + '#')
    print('# ' + clf_n + ' #')
    print('#' + pound_sign + '#')
    
    clf.fit(W_train, training_data.target)
  
    test_predicted = clf.predict(W_test)
    calculate_statistics(testing_data.target, test_predicted)
















