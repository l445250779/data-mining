# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 15:17:51 2018

@author: lixud
"""

################################# Problem 1 ###################################
"""
Created on Fri Mar  9 16:07:52 2018

@author: oliviajin
"""
import json
import matplotlib.pyplot as plt
import datetime
import pytz
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import nltk

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

stop_words = text.ENGLISH_STOP_WORDS

def plot_roc(fpr, tpr):
    fig, ax = plt.subplots()
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label='area under curve = %0.4f' % roc_auc)
    ax.grid(color='0.7', linestyle='--', linewidth=1)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=15)
    ax.set_ylabel('True Positive Rate', fontsize=15)
    ax.legend(loc="lower right")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(15)


def fit_predict_and_plot_roc(pipe, train_data, train_label, test_data, test_label):
    pipe.fit(train_data, train_label)
    prob_score = pipe.predict_proba(test_data)
    fpr, tpr, _ = roc_curve(test_label, prob_score[:, 1])
    plot_roc(fpr, tpr)
    return pipe

def output(test_label, predict_label):
    cm = confusion_matrix(test_label, predict_label)
    accuracy = accuracy_score(test_label, predict_label)
    recall = recall_score(test_label, predict_label, average='weighted', pos_label=1)
    precision = precision_score(test_label, predict_label, average='weighted', pos_label=1)

    print('Confusion Matrix:\n', cm)
    print("Accuracy: ", accuracy)
    print('Recall:', recall)
    print('Precision:', precision)
hashtags = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']
# file_names = ['tweet_data/tweets_#gohawks.txt', 'tweet_data/tweets_#gopatriots.txt', 
#              'tweet_data/tweets_#nfl.txt', 'tweet_data/tweets_#patriots.txt', 
#             'tweet_data/tweets_#sb49.txt', 'tweet_data/tweets_#superbowl.txt']
# hashtags = ['gopatriots']
file_names = []
tweet_data = {}

# Load data
for i, hashtag in enumerate(hashtags):
    file_name = 'tweet_data/tweets_#' + hashtag + '.txt'
    file_names.append(file_name)
    with open(file_name, encoding='utf-8') as f:
        print('Now loading',hashtag)
        file_data = []
        for line in f:
            file_data.append(json.loads(line))
        
        tweet_data[hashtag] = file_data
print()
 

########### Problem 1(1) ######################################################
for hashtag in hashtags:
    data = tweet_data[hashtag]
    print('Statistics For', hashtag)
    
    # Counting average tweets per hour
    retweet_time = []
    for tweet in data:
        retweet_time.append(tweet['citation_date'])
    start_time = min(retweet_time)
    end_time = max(retweet_time)
    total_time = (end_time - start_time) / 3600
    total_tweets = len(data)
    tweets_per_hour = total_tweets / total_time
    print('Average numver of tweets per hour = %.2f' % tweets_per_hour)
        
    # Counting average number of followers of users posting the tweets
    follower_count = 0
    retweets_count = 0
    for tweet in data:
        follower_count += tweet['author']['followers']
        retweets_count += tweet['metrics']['citations']['total']
    total_tweets = len(data)
    avg_follower = follower_count / total_tweets
    avg_retweets = retweets_count / total_tweets
    print('Average number of followers of users posting the tweets = %.2f' % avg_follower)
    print('Average number of retweets = %.2f' % avg_retweets)
    print()
    

# Plot
for hashtag in hashtags:
    if hashtag == 'nfl' or hashtag == 'superbowl':
        data = tweet_data[hashtag]
        time_bins = np.arange(start_time, end_time, 3600)
    
        bins = []
        pst_tz = pytz.timezone('US/Pacific')
        for time_bin in time_bins:
            bins.append(datetime.datetime.fromtimestamp(time_bin, pst_tz))
    
        tweets = []
        for tweet in data:
            tweets.append(tweet['citation_date'])
        plt.figure()
        plt.hist(tweets, time_bins)
        plt.ylabel('Number of tweets')
        plt.xlabel('Time(US/Pacific)')
        labels = np.arange(start_time, end_time, 360000)
        bins = []
        for label in labels:
            bins.append(datetime.datetime.fromtimestamp(label, pst_tz))
        plt.xticks(labels,bins,rotation=90)
        title = 'Number of tweets in hour of #' + hashtag
        plt.title(title)
plt.show()

########### Problem 1(2) ######################################################
from collections import defaultdict

for hashtag in hashtags:
    data = tweet_data[hashtag]
    print('Model Analysis for', hashtag)

    # Define a dictionary
    retweet_time = []
    tweet_dict = defaultdict(list)
    for tweet in data:
        retweet_time.append(tweet['citation_date'])
    start_time = min(retweet_time)
    end_time = max(retweet_time)
    time_bins = np.arange(start_time, end_time+3600, 3600)
    for time_bin in time_bins:
        tweet_dict[time_bin] = []
    for tweet in data:
        tweet_time = tweet['citation_date']
        for i, time_bin in enumerate(time_bins):
            if time_bins[i] <= tweet_time < time_bins[i+1]:
                tweet_dict[time_bin].append(tweet)
                
    # Extract Features
    feature = np.zeros((len(time_bins),5))
    next_hour_amount = np.zeros((len(time_bins)-1,1))
    for i,key in enumerate(tweet_dict):
        data = tweet_dict[key]

        pst_tz = pytz.timezone('US/Pacific')
        time_of_the_day = int(datetime.datetime.fromtimestamp(key, pst_tz).strftime('%H'))
        total_tweets = len(data)
        total_retweets = 0
        total_followers = 0
        maximum_followers = 0
    
        for tweet in data:
            total_followers += tweet['author']['followers']
            total_retweets += tweet['metrics']['citations']['total']
            if tweet['author']['followers'] > maximum_followers:
                maximum_followers = tweet['author']['followers']
        feature[i] = np.array([int(total_tweets), int(total_retweets), int(total_followers),
                               int(maximum_followers), time_of_the_day])
        if i>0 :
            next_hour_amount[i-1] = int(total_tweets)
        
    y = next_hour_amount
    X = feature[:y.shape[0]]
    
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    RMSE = np.sqrt(mean_squared_error(y, pred))
    R2_score =  r2_score(y, pred)
    results = sm.OLS(y, X).fit()
    print('RMSE = %.4f' % RMSE)
    print('R2_score = %.4f' % R2_score)
    print(results.summary())
    print()
  


##################### Problem 1.3 #############################################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 18:48:40 2018

@author: yangyang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:07:52 2018

@author: oliviajin
"""
import json
import matplotlib.pyplot as plt
import datetime
import pytz
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


hashtags = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']
# file_names = ['tweet_data/tweets_#gohawks.txt', 'tweet_data/tweets_#gopatriots.txt', 
#              'tweet_data/tweets_#nfl.txt', 'tweet_data/tweets_#patriots.txt', 
#             'tweet_data/tweets_#sb49.txt', 'tweet_data/tweets_#superbowl.txt']

file_names = []
tweet_data = {}

# Load data
for i, hashtag in enumerate(hashtags):
    file_name = 'tweet_data/tweets_#' + hashtag + '.txt'
    file_names.append(file_name)
    with open(file_name, encoding='utf-8') as f:
        print('Now loading',hashtag)
        file_data = []
        for line in f:
            file_data.append(json.loads(line))
        
        tweet_data[hashtag] = file_data
print()


from collections import defaultdict

for hashtag in hashtags:
    data = tweet_data[hashtag]
    print('Model Analysis for', hashtag)

    # Define a dictionary
    retweet_time = []
    tweet_dict = defaultdict(list)
    for tweet in data:
        retweet_time.append(tweet['citation_date'])
    start_time = min(retweet_time)
    end_time = max(retweet_time)
    time_bins = np.arange(start_time, end_time+3600, 3600)
    for time_bin in time_bins:
        tweet_dict[time_bin] = []
    for tweet in data:
        tweet_time = tweet['citation_date']
        for i, time_bin in enumerate(time_bins):
            if time_bins[i] <= tweet_time < time_bins[i+1]:
                tweet_dict[time_bin].append(tweet)
                
    # Extract Features
    feature = np.zeros((len(time_bins),10))
    next_hour_amount = np.zeros((len(time_bins)-1,1))
    for i,key in enumerate(tweet_dict):
        data = tweet_dict[key]

        pst_tz = pytz.timezone('US/Pacific')
        time_of_the_day = int(datetime.datetime.fromtimestamp(key, pst_tz).strftime('%H'))
        total_tweets = len(data)
        total_retweets = 0
        total_followers = 0
        maximum_followers = 0
        total_replies = 0
        count_impression = 0
        ranking_scores = 0
        num_favorite = 0
        user_id=0
        feature_names = ['Number of Tweets', 'Number of Retweets', 'Number of Followers', 'Max Number of Followers', 'Total Number of Replies',
                         'Count of Impressions', 'Favorite Count', 'Ranking Score','user_id','Time of Day']
    
        for tweet in data:
            total_followers += tweet['author']['followers']
            total_retweets += tweet['metrics']['citations']['total']
            
            total_replies += tweet['metrics']['citations']['replies']
            count_impression +=tweet['metrics']['impressions']
            num_favorite +=tweet['tweet']['favorite_count']
            ranking_scores +=tweet['metrics']['ranking_score']
            user_id += tweet['tweet']['user']['id']
            
            
            if tweet['author']['followers'] > maximum_followers:
                maximum_followers = tweet['author']['followers']
        feature[i] = np.array([int(total_tweets), int(total_retweets), int(total_followers),int(maximum_followers),
                               int(total_replies),int(count_impression),int(num_favorite),int(ranking_scores),int(user_id),
                                time_of_the_day])
        if i>0 :
            next_hour_amount[i-1] = int(total_tweets)
        
    y = next_hour_amount
    X = feature[:y.shape[0]]
    
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    RMSE = np.sqrt(mean_squared_error(y, pred))
    R2_score =  r2_score(y, pred)
    results = sm.OLS(y, X).fit()
    print('RMSE = %.4f' % RMSE)
    print('R2_score = %.4f' % R2_score)
    print(results.summary())
    print()
    

    #sort top 3 features
    best_features = results.pvalues.argsort()[:3]        
    print ("Best features selected are:")
    
    #plot of top 3 features
    for index in best_features:
        print(feature_names[index])
        plt.title("Scatter plot for number of tweets for next hour versus "+feature_names[index])
        plt.ylabel("Number of tweets for nexthour")
        plt.xlabel(feature_names[index])
        plt.scatter(X[:, index],pred)
        plt.show()






##################### Problem 1.4 #############################################


import json
import datetime
import pytz
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

#hashtags = ['sb49'] 
file_names = []
tweet_data = {}

# Load data

hashtag_dict = {'gohawks': 188136,
                       'gopatriots': 26232,
                       'nfl': 259024,
                       'patriots': 489713,
                       'sb49': 826951,
                       'superbowl': 1348767}



def build_matrix(raw, index='date'):

    raw = raw.set_index(index)
    time_series = raw.groupby(pd.Grouper(freq='60Min'))
    
    X = np.zeros((len(time_series), 10))
    y = np.zeros((len(time_series), 1))
    for i, (time_interval, g) in enumerate(time_series):


        X[i, 0] = g.Number_of_Tweets.sum()
        X[i, 1] = g.Number_of_Retweets.sum()
        X[i, 2] = g.Number_of_Followers.sum()
        X[i, 3] = g.Max_Number_of_Followers.max()
        X[i, 4] = time_interval.hour     #store the hour of the day -> preserve order
        X[i, 5] = g.Total_Number_of_Replies.sum()
        X[i, 6] = g.Count_of_Impressions.sum()
        X[i, 7] = g.Favourite_Count.sum()
        X[i, 8] = g.Ranking_Score.sum()
        X[i, 9] = g.user_id.sum()

        y[i, 0] = g.Number_of_Tweets.sum()
        
    return np.nan_to_num(X[:-1]), y[1:]

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


pst_tz = pytz.timezone('US/Pacific')

first_date_marker = datetime.datetime(2015, 2, 1, 8, 0, 0, 0)

second_date_marker = datetime.datetime(2015, 2, 1, 20, 0, 0, 0)


def filter_and_test(df, regressor):

    df_first = df[df.date < first_date_marker]

    #Between Feb. 1, 8:00 a.m. and 8:00 p.m. 
    df_second = df[(df.date > first_date_marker) &
               (df.date < second_date_marker)]

    #After Feb. 1, 8:00 p.m.
    df_third = df[df.date > second_date_marker]

    print("Before Feb. 1, 8:00 a.m.")
    X_df_first, y_df_first = build_matrix(df_first, index='date')
    y_pred = cross_val_predict(regressor, X_df_first, y_df_first.ravel(), cv=10)
    total_error = 0.0
    for (actual, predicted) in zip(y_df_first, y_pred):
        total_error += abs(actual - predicted)
    print("Averaged error is: ", total_error/len(y_df_first))
    print()
    print("Between Feb. 1, 8:00 a.m. and 8:00 p.m.")
    X_df_second, y_df_second = build_matrix(df_second, index='date')
    y_pred = cross_val_predict(regressor, X_df_second, y_df_second.ravel(), cv=10)
    total_error = 0.0
    for (actual, predicted) in zip(y_df_second, y_pred):
        total_error += abs(actual - predicted)
    print("Averaged error is: ", total_error/len(y_df_second))
    print()
    print("After Feb. 1, 8:00 p.m.")
    X_df_third, y_df_third = build_matrix(df_third, index='date')
    y_pred = cross_val_predict(regressor, X_df_third, y_df_third.ravel(), cv=10)
    total_error = 0.0
    for (actual, predicted) in zip(y_df_third, y_pred):
        total_error += abs(actual - predicted)
    print("Averaged error is: ", total_error/len(y_df_third))
    print()

df_aggregated = None

#total_tweets = len(tweet_data[hashtag])
for hashtag, num_tweets in hashtag_dict.items():
    print("Hashtag is: ", hashtag)
    print()
    
    total_retweets = 0
    total_followers = 0
    maximum_followers = 0
    total_replies = 0
    count_impression = 0
    ranking_scores = 0
    num_favorite = 0
    user_id=0
    
    file_name = 'tweet_data/tweets_#' + hashtag + '.txt'
    file_names.append(file_name)
    with open(file_name, encoding='utf-8') as f:
        print('Now loading',hashtag)
        print()
        df = pd.DataFrame({'date': "", 'Number_of_Tweets': "", 'Number_of_Retweets': "",'Number_of_Followers': "", 
            'Max_Number_of_Followers': "", 'Total_Number_of_Replies': "", 'Count_of_Impressions': "", 
            'Favourite_Count': "", 'Ranking_Score': "", 'user_id': ""}, index=range(num_tweets))
        for i, total_tweet in enumerate(f):
            tweet = json.loads(total_tweet)
            total_tweets = 1
            date = datetime.datetime.fromtimestamp(tweet['firstpost_date'])
            total_followers += tweet['author']['followers']
            total_retweets += tweet['metrics']['citations']['total']
            total_replies += tweet['metrics']['citations']['replies']
            count_impression +=tweet['metrics']['impressions']
            num_favorite +=tweet['tweet']['favorite_count']
            ranking_scores +=tweet['metrics']['ranking_score']
            user_id += tweet['tweet']['user']['id']
            if tweet['author']['followers'] > maximum_followers:
                maximum_followers = tweet['author']['followers']

            df.loc[i]= {'date': date, 'Number_of_Tweets': total_tweets, 'Number_of_Retweets': total_retweets,'Number_of_Followers': total_followers, 
                'Max_Number_of_Followers': maximum_followers, 'Total_Number_of_Replies': total_replies, 'Count_of_Impressions': count_impression, 
                'Favourite_Count': num_favorite, 'Ranking_Score': ranking_scores, 'user_id': user_id}
    #linear = LinearRegression(normalize=False)
    #neighb = KNeighborsRegressor()
        rfr =  RandomForestRegressor(random_state = 42)
    #filter_and_test(df, linear)
    #filter_and_test(df, neighb)
        filter_and_test(df, rfr)
        if df_aggregated is None: #first iteration 
            df_aggregated = df
        else: #aggregate
            df_aggregated = pd.concat([df_aggregated, df])
filter_and_test(df_aggregated, rfr)


##################### Problem 1.5 #############################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 00:34:38 2018

@author: oliviajin
"""

import json
import numpy as np
import datetime
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor

# Import Train Data
hashtags = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']
# hashtags = ['gopatriots', 'gopatriots']

def load_train_data(hashtag):
    file_name = 'tweet_data/tweets_#' + hashtag + '.txt'
    with open(file_name, encoding='utf-8') as f:
        print('Now loading', hashtag)
        file_data = []
        for line in f:
            file_data.append(json.loads(line))

        return file_data


# Import Test Data

testdatas = ['sample1_period1', 'sample2_period2', 'sample3_period3',
             'sample4_period1', 'sample5_period1', 'sample6_period2',
             'sample7_period3', 'sample9_period2', 'sample10_period3']

testdatas2 = ['sample8_period1']

# testdatas = ['sample1_period1', 'sample7_period3']

# Load data
def load_test_data(testdata):
    file_name = 'test_data/' + testdata + '.txt'
    with open(file_name, encoding='utf-8') as f:
        print('Now loading', testdata)
        file_data = []
        for line in f:
            file_data.append(json.loads(line))
        return file_data
        

def define_dic(tweets):
    tweet_dic = defaultdict(list)
    i = 0
    start_time = tweets[0]['firstpost_date']
    end_time = start_time + 3600
    for tweet in tweets:
        tweet_time = tweet['firstpost_date']
        if tweet_time > end_time:
            gap = (tweet_time - end_time) // 3600 + 1
            if gap > 1:
                for j in range(i + 1, i + gap):
                    tweet_dic[j] = []
                i += gap
                end_time += gap * 3600
            else:
                i += 1
                end_time += 3600
        tweet_dic[i].append(tweet)

    return tweet_dic


def data_split(data):
    first_date_marker = datetime.datetime(2015, 2, 1, 8, 0, 0, 0)
    second_date_marker = datetime.datetime(2015, 2, 1, 20, 0, 0, 0)

    for k, tweet in enumerate(data):
        tweet_time = datetime.datetime.fromtimestamp(tweet['firstpost_date'])
        if tweet_time >= first_date_marker:
            pos1 = k
            break
    for k, tweet in enumerate(data):
        tweet_time = datetime.datetime.fromtimestamp(tweet['firstpost_date'])
        if tweet_time >= second_date_marker:
            pos2 = k
            break
    data1 = data[:pos1]
    data2 = data[pos1:pos2]
    data3 = data[pos2:]
    return data1, data2, data3


def extract_feature(data, window_size):
    dic = define_dic(data)
    size = len(dic) - window_size
    if size <= 0:
        print('Size Error!')
        return 0
    feature = np.zeros((size, 5 * window_size))
    next_hour = np.zeros(size)
    for i, key in enumerate(dic):
        tweets = dic[key]
        if i > len(dic) - 2:
            next_hour[i - window_size] = len(tweets)
            break
        if i < window_size:
            index = i % window_size
            try:
                hour = tweets[0]['firstpost_date']
                time_of_the_day = int(datetime.datetime.fromtimestamp(hour).strftime('%H'))
            except:
                time_of_the_day = 0
            total_tweets = len(tweets)
            total_retweets = 0
            total_followers = 0
            maximum_followers = 0
            for tweet in tweets:
                try:
                    total_followers += tweet['author']['followers']
                    total_retweets += tweet['metrics']['citations']['total']
                    if tweet['author']['followers'] > maximum_followers:
                        maximum_followers = tweet['author']['followers']
                except:
                    pass
            feature[0, window_size * index:(window_size * index + 5)] = np.array([int(total_tweets),
                                                                                  int(total_retweets),
                                                                                  int(total_followers),
                                                                                  int(maximum_followers),
                                                                                  time_of_the_day])
        else:
            next_hour[i - window_size] = len(tweets)
            n = i - window_size + 1
            feature[n][:5 * (window_size - 1)] = feature[n - 1][5:]
            try:
                hour = tweets[0]['firstpost_date']
                time_of_the_day = int(datetime.datetime.fromtimestamp(hour).strftime('%H'))
            except:
                time_of_the_day = 0
            total_tweets = len(tweets)
            total_retweets = 0
            total_followers = 0
            maximum_followers = 0
            for tweet in tweets:
                try:
                    total_followers += tweet['author']['followers']
                    total_retweets += tweet['metrics']['citations']['total']
                    if tweet['author']['followers'] > maximum_followers:
                        maximum_followers = tweet['author']['followers']
                except:
                    pass
            feature[n][5 * (window_size - 1):] = np.array([int(total_tweets),
                                                           int(total_retweets),
                                                           int(total_followers),
                                                           int(maximum_followers),
                                                           time_of_the_day])
    return feature, next_hour


def merge_feature(window_size):
    for i, hashtag in enumerate(hashtags):
        train_data = load_train_data(hashtag)
        data1, data2, data3 = data_split(train_data)

        if i == 0:
            # Extract Feaure
            X_train_1, y_train_1 = extract_feature(data1, window_size)
            X_train_2, y_train_2 = extract_feature(data2, window_size)
            X_train_3, y_train_3 = extract_feature(data3, window_size)
        else:
            X1, y1 = extract_feature(data1, window_size)
            X2, y2 = extract_feature(data2, window_size)
            X3, y3 = extract_feature(data3, window_size)
            X_train_1 = np.vstack((X_train_1, X1))
            X_train_2 = np.vstack((X_train_2, X2))
            X_train_3 = np.vstack((X_train_3, X3))
            y_train_1 = np.hstack((y_train_1, y1))
            y_train_2 = np.hstack((y_train_2, y2))
            y_train_3 = np.hstack((y_train_3, y3))
    print()
    
    return X_train_1, y_train_1, X_train_2, y_train_2, X_train_3, y_train_3


if __name__ == '__main__':
    # 5 Hour Window
    window_size = 5
    X1, y1, X2, y2, X3, y3 = merge_feature(window_size)
    
    # Train
    clf = RandomForestRegressor(random_state=42)
    model1 = clf.fit(X1, y1)
    model2 = clf.fit(X2, y2)
    model3 = clf.fit(X3, y3)

    # Test
    for sample in testdatas:
        print('Predict for', sample)
        test_data = load_test_data(sample)
        if 'period1' in sample:
            Xt, yt = extract_feature(test_data, window_size)
            y_pred = model1.predict(Xt)
            MAE = mean_absolute_error(yt, y_pred)
        elif 'period2' in sample:
            Xt, yt = extract_feature(test_data, window_size)
            y_pred = model2.predict(Xt)
            MAE = mean_absolute_error(yt, y_pred)
        elif 'period3' in sample:
            Xt, yt = extract_feature(test_data, window_size)
            y_pred = model3.predict(Xt)
            MAE = mean_absolute_error(yt, y_pred)
        print('Actual Tweet Num:', yt)
        print('Predict Tweet Num', y_pred)
        print('MAE', MAE)
        print()

    # 4 Hour Window
    window_size = 4
    X1, y1, X2, y2, X3, y3 = merge_feature(window_size)
    
    # Train
    clf = RandomForestRegressor(random_state=42)
    model1 = clf.fit(X1, y1)
    model2 = clf.fit(X2, y2)
    model3 = clf.fit(X3, y3)

    # Test
    for sample in testdatas2:
        print('Predict for', sample)
        test_data = load_test_data(sample)
        if 'period1' in sample:
            Xt, yt = extract_feature(test_data, window_size)
            y_pred = model1.predict(Xt)
            MAE = mean_absolute_error(yt, y_pred)
        elif 'period2' in sample:
            Xt, yt = extract_feature(test_data, window_size)
            y_pred = model2.predict(Xt)
            MAE = mean_absolute_error(yt, y_pred)
        elif 'period3' in sample:
            Xt, yt = extract_feature(test_data, window_size)
            y_pred = model3.predict(Xt)
            MAE = mean_absolute_error(yt, y_pred)
        print('Actual Tweet Num:', yt)
        print('Predict Tweet Num', y_pred)
        print('MAE', MAE)
        print()

########### Problem 2 #########################################################
WAs = ['Seattle, Washington', 'Washington', 'WA', 'Seattle, WA', 
       'Kirkland, Washington','Kirkland', 'WASHINGTON']
MAs = ['Boston', 'Massachusetts', 'MA', 'Cambridge', 'massachusetts', 'boston',
       'MASSACHUSETTS']

tweet_text = []
y = []
for hashtag in hashtags:
    print('Fan Base Prediction for', hashtag)
    data = tweet_data[hashtag]
    location = []
    for tweet in data:
        loc = tweet['tweet']['user']['location']
        location.append(tweet['tweet']['user']['location'])
        
        # Find tweet users in WA
        for WA in WAs:
            if WA in loc:
                if not ('D.C.' in loc or 'DC' in loc or 'D.C' in loc):
                    tweet_text.append(tweet['tweet']['text'])
                    y.append(0)
        
        # Find tweet users in MA
        for MA in MAs:
            if MA in loc:
                tweet_text.append(tweet['tweet']['text'])
                y.append(1)
                
    data_len = len(tweet_text)
    position = int(0.8 * data_len)
    X_train = tweet_text[:position]
    X_test = tweet_text[position:]
    y_train = np.array(y[:position])
    y_test = np.array(y[position:])
                
    logsiticRegression_LSI_2 = Pipeline([
        ('vect', CountVectorizer(min_df=2, stop_words=stop_words)),
        ('tfidf', TfidfTransformer()),
        ('svd', TruncatedSVD(n_components=5, n_iter=10, random_state=42)),
        ('clf', LogisticRegression()),
    ])
    
    fit_predict_and_plot_roc(logsiticRegression_LSI_2, X_train, y_train, X_test, y_test)
    label_predict = logsiticRegression_LSI_2.predict(X_test)
    plt.title('#' + hashtag)
    output(y_test, label_predict)
    print()

        
########## Part 3 Design Your Own Project #####################################

# Import libraries
import json
import re
from textblob import TextBlob
from collections import OrderedDict
import numpy as np
import datetime
import pytz
import matplotlib.pyplot as plt

######################## Load Data for #gohawks Subjectivity #########################
hashtags = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']

data = []
with open('tweet_data/tweets_#gohawks.txt', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# This function removes any links and special characters in twitter message
def remove_links_chars(text):
    return ' '.join((re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(@[A-Za-z0-9]+)", " ", text).split()))

# Sentiment Analysis 1: Objective vs. Subjective
retweet_time = []
subjectivity_scores = []

for tweet in data:
    retweet_time.append(tweet['citation_date'])
    text = TextBlob(remove_links_chars(tweet['title']))
    subjectivity_scores.append(text.sentiment.subjectivity)

start_time = min(retweet_time)
end_time = max(retweet_time)

time_sub_scores_dict = dict(zip(retweet_time, subjectivity_scores))
sorted_time_sub_scores_dict = OrderedDict(sorted(time_sub_scores_dict.items()))

first_time = list(sorted_time_sub_scores_dict.keys())[0]
subjective_scores_30mins = []
temp_scores = []

pst_tz = pytz.timezone('US/Pacific')
pst_time_list = []

for key, value in sorted_time_sub_scores_dict.items():
    if (key <= first_time + 1800):
        temp_scores.append(value)
    else:
        subjective_scores_30mins.append(np.mean(temp_scores))
        pst_time_list.append(datetime.datetime.fromtimestamp(first_time, pst_tz))
        temp_scores = []
        first_time = first_time + 1800
        
plt.bar(pst_time_list, subjective_scores_30mins, width=0.01)
plt.title('Sentiment Analysis of Subjectivity for Seahawk Fans', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Subjective Scores', fontsize=16)
plt.ylim(0,1)
plt.show()

# Enlarge the Dataset during the game
plt.bar(pst_time_list[880:910], subjective_scores_30mins[880:910], width=0.01)
plt.title('Sentiment Analysis of Subjectivity for Seahawk Fans during the Game', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Subjective Scores', fontsize=16)
plt.ylim(0,1)
plt.show()



####################### Load Data for #patriots  Subjectivity#########################
hashtags = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']

data = []
with open('tweet_data/tweets_#patriots.txt', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# This function removes any links and special characters in twitter message
def remove_links_chars(text):
    return ' '.join((re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(@[A-Za-z0-9]+)", " ", text).split()))

# Sentiment Analysis 1: Objective vs. Subjective
retweet_time = []
subjectivity_scores = []

for tweet in data:
    retweet_time.append(tweet['citation_date'])
    text = TextBlob(remove_links_chars(tweet['title']))
    subjectivity_scores.append(text.sentiment.subjectivity)

start_time = min(retweet_time)
end_time = max(retweet_time)

time_sub_scores_dict = dict(zip(retweet_time, subjectivity_scores))
sorted_time_sub_scores_dict = OrderedDict(sorted(time_sub_scores_dict.items()))

first_time = list(sorted_time_sub_scores_dict.keys())[0]
subjective_scores_30mins = []
temp_scores = []

pst_tz = pytz.timezone('US/Pacific')
pst_time_list = []

for key, value in sorted_time_sub_scores_dict.items():
    if (key <= first_time + 1800):
        temp_scores.append(value)
    else:
        subjective_scores_30mins.append(np.mean(temp_scores))
        pst_time_list.append(datetime.datetime.fromtimestamp(first_time, pst_tz))
        temp_scores = []
        first_time = first_time + 1800
        
plt.bar(pst_time_list, subjective_scores_30mins, width=0.01)
plt.title('Sentiment Analysis of Subjectivity for Patriots Fans', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Subjective Scores', fontsize=16)
plt.ylim(0,1)
plt.show()

# Enlarge the Dataset during the game
plt.bar(pst_time_list[880:910], subjective_scores_30mins[880:910], width=0.01)
plt.title('Sentiment Analysis of Subjectivity for Patriots Fans during the Game', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Subjective Scores', fontsize=16)
plt.ylim(0,1)
plt.show()



############# Load Data for #gohawks Attitude Analysis #########################
hashtags = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']

data = []
with open('tweet_data/tweets_#gohawks.txt', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# This function removes any links and special characters in twitter message
def remove_links_chars(text):
    return ' '.join((re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(@[A-Za-z0-9]+)", " ", text).split()))

# Sentiment Analysis 1: Objective vs. Subjective
retweet_time = []
polarity_scores = []

for tweet in data:
    retweet_time.append(tweet['citation_date'])
    text = TextBlob(remove_links_chars(tweet['title']))
    polarity_scores.append(text.sentiment.polarity)

start_time = min(retweet_time)
end_time = max(retweet_time)

time_polarity_scores_dict = dict(zip(retweet_time, polarity_scores))
sorted_time_polarity_scores_dict = OrderedDict(sorted(time_polarity_scores_dict.items()))

first_time = list(sorted_time_polarity_scores_dict.keys())[0]
polarity_scores_30mins = []
temp_scores = []

pst_tz = pytz.timezone('US/Pacific')
pst_time_list = []

for key, value in sorted_time_polarity_scores_dict.items():
    if (key <= first_time + 1800):
        temp_scores.append(value)
    else:
        polarity_scores_30mins.append(np.mean(temp_scores))
        pst_time_list.append(datetime.datetime.fromtimestamp(first_time, pst_tz))
        temp_scores = []
        first_time = first_time + 1800
        

# Enlarge the Dataset during the game
plt.bar(pst_time_list[880:910], polarity_scores_30mins[880:910], width=0.01)
plt.title('Sentiment Analysis of Polarity for Seahawk Fans during the Game', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Polarity Scores', fontsize=16)
plt.ylim(0,0.3)
plt.show()


############# Load Data for #patriots Attitude Analysis #########################
hashtags = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']

data = []
with open('tweet_data/tweets_#patriots.txt', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# This function removes any links and special characters in twitter message
def remove_links_chars(text):
    return ' '.join((re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(@[A-Za-z0-9]+)", " ", text).split()))

# Sentiment Analysis 1: Objective vs. Subjective
retweet_time = []
polarity_scores = []

for tweet in data:
    retweet_time.append(tweet['citation_date'])
    text = TextBlob(remove_links_chars(tweet['title']))
    polarity_scores.append(text.sentiment.polarity)

start_time = min(retweet_time)
end_time = max(retweet_time)

time_polarity_scores_dict = dict(zip(retweet_time, polarity_scores))
sorted_time_polarity_scores_dict = OrderedDict(sorted(time_polarity_scores_dict.items()))

first_time = list(sorted_time_polarity_scores_dict.keys())[0]
polarity_scores_30mins = []
temp_scores = []

pst_tz = pytz.timezone('US/Pacific')
pst_time_list = []

for key, value in sorted_time_polarity_scores_dict.items():
    if (key <= first_time + 1800):
        temp_scores.append(value)
    else:
        polarity_scores_30mins.append(np.mean(temp_scores))
        pst_time_list.append(datetime.datetime.fromtimestamp(first_time, pst_tz))
        temp_scores = []
        first_time = first_time + 1800
        

# Enlarge the Dataset during the game
plt.bar(pst_time_list[880:910], polarity_scores_30mins[880:910], width=0.01)
plt.title('Sentiment Analysis of Polarity for Patriots Fans during the Game', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Polarity Scores', fontsize=16)
plt.ylim(0,0.4)
plt.show()



