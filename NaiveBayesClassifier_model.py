import re

import nltk
import pandas as pd
import unicodedata
from nltk import LancasterStemmer

import pickle
from nltk.corpus import twitter_samples
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from random import shuffle

class NaiveBayesClassifier_model:

    def __init__(self,file_name):

        self.file_name=file_name
        self.classify=[]
        self.date_list=[]
        self.name_list=[]
        self.text_list=[]
        self.num_of_death=[]
        self.After_analyzing_data_dict={}

    def train_model(self):

        #print(twitter_samples.fileids())

        pos_tweets = twitter_samples.strings('positive_tweets.json')
        #print(len(pos_tweets))  # Output: 5000

        neg_tweets = twitter_samples.strings('negative_tweets.json')
        #print(len(neg_tweets))  # Output: 5000

        # all_tweets = twitter_samples.strings('tweets.20150430-223406.json')
        # print (len(all_tweets)) # Output: 20000

        # positive tweets words list
        pos_tweets_set = []
        for tweet in pos_tweets:
            pos_tweets_set.append((tweet, 'pos'))

        # negative tweets words list
        neg_tweets_set = []
        for tweet in neg_tweets:
            neg_tweets_set.append((tweet, 'neg'))

      #  print(len(pos_tweets_set), len(neg_tweets_set))  # Output: (5000, 5000)

        pos_tweets = twitter_samples.strings('positive_tweets.json')
      #  print(len(pos_tweets))  # Output: 5000

        neg_tweets = twitter_samples.strings('negative_tweets.json')
       # print(len(neg_tweets))  # Output: 5000

        # all_tweets = twitter_samples.strings('tweets.20150430-223406.json')
        # print (len(all_tweets)) # Output: 20000

        # positive tweets words list
        pos_tweets_set = []
        for tweet in pos_tweets:
            pos_tweets_set.append((tweet, 'pos'))

        # negative tweets words list
        neg_tweets_set = []
        for tweet in neg_tweets:
            neg_tweets_set.append((tweet, 'neg'))

       # print(len(pos_tweets_set), len(neg_tweets_set))  # Output: (5000, 5000)

        # radomize pos_reviews_set and neg_reviews_set
        # doing so will output different accuracy result everytime we run the program

        shuffle(pos_tweets_set)
        shuffle(neg_tweets_set)

        test_set = pos_tweets_set[:1000] + neg_tweets_set[:1000]
        train_set = pos_tweets_set[1000:2000] + neg_tweets_set[1000:2000]

       # print(len(test_set), len(train_set))  # Output: (200, 400)

        # train classifier

        classifier = NaiveBayesClassifier(train_set)

        # calculate accuracy
        accuracy = classifier.accuracy(test_set)

        print("Accuracy")
        print(accuracy)  # Output: 0.715

        # show most frequently occurring words
        print(classifier.show_informative_features(10))

        # saving classfier
        #############################################################################
        save_classifier = open("naivebayes.pickle", "wb")
        pickle.dump(classifier, save_classifier)
        save_classifier.close()
        ############################################################################

    def model_classifier(self):

        df=pd.read_csv(self.file_name)
        #print(df.head())


        ##open classifier
        classifier_f = open("naivebayes.pickle", "rb")
        classifier = pickle.load(classifier_f)
        classifier_f.close()

        for row in df[1:5].iterrows():
            #print(row[1][1])
            self.date_list.append(row[1][1])
            #print(row[1][2])
            self.name_list.append(row[1][2])
            #print(row[1][3])
            self.text_list.append(row[1][3])
            #print(row[1][4])
            self.num_of_death.append(row[1][4])

            text=row[1][3]
            #blob = TextBlob(text, classifier=classifier.prob_classify(text))
            blob = TextBlob(text, classifier=classifier)
            print(blob.polarity)
            #print(blob.classify())

            self.classify.append(blob.classify())



            ##After analyzing the text
        self.After_analyzing_data_dict.update(
            {"Date": self.date_list, "name": self.name_list, "tweet": self.text_list, "death": self.num_of_death, "Classification":self.classify})

        new_data = pd.DataFrame(self.After_analyzing_data_dict)
        print(new_data.head())
        new_data.to_csv("A_Trump_D12.csv")


data=NaiveBayesClassifier_model("Data_after_filtering/Trump_D1f2.csv")
# data.model_classifier()
data.train_model()

# df=pd.read_csv("After_Classification_NY_6.csv")
# print(df.head())