import csv
import os
import re
import math
import random
import sys
import pandas as pd


regex_str = [
    r'(?:[.-]*@[\w]*[0-9]*[.:!,;\?\'\`")]*[\w]*)',
    r'http[s]*?[://]*(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),\/]|(?:%[0-9a-f][0-9a-f]))*',
]
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)
token_re2 = r = re.compile(r"([#?])(\w+)")


def tokenize(s):
    return tokens_re.findall(s)


def tokenize2(s2):
    return token_re2.findall(s2)


def removehash(s1):
    w1 = tokenize2(s1)
    slist = s1.split(' ')
    for word in w1:
        sword = word[0] + word[1]
        rword = word[1]
        slist = [w.replace(sword, rword) for w in slist]
    return ' '.join(slist)


# function to preprocess the tweets
def preprocess(text):
    text = removehash(text)
    newtext = list(set(text.split(' ')) - set(tokenize(text)))
    newtext = [nt.lower() for nt in newtext]
    return newtext


# function to calculate the jaccarddistance between two sets
def jaccarddistance(a, b):
    inter = list(set(a) & set(b))
    I = len(inter)
    union = list(set(a) | set(b))
    U = len(union)
    return round(1 - (float(I) / U), 4)


# function to calculate the sum of squared error
def calculate_SSE(clusters, tweets_dict, k):
    SSE = 0
    print("For k =", k)
    i = 1
    for id, cluster in clusters.items():
        for tweets in cluster:
            SSE_dist = jaccarddistance(tweets_dict[tweets], tweets_dict[str(id)])
            SSE = SSE + (SSE_dist * SSE_dist)
        print(i, ":", len(cluster), " tweets")
        i = i + 1
    print("\nSSE = ", SSE)


# function to assign each tweet to a cluster and update the centroids
def k_means(ids, tweets_dict, centroids, k):
    try:
        for i in range(k):
            clusters = {}
            for centroid in centroids:
                clusters[int(centroid)] = []
            # assign each tweet to the nearest cluster
            for item in tweets_dict:
                min_dist = sys.maxsize
                min_seed = ''
                for centroid in centroids:
                    dist = jaccarddistance(tweets_dict[centroid], tweets_dict[item])
                    if (dist < min_dist):
                        min_dist = dist
                        min_seed = centroid
                clusters[int(min_seed)].append(item)
            # update the centroids of the clusters by taking mean
            centroids = []
            for id, cluster in clusters.items():
                new_centroid_dist = 255
                new_centriod = ''
                for tweet in cluster:
                    distance = 0
                    for each_tweet in cluster:
                        distance = distance + jaccarddistance(tweets_dict.get(tweet), tweets_dict.get(each_tweet))
                    mean = distance / len(cluster)
                    if (mean < new_centroid_dist):
                        new_centroid_dist = dist
                        new_centroid = tweet
                centroids.append(new_centroid)
        calculate_SSE(clusters, tweets_dict, k)

    except(ValueError):
        print("error")


input_file =pd.read_csv("Data_after_filtering/clear_NY_1.csv")
all_terms = []
ids = []

for row in input_file['tweet']:
    tokens = preprocess(row[2])
    all_terms.append(tokens)
    ids.append(row[0])

tweets_dict = dict(zip(ids, all_terms))
l = len(all_terms)
k = input("Enter the value for k:")
# assign centroids for clusters using random function
seeds = random.sample(ids, int(k))
k_means(ids, tweets_dict, seeds, int(k))
