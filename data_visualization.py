import nltk
import pandas as pd
import webbrowser
import unicodedata
import scattertext as st
import spacy
import re, io
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import IFrame
from IPython.core.display import display, HTML
from pandas._libs.tslibs.timestamps import Timestamp
from wordcloud import WordCloud
import os, pkgutil, json, urllib
from urllib.request import urlopen
from pprint import pprint
from scattertext import CorpusFromPandas, produce_scattertext_explorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score
from sklearn.metrics import silhouette_score
import xlrd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

display(HTML("&lt;style>.container { width:98% !important; }&lt;/style>"))

class Data_visualization:

    def __init__(self,file_name):
        self.str=""
        self.word_array=[]
        self.file_name=file_name

    def wordcloud(self):
        stop_words = nltk.corpus.stopwords.words("english")
        Word__tokenize = nltk.word_tokenize(self.str)
        # filter word in stop word
        Word__tokenize_filter = [w for w in Word__tokenize if not w in stop_words]
        width = 12
        height = 12
        plt.figure(figsize=(width, height))
        # text = 'all your base are belong to us all of your base base base'
        wordcloud = WordCloud(width=1800, height=1400).generate(str(Word__tokenize_filter))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig("WORD_CLOUD_pos_After_Classification_NY_6.png")
        plt.close()

    def word_count(self):
        stop_words = nltk.corpus.stopwords.words("english")

        Word__tokenize = nltk.word_tokenize(self.str)

        # filter word in stop word
        Word__tokenize_filter = [w for w in Word__tokenize if not w in stop_words]
        # count the word
        Word__tokenize_counter = Counter(Word__tokenize_filter)
        word_dict = dict(Word__tokenize_counter)

        data = pd.DataFrame(list(word_dict.items()), columns=['word', 'count'])

        data.sort_values("count", axis=0, ascending=False, inplace=True, na_position='last')

       # data.to_csv("word_count"+self.file_name)

        fig, ax = plt.subplots(figsize=(30, 30))
        # Plot horizontal bar graph
        data[:50].plot.barh(x='word', y='count', ax=ax, color="purple")
        ax.set_title("Common Words Found in Tweets (Including All Words)")

        plt.savefig("WORD_COUNT_pos_After_Classification_NY_6.png")
        plt.close()

    def scattertext_function(self):

        ## START
        nlp = spacy.load('en_core_web_sm')
        convention_df = pd.read_csv("After_Classification/After_Classification_NY_6.csv")
        convention_df['parsed'] = convention_df.tweet.apply(nlp)

        ##Index(['Unnamed: 0', 'Date', 'name', 'tweet', 'death', 'Classification'], dtype='object')
        # print("Document Count")
        # print(convention_df.groupby('Classification')['tweet'].count())
        # print("Word Count")
        # print(convention_df.groupby('Classification').apply(lambda x: x.tweet.apply(lambda x: len(x.split())).sum()))
        # print(type(convention_df))

        ##Convert Dataframe into Scattertext Corpus
        corpus = st.CorpusFromParsedDocuments(convention_df, category_col='Classification', parsed_col='parsed').build()
        print(type(st.Scalers.log_scale_standardize))
        list(corpus.get_scaled_f_scores_vs_background().index[:10])
        html = st.produce_scattertext_explorer(corpus,category='pos', category_name='POS', not_category_name='NEG',minimum_term_frequency=5, width_in_pixels=1000,transform=st.Scalers.log_scale_standardize)


        file_name_1 = 'After_Classification_NY_6.html'
        open(file_name_1, 'wb').write(html.encode('utf-8'))
        print(IFrame(src=file_name_1, width=1200, height=700))
        #display(IFrame(html))

    def open_html(self):
        new = 2
        url = "Scattertext_HTML/After_Classification_NY_1.html"
        webbrowser.open(url, new=new)

    def test2(self):

        df=pd.read_csv(self.file_name)
        pos_df=df.loc[df["Classification"] == "pos"]
        neg_df=df.size-pos_df.size
        Tasks = [pos_df.size, neg_df]
        my_labels = 'POS', 'NEG'
        plt.pie(Tasks, labels=my_labels, autopct='%1.1f%%')
        plt.title('Trump')
        plt.axis('equal')
        plt.savefig("PI_Trump_D12.png")

        #plt.show()




    def start(self):
        self.test2()
        exit()


        #self.scattertext_function()

        df = pd.read_csv(self.file_name)
        neg_trump_D3 = df.loc[df["Classification"] == "pos"]
        text = neg_trump_D3['tweet'].tolist()

        for words in text:
            self.word_array.append(str(words))
            self.str = self.str + words

        self.word_array = list(set(self.word_array))
        self.wordcloud()
        self.word_count()







name="After_Classification/After_Classification_Trump_D12.csv"
data=Data_visualization(name)
data.start()

##corpus = st.CorpusFromParsedDocuments(convention_df, category_col='party', parsed_col='parsed').build()
# df=pd.read_csv("After_Classification/After_Classification_Trump_D1.csv")
# print(df["Date"].head())