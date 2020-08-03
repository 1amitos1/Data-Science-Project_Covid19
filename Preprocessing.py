import re

import nltk
import pandas as pd
import unicodedata
from nltk import LancasterStemmer



def clear_column(name):
       df=pd.read_csv(name)
       print(df.columns)
       col_name_to_remove=['id', 'conversation_id', 'created_at',  'time', 'timezone',
              'user_id', 'username',  'place', 'mentions', 'urls',
              'photos', 'replies_count', 'retweets_count', 'likes_count', 'hashtags',
              'cashtags', 'link', 'retweet', 'quote_url', 'video', 'near', 'geo',
              'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to',
              'retweet_date', 'translate', 'trans_src', 'trans_dest']

       final_data = df.drop(col_name_to_remove, axis=1)

       print(final_data.columns)
       print(final_data.shape)
       print(final_data.head())
       final_data.to_csv(name)

class Preprocessing:

       def __init__(self,file_name):

              self.file_name = file_name
              self.word_array=[]
              self.text_list=[]
              self.date_list=[]
              self.name_list=[]
              self.data_dict={}
              self.death_list=[]

       def clear_column(name):
              df = pd.read_csv(name)
              print(df.columns)
              col_name_to_remove = ['id', 'conversation_id', 'created_at', 'time', 'timezone',
                                    'user_id', 'username', 'place', 'mentions', 'urls',
                                    'photos', 'replies_count', 'retweets_count', 'likes_count', 'hashtags',
                                    'cashtags', 'link', 'retweet', 'quote_url', 'video', 'near', 'geo',
                                    'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to',
                                    'retweet_date', 'translate', 'trans_src', 'trans_dest']

              final_data = df.drop(col_name_to_remove, axis=1)

              final_data.to_csv("NY_7_COL.csv")

       def to_lowercase(self,text):
              words = nltk.word_tokenize(str(text))
              """Convert all characters to lowercase from list of tokenized words"""
              new_words = ""
              for word in words:
                     new_word = word.lower()
                     new_words=new_words+" "+new_word

              return new_words

       def remove_non_ascii(self,words):
              """Remove non-ASCII characters from list of tokenized words"""
              new_words = []

              for word in words:
                     new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
                     new_words.append(new_word)

              #return new_words
              return ''.join(new_words)

       def remove_punctuation(self,words):
              """Remove punctuation from list of tokenized words"""
              new_words = ""
              for word in words.split(" "):
                     new_word0=re.sub(r'''(?i)\b((?:https?://|http?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',  " ", word)

                     new_word1 = re.sub(r'[^\w\s]', '', new_word0)
                     new_word2 = re.sub(r'http\S+', '', new_word1)
                     new_word3 = re.sub(r'www\S+', '', new_word2)
                     new_word4 = re.sub(r'html\S+', '', new_word3)



                     if new_word4 != '':
                            new_words=new_words + " "+new_word4


              return new_words

       def stem_words(self,words):
              """Stem words in list of tokenized words"""
              stemmer = LancasterStemmer()
              stems = ""
              for word in words.split(" "):
                     stem = stemmer.stem(word)
                     stems=stems+" "+stem
              return stems

       def normalize(self):

              df = pd.read_csv(self.file_name)
              #col_name_to_remove = ['id', 'conversation_id', 'created_at', 'time', 'timezone',
                                    # 'user_id', 'username', 'place', 'mentions', 'urls',
                                    # 'photos', 'replies_count', 'retweets_count', 'likes_count', 'hashtags',
                                    # 'cashtags', 'link', 'retweet', 'quote_url', 'video', 'near', 'geo',
                                    # 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to',
                                    # 'retweet_date', 'translate', 'trans_src', 'trans_dest']

              for_D_remove=['id', 'conversation_id', 'created_at',  'time', 'timezone',
                            'user_id', 'username', 'place',  'mentions', 'urls',
                            'photos', 'replies_count', 'retweets_count', 'likes_count', 'hashtags',
                            'cashtags', 'link', 'retweet', 'quote_url', 'video', 'near', 'geo',
                            'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to',
                            'retweet_date', 'translate', 'trans_src', 'trans_dest']
              df = df.drop(for_D_remove, axis=1)


              for word in df.iterrows():
                     print(word[1][0],word[1][1],word[1][2])
                     self.date_list.append(word[1][0])
                     self.name_list.append(word[1][1])
                     pre_text=str(word[1][2])
                     words = self.to_lowercase(pre_text)
                     words = self.remove_punctuation(words)
                     words = self.remove_non_ascii(words)
                     self.text_list.append(words)
                     self.death_list.append("0")


              self.data_dict.update({"Date":self.date_list,"name":self.name_list,"tweet":self.text_list,"death":self.death_list})



              new_data=pd.DataFrame(self.data_dict)
              print(new_data["tweet"].head())
              new_data.to_csv("test_D1.csv")







name="orignal_data/Donald_J_Trump.csv"

new_data=Preprocessing(name)

new_data.normalize()
