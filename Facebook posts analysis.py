import glob
import os
import pandas as pd
import textblob
from textblob import TextBlob
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

from nltk.stem import WordNetLemmatizer

from nltk import word_tokenize          
nltk.download('punkt')
nltk.download('wordnet')

#Getting the data 


filenames=glob.glob(os.path.join('', "/Users/ashikshafi/Downloads/Facebook dataset R/*.csv"))

filenames=glob.glob(os.path.join('', "/Users/ashikshafi/Downloads/Facebook dataset R/*"))



keylist=["ABC", "Foxnews", "CBS", "WSJ", "HuffPost", "USAToday", "BBC", "NPR", "CNN", "NYtimes", "LATimes", "NBC", "TimeMagazine"]

df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "/Users/ashikshafi/Downloads/Facebook dataset R/*"))), keys=keylist)

df.to_csv('CombinedFBpost.csv')



Shortdf=df.sample(n=5000)

Shortdf.loc[["CNN"]]



#Preprocessing
Shortdf["message"]=Shortdf["message"].str.replace('http.*', "", regex=True)

def Preprocessed(text):
    text= re.sub(r'[^A-Za-z]+', " ",str(text))
    text=text.lower()
    return text

Shortdf["message"]=Shortdf["message"].apply(lambda x: Preprocessed(x))

Shortdf["message"] = Shortdf["message"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop) and len(word)>3]))



#NMF
no_features = 5000
no_topics = 15
no_top_words = 15

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

def display_topics(model, feature_names, no_top_words):
    for x, y in enumerate(model.components_):
        print ("Topic %d:" % (x))
        print (" ".join([feature_names[i] for i in y.argsort()[:-no_top_words - 1:-1]]))


tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, max_features=no_features, tokenizer=LemmaTokenizer())
tfidf = tfidf_vectorizer.fit_transform(Shortdf["message"].astype(str))
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


nmf = NMF(n_components=15, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
display_topics(nmf, tfidf_feature_names, no_top_words)

NMFdocXtopics = nmf.transform(tfidf)

#NMF column name

DocandTopics=pd.DataFrame(NMFdocXtopics).reset_index(drop=True)
NewShortdf=pd.DataFrame(Shortdf).reset_index(level=0, inplace=True)

#Now, Adding likes and shares to the dataframe

DocandTopics["Likes"]=Shortdf['likes_count'].reset_index(drop=True)

DocandTopics["Comments"]=Shortdf['comments_count'].reset_index(drop=True)
DocandTopics["Shares"]=Shortdf['shares_count'].reset_index(drop=True)
DocandTopics["Media Type"]=Shortdf['level_0'].reset_index(drop=True)

         
#Polarity code worked
    
def Polarity2(text):
    try:
        return TextBlob(text).sentiment[0]
    except:
        return None

DocandTopics["PolarityScore"]=Shortdf["message"].apply(lambda x: Polarity2(x)).reset_index(drop=True)

NewDocMix=DocandTopics

#Updating columnnames


Colnames= ["Rescue/Positive events", "Republican presidential race", "Sharing Story", "Crime/Negative events", "Hope/Positive events", "Sharing News", "Democratic Presidential race", "Petty crime", "Bernie Sanders", "NA1", "Family/Women involved crime", "NA2", "Finance/Living", "Health/Food", "Polarity score", "Likes", "Comments", "Shares", "Media Type", "Polarity Scores"]

Colnames= ["Rescue/Positive events", "Republican presidential race", "Sharing Story", "Crime/Negative events", "Hope/Positive events", "Sharing News", "Democratic Presidential race", "Petty crime", "Bernie Sanders", "NA1", "Family/Women involved crime", "NA2", "Finance/Living", "Health/Food", "Polarity score", "Likes", "Comments", "Shares", "Media Type"]


NewDocMix.columns= Colnames

#dealing with missing value


NewDocMix=NewDocMix.replace(0, np.NaN)
#Very important descriptives

Countvalues=NewDocMix.groupby(NewDocMix['Media Type']).count()
Sumvalues=NewDocMix.groupby(NewDocMix['Media Type']).sum()
Meanvalues=NewDocMix.groupby(NewDocMix['Media Type']).mean()

Countvalues.to_csv('Countvalues.csv')
Sumvalues.to_csv("SumValues.csv")
Meanvalues.to_csv("MeanValues.csv")

#plotting heatmap

Var_corr=NewDocMix.corr()

plt.figure(figsize = (20, 20))

sns.heatmap(Var_corr, xticklabels=Var_corr.columns, yticklabels=Var_corr.columns, annot=True, annot_kws={"size":15})

sns.set(font_scale=1.4)

#Plotting regline
import seaborn as sns

sns.lmplot(x='Republican presidential race', y='Likes', hue="Media Type", markers=['o', 'x', '^', '+', '*', '8'], data=NewDocMix, palette="husl",fit_reg=True)
sns.lmplot(x='Democratic Presidential race', y='Likes', hue="Media Type", markers=['o', 'x', '^', '+', '*', '8'], data=NewDocMix, palette="husl",fit_reg=True)

sns.set(style="ticks", color_codes=True)

sns.pairplot(data=NewDocMix[["Republican presidential race", "Likes", "Shares", "Comments", "Media Type"]], hue="Media Type", palette="husl", markers=['o', 'x', '^', '+', '*', '8'], height=3)
