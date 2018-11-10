import pandas as pd
import numpy as np

import glob
import re
df1= pd.read_csv("/Users/ashikshafi/Downloads/martinchek-2012-2016-facebook-posts/data/cnn_5550296508.csv")
df2= pd.read_csv("/Users/ashikshafi/Downloads/martinchek-2012-2016-facebook-posts/data/fox_news_15704546335.csv")
Data=pd.concat((df1, df2), ignore_index=True, keys= ("CNN", "Fox"))
Data=pd.concat((df1, df2), ignore_index=True, keys= ["CNN", "Fox"], names=["Media"])
Data.columns
Data.shape
Data.head(5)
Data.tail(5)
Data.dtypes

#Select only CNN
Data.loc[["CNN"]]
Data['message'].iloc[400]

re.sub("v.*", "", "i love you")

re.sub("http.*", "", Data['message'])

Data.message=Data["message"].str.replace('http.*', "", regex=True)


Data.message.str.contains("Facebook").sum()


meancnn=
mean(i) for i in (cnndata.love_count), (cnndata.likes_count), (cnndata.comments_count)]:
    list(i.mean())

       cnnmeans=list(cnndata[['likes_count', 'comments_count','shares_count', 'love_count', 'wow_count', 'haha_count', 'sad_count']].mean())
foxmeans = list(foxdata[['likes_count', 'comments_count', 'shares_count', 'love_count', 'wow_count', 'haha_count',
                         'sad_count']].mean())

len(Data['message'].iloc[403])

cnnlen=list(cnndata["message"].str.len())
foxlen=list(cnndata["message"].str.len())

cnndata["message"].str.len()

for i in cnndata["message"].iloc[i]:
    len(i)

import numpy as np
import matplotlib.pyplot as plt

# data to plot


# create plot
fig, ax = plt.subplots()
index = np.arange(7)
bar_width = 0.35
opacity = 0.8+

rects1 = plt.bar(index, cnnmeans, bar_width,
alpha = opacity,
color = 'b',
label = 'CNN')

rects2 = plt.bar(index + bar_width, foxmeans, bar_width,
alpha = opacity,
color = 'g',
label = 'Fox')

plt.xlabel('Category of emotions for each post')
plt.ylabel('Means of emotions')
plt.title('Reactions for posts by CNN and Fox News')
plt.xticks(index + bar_width, ('likes_count', 'comments_count','shares_count', 'love_count', 'wow_count', 'haha_count', 'sad_count'), rotation=45)
plt.legend()

plt.tight_layout()
plt.show()


#second graph
import seaborn as sns
sns.regplot(cnnlen,cnndata.likes_count)
plt.title('Length of posts vs. "Likes" on CNN Facebook posts')
plt.ylabel('Likes')
plt.xlabel('Length of posts')
plt.subplots_adjust(right=0.93, top=0.90, left=0.10, bottom=0.10)
plt.tight_layout()
plt.show()


fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
sns.regplot(foxlen,foxdata.likes_count)
