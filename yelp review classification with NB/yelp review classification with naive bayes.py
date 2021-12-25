import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


yelp_df = pd.read_csv("yelp.csv")
# length of the messages
yelp_df['length'] = yelp_df['text'].apply(len)

#sns.countplot(y = 'stars', data=yelp_df)

#g = sns.FacetGrid(data=yelp_df, col='stars', col_wrap=5)
#g.map(plt.hist, 'length', bins = 20, color = 'r')

yelp_df_1 = yelp_df[yelp_df['stars']==1] # 1 star reviews
yelp_df_5 = yelp_df[yelp_df['stars']==5] # 5 star reviews

yelp_df_1_5 = pd.concat([yelp_df_1 , yelp_df_5])
#sns.countplot(x = yelp_df_1_5['stars'], label = "Count")

# remove punctuation, remove stopwords
def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


yelp_df_clean = yelp_df_1_5['text'].apply(message_cleaning)


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = message_cleaning)
yelp_countvectorizer = vectorizer.fit_transform(yelp_df_1_5['text'])


X = yelp_countvectorizer
y = yelp_df_1_5['stars'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# Evaluation
from sklearn.metrics import classification_report, confusion_matrix
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)
