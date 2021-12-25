import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


##<==============Data=======================>
spam_df = pd.read_csv("emails.csv")
## adding a col of the length of the messages
spam_df['length'] = spam_df['text'].apply(len)

ham = spam_df[spam_df['spam']==0]
spam = spam_df[spam_df['spam']==1]

print( 'Spam percentage =', (len(spam) / len(spam_df) )*100,"%")
print( 'Ham percentage =', (len(ham) / len(spam_df) )*100,"%")
#sns.countplot(x = spam_df['spam'], label = "Count")



##<====================data cleaning============>
import string
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer

## Let's define a pipeline to clean up all the messages 
## The pipeline performs the following: (1) remove punctuation, (2) remove stopwords
def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


vectorizer = CountVectorizer(analyzer = message_cleaning)
spamham_countvectorizer = vectorizer.fit_transform(spam_df['text'])


##<=======================Divide the data============>
X = spamham_countvectorizer
y = label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


##<========================training the model==============>
from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


##<=====================Evaluation======================>

from sklearn.metrics import classification_report, confusion_matrix


y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)



# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


































