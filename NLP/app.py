

#Importing the libraries
import time
time.clock = time.time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

# Importing the warnings
import warnings
warnings.filterwarnings('ignore')

#Loading the dataset
df = pd.read_csv("data.csv",encoding='latin-1')

df.head()

#Checking information of dataset
df.info()

#Checking the shape of the dataset
print("Shape of the dataset:", df.shape)

#Checking for the null values
df.isnull().values.any()

#Checking total number of mails
print("Count of label:\n",df['Category'].value_counts())

#Creating the new column for length of message column
df['length'] = df.Message.str.len()
df.head()

#Converting all messages to lower case
df['Message'] = df['Message'].str.lower()
df.head()

# DATA PREPROCESSING

# Tokenization
# Tokenization simply splits the message into individual tokens.

from textblob import TextBlob

def tokenize(Message):
    message = (Message, 'utf8')
    return TextBlob(Message).words

# Original messages to be tokenized.
df['Message'].head()

# Tokenized messages.
df['Message'].head().apply(tokenize)

# Lemmatization
# Convert each word in a message to its base form (lemma).

from textblob import TextBlob

def lemmatize(Message):
    message = Message.lower()
    return [word.lemma for word in TextBlob(message).words]


# Original messages to be Lemmatized.

df['Message'].head()

# Lemmatized messages.

df['Message'].head().apply(lemmatize)

# Vectorization
# Count Vectorization - Count Vectorization obtains frequency of unique words in each tokenized message.

from sklearn.feature_extraction.text import CountVectorizer

"""Bag of Words Transformer using lemmatization"""

bow_transformer = CountVectorizer(analyzer=lemmatize)
bow_transformer.fit(df['Message'])

#  Trying out the Bag of Words transformer on some dummy message.

dummy_vectorized = bow_transformer.transform(['Hey you... you of the you... This message is to you.'])
print (dummy_vectorized)

# Transforming entire set of messages in our dataset.
msgs_vectorized = bow_transformer.transform(df['Message'])
msgs_vectorized.shape

# TF-IDF Transformation
# Now that we have obtained a vectorized representation of messages in our dataset, we use it to weigh words in our dataset such that words with high frequency have a lower weight (Inverse Document Frequency). Also, this process also performs normalization of messages.

from sklearn.feature_extraction.text import TfidfTransformer

"""TFIDF Transformer using vectorized messages"""

tfidf_transformer = TfidfTransformer().fit(msgs_vectorized)

# Lets use this transformer to weigh the previous message

dummy_transformed = tfidf_transformer.transform(dummy_vectorized)
print (dummy_transformed)

# To weigh and normalize all messages in our dataset.
msgs_tfidf = tfidf_transformer.transform(msgs_vectorized)
msgs_tfidf.shape

# Naive Bayes Classifier -Having converted text messages into vectors, it can be parsed by machine learning algorithms. Naive Bayes is a classification algorithm commonly used in text processing

from sklearn.naive_bayes import MultinomialNB

"""Naive Bayes classifier trained with vectorized messages and its corresponding labels"""

nb_clf = MultinomialNB(alpha=0.25)
nb_clf.fit(msgs_tfidf, df['Category'])

# Predictions - Now that we have a trained classifier, it can be used for prediction.
msgs_pred = nb_clf.predict(msgs_tfidf)

# Accuracy Score - checks the accuracy of our classifier.

from sklearn.metrics import accuracy_score

print ('Accuracy Score: {}'.format(accuracy_score(df['Category'], msgs_pred)))

# Replace email addresses with 'email'
df['Message'] = df['Message'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress')

# Replace URLs with 'webaddress'
df['Message'] = df['Message'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')

# Replace currency symbols with 'moneysymb' (£ can by typed with ALT key + 156)
df['Message'] = df['Message'].str.replace(r'£|\$', 'dollers')
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
df['Message'] = df['Message'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumber')
   
# Replace numeric characters with 'numbr'
df['Message'] = df['Message'].str.replace(r'\d+(\.\d+)?', 'numbr')

# Remove punctuation
df['Message'] = df['Message'].str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
df['Message'] = df['Message'].str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
df['Message'] = df['Message'].str.replace(r'^\s+|\s+?$', '')

df.head()

#Removing the stopwords
import string
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'])

df['Message'] = df['Message'].apply(lambda x: " ".join(term for term in x.split() if term not in stop_words))

# New column (clean_length) after puncuations,stopwords removal
df['clean_length'] = df.Message.str.len()
df.head()

#Total length removal
print("Original Length:",df.length.sum())
print("Cleaned Length:",df.clean_length.sum())
print("Total Words Removed:",(df.length.sum()) - (df.clean_length.sum()))

#Graphical Visualisation for counting number of labels.
plt.figure(figsize=(6,4))
sns.countplot(df['Category'],palette= 'Reds')
plt.title("Counting the number of labels",fontsize=15)
plt.xticks(rotation='horizontal')
plt.show()

print(df.Category.value_counts())

#Message distribution before cleaning
f,ax = plt.subplots(1,2,figsize=(15,8))

sns.distplot(df[df['Category']=='spam']['length'],bins=20, ax=ax[0],label='Spam Message Distribution',color='r')
ax[0].set_xlabel('Spam message length')
ax[0].legend()

sns.distplot(df[df['Category']=='ham']['length'],bins=20, ax=ax[1],label='Not Spam Message Distribution',color='b')
ax[1].set_xlabel('Not Spam message length')
ax[1].legend()

plt.show()

#Message distribution after cleaning
f,ax = plt.subplots(1,2,figsize=(15,8))

sns.distplot(df[df['Category']=='spam']['clean_length'],bins=20, ax=ax[0],label='Spam Message Distribution',color='r')
ax[0].set_xlabel('Spam message length')
ax[0].legend()

sns.distplot(df[df['Category']=='ham']['clean_length'],bins=20, ax=ax[1],label='Not Spam Message Distribution',color='g')
ax[1].set_xlabel('Not a Spam message length')
ax[1].legend()

plt.show()

#Getting sense of loud words in spam 
from wordcloud import WordCloud

spams = df['Message'][df['Category']=='spam']

spam_cloud = WordCloud(width=800,height=500,background_color='white',max_words=50).generate(' '.join(spams))

plt.figure(figsize=(10,8),facecolor='b')
plt.imshow(spam_cloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

#Getting sense of loud words in not-spam 
from wordcloud import WordCloud

not_spams = df['Message'][df['Category']=='ham']

spam_cloud = WordCloud(width=800,height=500,background_color='white',max_words=50).generate(' '.join(not_spams))

plt.figure(figsize=(10,8),facecolor='b')
plt.imshow(spam_cloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# Converting the text into vectors using TF-IDF
# 1. Convert text into vectors using TF-IDF
# 2. Instantiate MultinomialNB classifier
# 3. Split feature and label


tf_vec = TfidfVectorizer()

#naive = MultinomialNB()

SVM = SVC(C=1.0, kernel='linear', degree=3 , gamma='auto')

features = tf_vec.fit_transform(df['Message'])

X = features
y = df['Category']

# Train and predict
X_train,x_test,Y_train,y_test = train_test_split(X,y,random_state=42)         #test_size=0.20 random_state=42 test_size=0.15

#naive.fit(X_train,Y_train)
#y_pred= naive.predict(x_test)

SVM.fit(X_train,Y_train)
y_pred = SVM.predict(x_test)

print ('Final score = > ', accuracy_score(y_test,y_pred))

y_pred

# Checking Classification report
print(classification_report(y_test, y_pred))

# plot confusion matrix heatmap
conf_mat = confusion_matrix(y_test,y_pred)

ax=plt.subplot()

sns.heatmap(conf_mat,annot=True,ax=ax,linewidths=5,linecolor='b',center=0)

ax.set_xlabel('Predicted Labels');ax.set_ylabel('True Labels')

ax.set_title('Confusion matrix')
ax.xaxis.set_ticklabels(['not spam','spam'])
ax.yaxis.set_ticklabels(['not spam','spam'])
plt.show()


# Observation: 
#     In naive_bayes the accuracy score was in between 99% in SVM the accuracy score is around 98%.

# **From both models we see naive_bayes performs better than SVM **

#We see naive_bayes to perform the best.
#save the best model.
import pickle
filename='Email_spam_detect.pkl'
M=open(filename,'wb')
pickle.dump(SVM,M)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
M.close()

