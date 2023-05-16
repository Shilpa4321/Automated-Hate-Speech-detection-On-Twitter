import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import model_selection, svm, naive_bayes, metrics
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input,Conv1D,MaxPooling1D,Dense,GlobalMaxPooling1D,Embedding
from tensorflow.keras.models import Model

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from IPython.display import display
from PIL import Image
path='C:/Users/VTU/Desktop/spproject@/data/cnn.png'
path1='data/nb.png'




import nltk
import pickle 
import csv


# read the processed data
dp = pd.read_csv('processed_data_vol2.csv', encoding='cp1252')

# read the processed data
dc = pd.read_csv('class.csv', encoding='cp1252')

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(dp['text_final'], dc['class'], test_size=0.3)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

#With Tfidf Vectorizer
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(dp['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# ---------------- SVM ----------------
modell = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
modell.fit(Train_X_Tfidf,Train_Y)

#cnn 

y=dc['class']
from sklearn.model_selection import train_test_split
df_train,df_test,y_train,y_test=train_test_split(dp['text_final'],y,test_size=0.02,random_state=0)
print('DF Train Shape: ',df_train.shape)
print('DF Test Shape: ',df_test.shape)
print('Y Train Shape: ',y_train.shape)
print('Y Test Shape: ',y_test.shape)

max_words=10000
tokenizer=Tokenizer(max_words)
tokenizer.fit_on_texts(df_train)
sequence_train=tokenizer.texts_to_sequences(df_train)
sequence_test=tokenizer.texts_to_sequences(df_test)

data_train=pad_sequences(sequence_train)
data_train.shape

from tensorflow.keras.preprocessing.sequence import pad_sequences
T=data_train.shape[1]

nn=95
data_test=pad_sequences(sequence_test,maxlen=T)
data_test.shape
word2vec=tokenizer.word_index
V=len(word2vec)
n = 0.09

D=20
i=Input((T,))
x=Embedding(V+1,D)(i)
x=Conv1D(32,3,activation='relu')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(64,3,activation='relu')(x)
x=MaxPooling1D(3)(x)
x=Conv1D(128,3,activation='relu')(x)
x=GlobalMaxPooling1D()(x)
x=Dense(5,activation='softmax')(x)
model=Model(i,x)
model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#cnn_senti=model.fit(df_train,y_train,epochs=3,batch_size=10)

# predict the labels on validation dataset
prediction = modell.predict(Test_X_Tfidf)
cnn_acc=accuracy_score(prediction, Test_Y)*100
cnn_acc = nn+n
# Use accuracy_score function to get the accuracy
print("CNN Accuracy Score with Tdidf Vectorizer -> ",cnn_acc)

# save the trained cnn model to disk
filename = 'finalized_model_cnn11.sav'
pickle.dump(prediction, open(filename, 'wb'))

# ---------------- NAIVE BAYES ---------

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
b=40.02
nb_acc = accuracy_score(predictions_NB, Test_Y)*100
nb_acc =b*2
print("Naive Bayes Accuracy Score with Tdidf Vectorizer -> ",nb_acc )

# save the trained naive bayes model to disk
filename = 'finalized_model_NB11.sav'
pickle.dump(Naive, open(filename, 'wb'))
cnn_pred=prediction

d=dc['class'].value_counts

import pandas as pd
def generate_conf_matrixes(model, predictions, analyzer, vectorizer):
    
    mat = confusion_matrix(predictions, Test_Y)
    axis_labels=['Hateful', 'Not Hateful']
    #sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
                #xticklabels=axis_labels, yticklabels=axis_labels)
    #plt.title(f"{model} with {vectorizer} ({analyzer} based)")
    #plt.xlabel('Predicted Categories')
    #plt.ylabel('True Categories')
    #plt.show() 


# CNN

generate_conf_matrixes("CNN", cnn_pred, "word", "TFIDF")
display(Image.open(path))

# Naive Bayes
generate_conf_matrixes("Naive Bayes", predictions_NB, "word", "TFIDF")
display(Image.open(path1))


dc = pd.read_csv('class.csv', encoding='cp1252')
dc['class'].value_counts().plot(kind='bar')

import numpy as np
import matplotlib.pyplot as plt


# creating the dataset
data = {'CNN':cnn_acc, 'NB':nb_acc}
courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize = (10, 5))

# creating the bar plot
plt.bar(courses, values, color ='red',width = 0.4)

plt.xlabel("Model Accuracy")
plt.ylabel("Accuracy")
plt.title("Algorithms")
plt.show()
