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
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
import nltk
import pickle 
import csv
import tkinter as tk
import pandas as pd

# load the model CNN from disk
loaded_model_cnn = pickle.load(open('finalized_model_CNN.sav', 'rb'))

# load the model NB from disk
loaded_model_nb = pickle.load(open('finalized_model_NB.sav', 'rb'))

# read the processed data
df = pd.read_csv('processed_data_vol2.csv', encoding='cp1252')

# With Tfidf Vectorizer
Tfidf_vect = TfidfVectorizer()
Tfidf_vect.fit(df['text_final'])


# GUI

root= tk.Tk()

canvas = tk.Canvas(root, width = 1200, height = 720,  relief = 'raised')
canvas.pack()

label = tk.Label(root, text='Hate Speech Detection')
label.config(font=('Trebuchet MS', 32, 'bold'))
canvas.create_window(600, 155, window=label)

label = tk.Label(root, text='Enter Speech to Detect :')
label.config(font=('Avanta Garde', 22, 'bold'))
canvas.create_window(600, 250, window=label)

entry = tk.Entry(root, font=('Optima', 14),width=100) 
canvas.create_window(600, 300, window=entry)

def formatPrediction(model, output, index, user_input):
    label = tk.Label(root, text=f"'{user_input}' is recognized as: ", font=('helvetica', 24), width=70, height=3)
    canvas.create_window(600, 440, window=label)

    labelPred = tk.Label(root, text="", width=20, height=3, font=('helvetica', 18))

    if (output == 0):
        labelPred.config(text=f"{model}: Hateful") 
        labelPred.config(bg="red")

    else:
        labelPred.config(text=f"{model}: Not Hateful") 
        labelPred.config(bg="green")

    canvas.create_window(600, (480 + (55 * index)), window=labelPred)
    

def hateSpeech():
    df = pd.read_csv('results.csv')
	#data = ''
	#for i in range(len(df)):
		#data=df['text'][i]
    #user_input = entry.get() # get input sentence
    user_input = df['text'][10]

    new_input = [user_input] # put input sentence in array format (to use in prediction)
    new_input_Tfidf = Tfidf_vect.transform(new_input) # vectorize input

    # NB prediction
    new_output_nb = loaded_model_nb.predict(new_input_Tfidf)
    # Cnn prediction
    new_output_cnn = loaded_model_cnn.predict(new_input_Tfidf)
    
    # configure the prediction  labels
    formatPrediction("Naive Bayes", new_output_nb, 1, user_input)
    formatPrediction("CNN", new_output_cnn, 2, user_input)
    
button = tk.Button(text='Predict', command=hateSpeech, bg='lightblue', fg='white', font=('helvetica', 19, 'bold'))
canvas.create_window(600, 350, window=button)

root.mainloop()


