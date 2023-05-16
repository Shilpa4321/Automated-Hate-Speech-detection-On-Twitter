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
from PIL import Image, ImageTk

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


bg= ImageTk.PhotoImage(file= "C:\\Users\\VTU\\Desktop\\shilpa\\spproject@\\scrop.jpg")
canvas.create_image(0,0,image=bg, anchor="nw")



label = tk.Label(root, text='Hate Speech Detection')
label.config(font=('Trebuchet MS', 32, 'bold'))
canvas.create_window(600, 155, window=label)

label = tk.Label(root, text='Enter Speech Index :')
label.config(font=('Avanta Garde', 22, 'bold'))
canvas.create_window(600, 250, window=label)

entry = tk.Entry(root, font=('Optima', 14),width=100) 
canvas.create_window(600, 300, window=entry)

def formatPrediction(model, output, index, user_input):
    label = tk.Label(root, text=f"'{user_input}' is recognized as: ", font=('helvetica', 12), width=150, height=3)
    canvas.create_window(600, 440, window=label)

    labelPred = tk.Label(root, text="", width=20, height=3, font=('helvetica', 12))
     
    data={}
    yes=[]
    no=[]
	
    with open('final_result.csv','a',newline='') as f:
        wrt=csv.writer(f)
        #wrt.writerow(['Label','Message'])
        if (output == 0):
            labelPred.config(text=f"{model}: Hateful") 
            labelPred.config(bg="red")
            #yes.append(user_input)
            #data={'Hateful':yes}
            wrt.writerow(['Hateful',user_input,model])
		
        else:
            labelPred.config(text=f"{model}: Not Hateful") 
            labelPred.config(bg="green")
            #no.append(user_input)
            #data={'Not-Hateful':no}
            wrt.writerow(['Non-Hateful',user_input,model])
		
    #df=pd.DataFrame(data)
    #print(df)
    #df.to_csv('Final_result.csv')
    #print(data)

    canvas.create_window(600, (480 + (55 * index)), window=labelPred)
	
def viewdata():
        wn4=tk.Tk()
        wn4.geometry("1200x400+0+0")
        wn4.configure(background="Lightyellow3")
        #ts = time.time()
        #date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        #df=pd.read_csv('Final_result.csv')
        #df=df.drop_duplicates(subset=["Message"],keep=False)
        #df.to_csv('Final_result.csv')
        with open("Final_result.csv", newline = "") as file:
            reader = csv.reader(file)
            # r and c tell us where to grid the labels
            r = 20
            s=''
            for col in reader:
                c = 0
                for row in col:
                    print(row)                    
                    if row!=s:
                    # i've added some styling
                        label = tk.Label(wn4, width = 50, height = 2, \
                                       text = row)
                        label.grid(row = r, column = c)
                    s=row
                    c += 1
                r += 1
        def destroyWin():
                wn4.destroy()        
    
    

def hateSpeech():
    df = pd.read_csv('results.csv')
    data = []
    with open('final_result.csv','w',newline='') as f:
        wrt=csv.writer(f)
		
    for i in range(10):
        data.append(df['text'][i])
    #print(data)
        user_input = int(entry.get()) # get input sentence
        user_input = df['text'][user_input]

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

#b3=tk.Button(text="View Result" ,command=viewdata,padx=10 ,pady=10 ,bd=8 ,fg="Black", font=('arial' ,10 ,'bold'), bg="powder blue")
#canvas.create_window(800, 550, window=b3)

import table

root.mainloop()


