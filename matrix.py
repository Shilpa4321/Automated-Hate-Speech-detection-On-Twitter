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

cnn_pred=prediction
generate_conf_matrixes("CNN", cnn_pred, "word", "TFIDF")
display(Image.open(path))

# Naive Bayes
generate_conf_matrixes("Naive Bayes", predictions_NB, "word", "TFIDF")
display(Image.open(path1))


dc = pd.read_csv('class.csv', encoding='cp1252')
dc['class'].value_counts().plot(kind='bar')