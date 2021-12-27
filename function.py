from flask import Flask,render_template,request
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import html
from nltk.corpus import stopwords
import string
def analysing_message(text):
    message1=[char for char in text if char not in string.punctuation]
    message1="".join(message1)
    message2=[word for word in message1.split() if word.lower() not in stopwords.words('English')]
    return message2
file=open('spam.pkl','rb')
model=pickle.load(file)
app=Flask(__name__,template_folder='ui',static_folder='public', static_url_path='/')
@app.route('/')
def run():
    return render_template('front.html')
@app.route('/result',methods=['POST'])
def check():
    text= request.form['msg']
    text1=[html.unescape(text)]
    vectorizer=CountVectorizer(analyzer=analysing_message, vocabulary=pickle.load(open("vect.pkl", "rb")))
    message=vectorizer.transform(text1)
    print(message.shape)
    value=model.predict(message)
    return render_template('result.html',data=value)
if(__name__=="__main__"):
    app.run(debug=True)
