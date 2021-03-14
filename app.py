from flask import Flask,render_template,request
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

app = Flask(__name__)

model = pickle.load(open('models\sentiment_classifier.pkl','rb'))
vectorizer = pickle.load(open('tfidf.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    text=""
    final_1 = [str(x) for x in request.form.values()]
    for element in final_1:
        text+= element
    final_x=[text]
    final= vectorizer.transform(final_x)
    prediction = model.predict(final)
    if(prediction[0] == 'POSITIVE'):
        return render_template('index.html',pred='nice comment')
    else:
        return render_template('index.html',pred='bad comment')
        

if __name__ == '__main__':
    app.run(debug=True)