from flask import Flask, jsonify, make_response, request, redirect, render_template, url_for, request
import pickle
import os

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

vectorizer = pickle.load(open('models/vectorizer.sav', 'rb'))
classifier = pickle.load(open('models/classifier.sav', 'rb'))

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/form')
def form():
	return render_template('form.html')

@app.route('/result', methods=['POST'])
def result():
    message = request.form['message']
    text_vector = vectorizer.transform([message])
    text_classified = classifier.predict(text_vector)
    result = text_classified[0]
    return render_template('result.html', prediction = result[0], message=message)

if __name__ == '__main__':
   app.run(port=5000, debug=True)
