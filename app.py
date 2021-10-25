import numpy as np
import pickle
from flask import Flask, request,jsonify, render_template


app = Flask(__name__)
model = pickle.load(open('modelSVM.pkl','rb'))
vectorizer = pickle.load(open('modelvectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    review = list()

    try:
        reviewStr = request.args.get('review')
        if not reviewStr:
            raise ValueError
    except ValueError:
        return render_template('error.html')
    
    review.append(reviewStr)
    
    dtm = vectorizer.transform(review)
    prediction = int(model.predict(dtm))
    

    text = "--->>>>> Rating for the Game  {} ".format(prediction)


    return render_template('index.html', prediction_text = text)

if __name__ == "__main__":
    app.run(debug=True)