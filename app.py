import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modelForPrediction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    Embarked = request.form['Embarked']
    Pclass = request.form['Pclass']
    Name = request.form['Name']
    SibSp = request.form['SibSp']
    Age = request.form['Age']
    Fare = request.form['Fare']
    Sex = request.form['Sex']
    if Embarked == "S":
        S = 1
        Q = 0
    elif Embarked == "Q":
        S = 0
        Q = 1
    else:
        S = 0
        Q = 0

    if Pclass == "3":
        t = 1
        s = 0
    elif Pclass == "2":
        t = 0
        s = 1
    else:
        t = 0
        s = 0
    Age = int(Age)
    SibSp = int(SibSp)
    Fare = float(Fare)
    Sex = int(Sex)

    final_features = [np.array([Age ,SibSp , Fare , t , s , Sex , Q, S])]
    filename2 = 'sandardScalar.pkl'
    loaded_model2 = pickle.load(open(filename2, 'rb'))

    prediction= model.predict(loaded_model2.fit(final_features).transform(final_features))

    if prediction == 1:
        output = "Survive"
    else:
        output = "Not Survive"

    return render_template('index.html', prediction_text='The Person is  $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)