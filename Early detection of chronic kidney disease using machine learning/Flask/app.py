#`importing the necessary dependencies
import numpy as np
import pandas as pd 
from flask import Flask, request, render_template 
import pickle
import os


app = Flask(__name__) #intializing a flask app

model = pickle.load(open(os.path.join('C:/Users/asuma/OneDrive/Desktop/Early detection of chronic kidney disease using machine learning/Flask',
                             'pkl.object','CKD.pk1'), 'rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/Prediction',methods=['POST','GET'])
def prediction():
    return render_template('indexnew.html')
@app.route('/Prediction',methods=['POST','GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])#route to show the prediction in a web UI
def predict():

    #reading the input given by the user
    input_features =[float(x) for x in request.form.values()]
    features_values =[np.array(input_features)]

    features_name = ['blood_urea', 'blood glucose random' , 'anemia' ,
                     'coronary_artery_disease', 'pus_cell', 'red_blood_cells',
                     'diabetesmellitus','pedal_edema']


    df =pd.Dataframe(features_values,columns=features_name)

    #predictions using the loaded model file
    output =model.predict(df)

    #showing the prediction results in a UI# showing the prediction results in a UI
    return render_template('result.html',prediction_text=output)

if __name__ =='__main__':
    # running the app
    app.run(debug=False)

        
