import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
ca_loaded = joblib.load('./static/ca.joblib')
pso_loaded = joblib.load('./static/pso.joblib')
iwo_loaded = joblib.load('./static/iwo.joblib')
gbo_loaded = joblib.load('./static/gba.joblib')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features=[]
    for x in request.form.values():
        try:
            int_features.append(int(float(x)))
        except ValueError:
            # Handle the case where conversion is not possible
            print(f"Skipping invalid value: {x}")

    # Now int_features contains the successfully converted integer values
    print(int_features)
    # int_features = [int(x) for x in request.form.values()]
    mo= int_features[0]
    prediction=[]
    output=0
    if mo==1:
        int_features= int_features[1:]
        final_features = np.array(int_features)
        print(ca_loaded)
        prediction = ca_loaded.predict([final_features])

    elif mo==2:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = pso_loaded.predict(final_features)
    elif mo==3:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = iwo_loaded.predict(final_features)
    else:
        int_features= int_features[1:]
        final_features = [np.array(int_features)]
        prediction = gbo_loaded.predict(final_features)

    output = round(prediction[0], 2)
    print(output)

    return render_template('temp.html', prediction_text='The predicted critical shear stress of bed channel is {}N/m2'.format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
