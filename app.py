from flask import Flask,render_template,request
import pickle
import numpy as np

model=pickle.load(open('classification.pkl','rb'))
app=Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_charges():
    age=int(request.form.get('age'))
    gender=request.form.get('gender')
    balance=float(request.form.get('balance'))
    active=int(request.form.get('active'))
    geography=request.form.get('geography')
    result=model.predict(np.array([age,gender,balance,active,geography]).reshape(1,5))
    if result[0]==0:
        result="Customer will stay"
    else:
        result="Customer is going to leave the bank"
    
    return render_template('index.html',result=result)
if __name__=='__main__':
    app.run(debug=True)

