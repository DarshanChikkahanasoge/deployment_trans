import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from flask import Flask,request,jsonify,render_template,url_for
import os
data=pd.read_csv('creditcard/transaction1.csv')

to_update=data.copy()
del to_update['TransactionDate']

x=to_update.drop(['IsFraud'],axis=1)
y=to_update['IsFraud']

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
x['TransactionID']=le.fit_transform(x.TransactionID.values)


sc=StandardScaler()
x3=sc.fit_transform(x)

xtrain,xtest,ytrain,ytest=train_test_split(np.array(x3),np.array(y),test_size=0.3,shuffle=True)




from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,
                                verbose=1,class_weight="balanced")

rf_clf.fit(xtrain,ytrain)


if 'model.pkl' not in os.listdir():
    pickle.dump(rf_clf,open('model.pkl','wb'))


app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    l=le.transform(np.array([int_features[0]]))
    int_features[0]=l[0]
    final_features = sc.transform(np.array([int_features]))
    
    prediction = model.predict(final_features)

    output = 'Fraud' if prediction==1 else 'Not fraud'

    return render_template('index.html', prediction_text='The transaction is  {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)