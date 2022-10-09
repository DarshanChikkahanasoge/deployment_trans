import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from flask import Flask,request,jsonify,render_template,url_for

data=pd.read_csv('transaction1.csv')

to_update=data.copy()
del to_update['TransactionDate']

x=to_update.drop(['IsFraud'],axis=1)
y=to_update['IsFraud']

#print(x.head())

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
x['TransactionID']=le.fit_transform(x.TransactionID.values)
#print(x.head())


sc=StandardScaler()
x3=sc.fit_transform(x)
#print(x3[:5,:])

xtrain,xtest,ytrain,ytest=train_test_split(np.array(x3),np.array(y),test_size=0.3,shuffle=True)




from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,
                                verbose=1,class_weight="balanced")
#print(xtrain[0])
rf_clf.fit(xtrain,ytrain)

pickle.dump(rf_clf,open('model.pkl','wb'))
#y_pred1=rf_clf.predict(xtest)



#print("Classification Report for Random Forest: \n", classification_report(ytest, y_pred1))

#print("Confusion Matrix of Random Forest: \n", confusion_matrix(ytest,y_pred1))


from sklearn.metrics import accuracy_score
#print('the accuracy is :',accuracy_score(ytest,y_pred1))

#id=le.transform(np.array(['T5']))
#print(id[0])

#l=np.array([[id[0],8300,0,'9088909903']])
#l=sc.transform(l)
#print(l)
#print(rf_clf.predict(l)==1)
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict(l))


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
    #return render_template('result.html',result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
