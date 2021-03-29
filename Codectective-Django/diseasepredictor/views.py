from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

def heart(request):
    heart = pd.read_csv('static/heart.csv')
    heart = heart.drop(['age', 'sex', 'fbs', 'restecg', 'slope', 'trestbps','thal'], axis=1)
       
    value = ''

    if request.method == 'POST':

        # age = float(request.POST['age'])
        # sex = float(request.POST['sex'])
        cp = float(request.POST['cp'])
        # trestbps = float(request.POST['trestbps'])
        chol = float(request.POST['chol'])
        # fbs = float(request.POST['fbs'])
        # restecg = float(request.POST['restecg'])
        thalach = float(request.POST['thalach'])
        exang = float(request.POST['exang'])
        oldpeak = float(request.POST['oldpeak'])
        # slope = float(request.POST['slope'])
        ca = float(request.POST['ca'])
        # thal = float(request.POST['thal'])

        # user_data = np.array(
        #     (age,
        #      sex,
        #      cp,
        #      trestbps,
        #      chol,
        #      fbs,
        #      restecg,
        #      thalach,
        #      exang,
        #      oldpeak,
        #      slope,
        #      ca,
        #      thal)
        # ).reshape(1, 13)

        # cp=3
        # thal=1
        # chol=233
        # thalach=150 
        # exang=0
        # oldpeak=2.3
        # ca=0
        user_data = np.array(
                    (
                    cp,
                    thalach,
                    exang,
                    oldpeak,
                    chol,
                    ca)
                ).reshape(1, 6)

        user_data=pd.DataFrame(user_data,columns =['cp','thalach','exang','oldpeak','chol','ca'])
        heart=heart.append(user_data,ignore_index = True)

        data= pd.get_dummies(heart, columns=['cp', 'ca', 'exang'])
        standardScaler = StandardScaler()
        columns_to_scale = ['chol', 'thalach', 'oldpeak']
        data[columns_to_scale] = standardScaler.fit_transform(data[columns_to_scale])
        user_data=data.values[303]
        user_data=user_data.reshape(1, 15)

        data=data.drop(labels=303, axis=0)
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(data, data['target']):
            train_set= data.loc[train_index]
            test_set= data.loc[test_index] 
        user=pd.DataFrame(user_data , columns =['chol','thalach','oldpeak','target','cp_0.0','cp_1.0','cp_2.0','cp_3.0','ca_0.0','ca_1.0','ca_2.0','ca_3.0','ca_4.0','exang_0.0','exang_1.0'])
        user_data= user.drop("target",axis=1)
        X= train_set.drop("target",axis=1)  #independent columns for training 
        Y= train_set["target"].copy()   #o/p for train set
        X_test = test_set.drop("target",axis=1)     #independent columns for testing 
        y_test = test_set["target"].copy()   
        X=X.values.tolist()
        Y=Y.values.tolist()
        X_test=X_test.values.tolist()
        y_test=y_test.values.tolist()

        error_rate = []
        for i in range(1,50): 
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X,Y)
            pred_i = knn.predict(X_test)
            error_rate.append(np.mean(pred_i != y_test))

        min = (np.argmin(error_rate) +1)
        knn = KNeighborsClassifier(n_neighbors=min)
        knn.fit(X,Y)
      
        predictions = knn.predict(user_data)

        if int(predictions[0]) == 1:
            value = 'have'
        elif int(predictions[0]) == 0:
            value = "don\'t have"

    return render(request,
                  'heart.html',
                  {
                      'context': value,
                      'title': 'Heart Disease Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'heart': True,
                      'background': 'bg-danger text-white'
                  })

def home(request):

    return render(request,
                  'index.html')

def handler404(request):
    return render(request, '404.html', status=404)
