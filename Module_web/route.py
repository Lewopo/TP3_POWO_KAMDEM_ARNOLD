from flask import Flask, render_template, url_for, request,redirect,jsonify
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import pandas as pd
import os 
import base64
from  io import BytesIO
from MODULE_REG import reg 
import numpy as np

app=Flask(__name__)

#########
path=os.path.abspath('Module_web/date.csv')
    #df = pd.read_csv('C:/Users/LENOVO/Desktop/TP/Module_web/Salary_dataset.csv')
data= pd.read_csv(path)
column = data.columns.to_list()
data.columns = [a.strip() for a in column]
missing_data = pd.DataFrame(data.isnull().sum(), columns = ['missing'])
imputer = SimpleImputer(missing_values=np.nan, strategy="median")  
data['Life expectancy'] = imputer.fit_transform(data[['Life expectancy']])
data['Adult Mortality'] = imputer.fit_transform(data[['Adult Mortality']])
data['Alcohol'] = imputer.fit_transform(data[['Alcohol']])
data['Hepatitis B'] = imputer.fit_transform(data[['Hepatitis B']])
data['BMI'] = imputer.fit_transform(data[['BMI']])
data['Polio'] = imputer.fit_transform(data[['Polio']])
data['Total expenditure'] = imputer.fit_transform(data[['Total expenditure']])
data['Diphtheria'] = imputer.fit_transform(data[['Diphtheria']])
data['GDP'] = imputer.fit_transform(data[['GDP']])
data['Population'] = imputer.fit_transform(data[['Population']])
data['thinness  1-19 years'] = imputer.fit_transform(data[['thinness  1-19 years']])
data['thinness 5-9 years'] = imputer.fit_transform(data[['thinness 5-9 years']])
data['Income composition of resources'] = imputer.fit_transform(data[['Income composition of resources']])
data['Schooling'] = imputer.fit_transform(data[['Schooling']])



######
def Berthe(nom_pays):
    nom_pays=str(nom_pays)
    dev=data.loc[data['Country']==nom_pays]
    return dev
#######

#####
def figure(x,y):
    
    fig = plt.figure(figsize=(4, 4))
    p = plt.bar(x=x, height=y) 
    # plt.ylabel(y, size=10)
    # plt.xlabel(x, size=10)
   
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # return
    return base64.b64encode(buf.getbuffer()).decode("utf-8")
######
def powo(x,y,z):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x,y) 
    plt.plot(x,z, 'r-')
    buf = BytesIO()
    fig.savefig(buf, format="png")
    return base64.b64encode(buf.getbuffer()).decode("utf-8")

def kam(x,y):
    fig = plt.figure(figsize=(6, 6))
    plt.plot(x,y)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    return base64.b64encode(buf.getbuffer()).decode("utf-8")

@app.route("/")
def hello():
    return render_template("index.html")


# @app.route('/question1')
# def index():
#     cour_reg=powo(x,y,predictions)
#     cour_per=kam(range(n_iterations),cost_history)
#     return render_template("table.html",cour_reg=cour_reg,cour_per=cour_per)

@app.route('/question2')
def coef():
    return render_template("table2.html")
Pays=data['Country'].unique()
lis_pays=Pays.tolist()



@app.route('/', methods=['POST'])
def home():
    pays_choisi = request.json['Country']
    annee = int(request.json['year'])
    Icome = float(request.json['Income'])
    Ecole = float(request.json['schooling'])
    data_bon=Berthe(pays_choisi)
    X=data_bon[['Year','Income composition of resources', 'Schooling']]
    Y=data_bon['Life expectancy']
     #construction du modele
    regre=LinearRegression()
    regre.fit(X,Y)
#         #########Prediction
    prediction = regre.predict([[annee,Icome,Ecole]]).item()

    return jsonify({"response": prediction})

@app.route("/berthe")
def llo():
    Pays=data['Country'].unique()
    return render_template("table3.html", countries=Pays)



@app.route("/arnold")
def powo():
    pays_afc=['Cameroon','Gabon','Equatorial Guinea','Congo','Chad','Central African Republic']
    powo=[]
    pacf=[]
    for k in pays_afc:
        g=data.loc[data['Country']==k]
        pacf.append(k)
        X=g[['Year','Income composition of resources','Schooling']]
        Y=g['Life expectancy']
        regre=LinearRegression()
        regre.fit(X,Y)
        p=regre.predict([[2016,X['Income composition of resources'].mean(),X['Schooling'].mean()]])
        q=p.item()
        powo.append(q)
    ab_pays=['CAM','GAB','EQG','CON','TCHAD','RCA']
    hos=figure(ab_pays,powo)
    return render_template("table.html", hos=hos,powo=powo)