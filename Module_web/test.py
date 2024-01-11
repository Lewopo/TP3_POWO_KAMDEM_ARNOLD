import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import os 

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

k=data['Schooling'].min()
d=data['Schooling'].max()
print(d)
######
def Berthe(nom_pays):
    nom_pays=str(nom_pays)
    dev=data.loc[data['Country']==nom_pays]
    return dev
#######
# Pays=data['Country'].unique()
# lis_pays=Pays.tolist()
# print(lis_pays)
# val=input('entrer le nom ')
# data_bon=Berthe(val)
# X=data_bon[['Year','Income composition of resources', 'Schooling']]
# Y=data_bon['Life expectancy']
# #construction du modele
# regre=LinearRegression()
# regre.fit(X,Y)
# prediction=regre.predict([[2016,0.5,12]]).item()
# print(prediction)


################### Question 2 
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
    return pacf, powo

 
