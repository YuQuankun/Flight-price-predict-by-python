
import csv
from numpy import *
import numpy as np
import operator
import pylab as pl
import pandas as pd
import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model, datasets, metrics,svm
from sklearn import neural_network
from sklearn.linear_model import ElasticNet,LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import NearestNeighbors
from flask import Flask
from flask import request

app=Flask(__name__)

def findmin(csvfile):
    with open(csvfile, mode='r', encoding='utf-8', newline='')as f:
        reader = csv.reader(f)
        tmp = []
        for line in reader:
            str = "".join(line).split('\t')
            tmp.append(str)
    label=[]#观测时间
    price=[]#观测价格
    for i in tmp:
        price.append(float(i[1]))
        label.append(i[0])
    label0=list(set(label))
    label0.sort()
    numday0=len(label0)
    numday=len(label)
    minprice=[]
    i=0
    for i in range(numday0):
        lowerp=100000.0
        for j in range (numday):
            if label[j]==label0[i]:
                if price[j]<lowerp:
                    lowerp=price[j]
        minprice.append(lowerp)
    # print(minprice)
    # print(label0)无误
    return minprice,label0

def New(csvfile,d_date):
    T=365
    predictprice=[]
    predictlabel = []  # 从外部导入
    Start=pl.datestr2num(d_date)
    for k in range(T):
        predictlabel.append(Start+k)
    minprice,label=findmin(csvfile)
    length=len(label)
    label=pl.datestr2num(label)
    i=0
    label0=[]
    for i in range(length):
        label0.append(label[i])
    label0_new=np.array(label0).reshape(-1,1)
    minprice_new=np.array(minprice).reshape(-1,1)
    predictlabel_new=np.array(predictlabel).reshape(-1,1)
    clf=svm.SVR(kernel='rbf',C=1e3,gamma=0.01)
    clf.fit(label0_new,minprice_new)
    predictprice.append(clf.predict(label0_new))
    return predictprice

@app.route('/', methods=['GET', 'POST'])
def home():
    dep_city=request.args.get("dep_city")
    arrive_city=request.args.get("arrive_city")
    if dep_city=='PEK' and arrive_city=='HGH':
        res={ 'predict_date': o_date[0],'predict_price': min_price[0],'predict_month_price':predict_month0}
        return res
    elif dep_city=='PEK' and arrive_city=='CTU':
        res={ 'predict_date': o_date[1],'predict_price': min_price[1] ,'predict_month_price':predict_month1}
        return res
    elif dep_city=='PEK' and arrive_city=='CAN':
        res={ 'predict_date': o_date[2],'predict_price': min_price[2] ,'predict_month_price':predict_month2}
        return res
    return dep_city + arrive_city

if __name__=="__main__":
    # citypair="PEK-SHA.txt"#citypair从外部获取
    csvfile_path = ["Data/PEK_HGH.csv","Data/PEK_CTU.csv","Data/PEK_CAN.csv"]  # +citypair
    s_date="2020-6-30"
    d_date = datetime.datetime(2019,12,31)
    Predict0=[]
    Predict1=[]
    Predict2=[]
    predict_month0=[]
    predict_month1=[]
    predict_month2=[]
    min_price=[]
    o_date=[]
    index=[]
    length=len(csvfile_path)
    for i in range(length):
        predict=New(csvfile_path[i],s_date)
        min_index = np.argmin(predict)
        index.append(min_index)
        min_price.append(np.min(predict))
        if i==0:
            Predict0=predict
        elif i==1:
            Predict1=predict
        elif i==2:
            Predict2=predict
    for i in range(length):
        o_date.append(d_date+datetime.timedelta( int(index[i]) ) )
    for i in range(length):
        for j in range(index[i]-20,index[i]+40):
            if i==0:
                predict_month0.append(Predict0[0][j])
            elif i==1:
                predict_month1.append(Predict1[0][j])
            else:
                predict_month2.append(Predict2[0][j])
    print(predict_month0)
    app.run()
