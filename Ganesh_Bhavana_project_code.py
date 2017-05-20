# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 18:31:03 2017

@author: Bhavana
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D #Plot
from sklearn.model_selection import train_test_split #Preprocessing
from sklearn.model_selection import cross_val_score #Crossvalidation
from sklearn.decomposition import PCA #PCA
from sklearn.preprocessing import OneHotEncoder #Preprocessing
from sklearn import neighbors  #KNN
from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier) #Random Forest and AdaBoost
from svmutil import * #SVM using LibSVM
from sklearn.naive_bayes import GaussianNB #Naive Bayes
from sklearn.metrics import classification_report #Confusion matrix and F-score


'Read dataset into a dataframe using pandas'
print('\n\n')
print('Loading dataset')
print('\n')
data_raw = pd.read_csv('D:\Codes\Kaggle\Germanrisk\Proj_dataset_1.csv', na_values = ['.'])
data = data_raw.ix[:,1:10]
label = data_raw.ix[:,10]

'Replace text data with equivalent numeric data'

datan = data
datan = datan.replace(['male','female'],[0,1])
datan = datan.replace(['free','rent','own'],[0,1,2])
datan = datan.replace(['little','moderate','quite rich','rich'],[1,2,3,4])
datan = datan.replace(['radio/TV','repairs','domestic appliances','vacation/others','furniture/equipment','car','education','business'],[1,2,3,4,5,6,7,8])

'Clean data to get rid of samples that have missing data'
'Replacing all the missing data to 0 as in the context of dataset it means that facility is not available or not applicable'

datan = datan.fillna(0)

'Expand categorical data present in columns 1,2,3,4,5,8'
print('\n\n')
print('Processing categorical data')
print('\n')
enc = OneHotEncoder(categorical_features = [1,2,3,4,5,8],sparse = False)

enc.fit(datan)
dataprocessed = enc.transform(datan)

'Split data for test and train'
print('\n\n')
print('Split the whole dataset into two parts in ratio 80:20 for training and testing ')
print('\n')
data_train,data_test,label_train,label_test = train_test_split(dataprocessed,label,test_size = 0.2)

'Normalize the data in the non categorical columns' 
print('\n\n')
print('Normalize train and test data')
print('\n')
normtrain = data_train[:,[26,27,28]]
normtest = data_test[:,[26,27,28]]
mean = np.mean(normtrain)
std = np.std(normtrain)
normtrain = (normtrain-mean)/std
normtest = (normtest-mean)/std
data_train_f = np.concatenate((data_train[:,0:26],normtrain),axis=1)
data_test_f = np.concatenate((data_test[:,0:26],normtest),axis=1)

#datan_train_red = data_train_f
#datan_test_red = data_test_f
print('Preprocessing Completed')

'Dimension reduction - PCA'
print('\n\n')
print('PCA for dimensionality reduction')
print('\n')
pcaref = PCA(n_components=29).fit(data_train_f)
varref = pcaref.explained_variance_
plt.figure(1)
plt.stem(varref)
refline = np.full([29,1],0.1*varref[0])
plt.plot(refline,'g--')
plt.text(20,0.12*varref[0],'10% of highest variance')
plt.title('Variance of each component')
for i in range (0,29):
    if(varref[i] < 0.1*varref[0]): #Atleast greater than 10% of max variance
        ncomp = i
        break
    
#ncomp = 29
pca = PCA(n_components=ncomp).fit(data_train_f)
datan_train_red = pca.transform(data_train_f)
datan_test_red = pca.transform(data_test_f)

'The features selected are  '
features = np.zeros([1,ncomp])
corr = pca.components_
for f in range (0,ncomp):
    features[0,f] = np.argmax((corr[f,:]))
print('The features selected in order of importance are - ',features)
plt.figure(2)
plt.hist(features,bins=29)
plt.title('Features selected')


'SVM'
print('\n\n')
print('Support Vector Machine - LibSVM')
print('\n')
npoint = 50
acc = np.zeros([npoint,npoint])
gamma = np.logspace(np.log10(0.00001),np.log10(10000),npoint)
C = np.logspace(np.log10(0.1),np.log10(10000),npoint)
for k in range (0,5):
    for i in range (0,npoint):
        for j in range (0,npoint):
            prob = svm_problem(label_train.tolist(), datan_train_red.tolist())
            param = svm_parameter('-s 0 -t 2 -g '+str(gamma[i])+' -c '+str(C[j])+' -v 5 -q')
            #print(gamma[i],C[j])
            acc[i,j] = acc[i,j] + svm_train(prob,param)

acc = acc/5

block = np.argmax(acc)
row = int(np.floor(block/npoint))
col = int(block - (row * npoint))
x,y = np.meshgrid(np.log10(gamma),np.log10(C))
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,y,acc,cmap = cm.coolwarm)
plt.title('Gridsearch for parameters of the RBF kernel')
plt.show()

svmtrainacc = acc[row,col]
print('SVM: Training accuracy - ',svmtrainacc)
gammafinal = gamma[row]
Cfinal = C[col]
prob = svm_problem(label_train.tolist(), datan_train_red.tolist())
param = svm_parameter('-s 0 -t 2 -g '+str(gammafinal)+' -c '+str(Cfinal))
model = svm_train(prob,param)
#svmtrainlabel, svmtrainacc, svmtrainvals = svm_predict(label_train.tolist(),datan_train_red.tolist(),model)
svmtestlabel, svmtestacc, svmvals = svm_predict(label_test.tolist(),datan_test_red.tolist(),model)
print('SVM: Test accuracy - ',svmtestacc[0])
print('SVM: Classification Report')
print(classification_report(label_test.values,svmtestlabel))



'KNN'
print('\n\n')
print('K Nearest Neighbor Classifier')
print('\n')
score_knn = np.zeros([2,100])
for k in range (0,20):
    n = 0
    for i in range (0,100):
        n = n+1
        n_neighbors = n
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights = 'distance',p=2)#p specifies the type of distance being used
        clf.fit(datan_train_red,label_train)
        score_train_knn = (cross_val_score(clf,datan_train_red,label_train,cv=5))
        score_knn[1,i] = score_knn[1,i] + np.mean(score_train_knn)*100
        score_knn[0,i] = n

score_knn[1,:] = score_knn[1,:]/20
plt.figure(4)
plt.plot(score_knn[0,:],score_knn[1,:])
plt.title('k vs Training accuracy')

#Choosing a value for the number of neighbors based on the value that gives 
#maximum accuracy - lies in the range 45 to 60
#Below evaluation is for n_neighbors = 45

clffinal = neighbors.KNeighborsClassifier(n_neighbors = 45, weights = 'distance',p=2)
clffinal.fit(datan_train_red,label_train)
knntrainacc = np.mean(cross_val_score(clffinal,datan_train_red,label_train,cv=5))
print('KNN: Training accuracy - ',knntrainacc*100)
knntestlabel = clffinal.predict(datan_test_red)
knntestacc = clffinal.score(datan_test_red,label_test)
print('KNN: Test accuracy - ',knntestacc*100)
print('KNN: Classification Report')
print(classification_report(label_test.values,knntestlabel))


'Naive Bayes'
print('\n\n')
print('Naive Bayes')
print('\n')
gnb = GaussianNB()
score_nb = cross_val_score(gnb,datan_train_red,label_train,cv=5)
gnbtrainacc = np.mean(score_nb)
print('Naive Bayes: Training accuracy - ',gnbtrainacc*100)
gnb.fit(datan_train_red,label_train)
gnbtestlabel = gnb.predict(datan_test_red)
gnbtestacc = gnb.score(datan_train_red,label_train)
print('Naive Bayes: Test accuracy - ',gnbtestacc*100)
print('Naive Bayes: Classification Report')
print(classification_report(label_test.values,gnbtestlabel))


'AdaBoost Naive Bayes'
print('\n\n')
print('AdaBoost Naive Bayes')
print('\n')
abmodel = AdaBoostClassifier(GaussianNB(),n_estimators = 50, algorithm = 'SAMME')
scoreab = cross_val_score(abmodel,datan_train_red,label_train,cv=5)
abtrainacc = (np.mean(scoreab)*100)
print(' AdaBoost Naive Bayes: Training accuracy - ',abtrainacc)
abmodel.fit(datan_train_red,label_train)
abtestlabel = abmodel.predict(datan_test_red)
abtestacc = abmodel.score(datan_train_red,label_train)
print('AdaBoost Naive Bayes: Test accuracy - ',abtestacc*100)
print('Adaboost Naive Bayes: Classification Report')
print(classification_report(label_test.values,abtestlabel))


'Random Forest'
print('\n\n')
print('Random Forest')
print('\n')
p=0
score_rf = np.zeros([2,100])
for i in range(0,100):
    p=p+5
    rfmodel = RandomForestClassifier(n_estimators=p, n_jobs=1,random_state = 0)
    rfmodel.fit(datan_train_red,label_train)
    scorerf = cross_val_score(rfmodel,datan_train_red,label_train,cv=5)
    score_rf[0,i] = p    
    score_rf[1,i] = np.mean(scorerf)*100

plt.figure(5)
plt.plot(score_rf[0,:],score_rf[1,:])
plt.title('n_estimators vs Train accuracy')

#Choosing a value for the number of estimators that gives the maximum efficiency
#maximum accuracy is achieved in the range 100-120
#Below evaluation for n_estimators = 110

rffinal = RandomForestClassifier(n_estimators=110, n_jobs=1,random_state = 0)
rffinal.fit(datan_train_red,label_train)
rftrainacc = np.mean(cross_val_score(rffinal,datan_train_red,label_train,cv=5))
print('Random Forest: Training accuracy - ',rftrainacc*100)
rftestlabel = rffinal.predict(datan_test_red)
rftestacc = rffinal.score(datan_test_red,label_test)
print('Random Forest: Test accuracy - ',rftestacc*100)
print('Random Forest: Classification Report')
print(classification_report(label_test.values,rftestlabel))


'AdaBoost Random Forest'
print('\n\n')
print('AdaBoost Random Forest')
print('\n')
abrfmodel = AdaBoostClassifier(RandomForestClassifier(n_estimators=110, n_jobs=1,random_state = 0),n_estimators = 100, algorithm = 'SAMME')
abrfmodel.fit(datan_train_red,label_train)
scoreabrf = cross_val_score(abrfmodel,datan_train_red,label_train,cv=5)
abrftrainacc = (np.mean(scoreabrf)*100)
print('AdaBoost Random Forest: Training accuracy - ',abrftrainacc)
#abrfmodel.fit(datan_train_red,label_train)
abrftestlabel = abrfmodel.predict(datan_test_red)
abrftestacc = abrfmodel.score(datan_train_red,label_train)
print('AdaBoost Random Forest: Test accuracy - ',abrftestacc*100)
print('AdaBoost Random Forest: Classification Report')
print(classification_report(label_test.values,abrftestlabel))

