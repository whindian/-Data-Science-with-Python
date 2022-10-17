# -*- coding: utf-8 -*-
"""
James Bloor
College Football Rankings'
"""
#-------------------------load libs-----------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


#-------------------------#read in data----------------------------------------
cfp_2014 = pd.DataFrame(pd.read_csv("cfb14.csv"))
cfp_2015 = pd.DataFrame(pd.read_csv("cfb15.csv"))
cfp_2016 = pd.DataFrame(pd.read_csv("cfb16.csv"))
cfp_2017 = pd.DataFrame(pd.read_csv("cfb17.csv"))

#-------------------------create train dataframe 2014-2016 & test 2017---------
cfp_train = pd.concat([cfp_2014, cfp_2015, cfp_2016])

#2017
cfp_test = pd.DataFrame(cfp_2017)

#-------------------------create a winning % for every team--------------------
cfp_test["Winning Percent"] = round(cfp_test["Win"]/cfp_test["Games"],2)
cfp_train["Winning Percent"] = round(cfp_train["Win"]/cfp_train["Games"],2)

#-------------------------feature selection------------------------------------
#train
X_train = cfp_train[['Winning Percent','Off.Rank','Off.TDs',\
                     'Off.Yards.per.Game','Def.Rank','Off.TDs.Allowed','Total.TDs.Allowed',\
                         'Yards.Per.Game.Allowed','First.Down.Rank','X4th.Down.Rank','Kickoff.Return.Rank',\
                             'Passing.Off.Rank','Interceptions.Thrown.x','Pass.Yards.Per.Game',\
                                 'Pass.Yards.Per.Game.Allowed','Punt.Return.Rank','Punt.Return.Def.Rank',\
                                     'Redzone.Off.Rank','Redzone.Def.Rank','Rushing.Off.Rank','Rushing.Yards.per.Game',\
                                         'Rushing.Def.Rank','Rush.Yards.Per.Game.Allowed','Sack.Rank','Sacks',\
                                            'Scoring.Def.Rank','Avg.Points.per.Game.Allowed','Points.Per.Game','Turnover.Rank']].values
Y_train = cfp_train[['Rank']].values

#test
X_test = cfp_test[['Winning Percent','Off.Rank','Off.TDs',\
                     'Off.Yards.per.Game','Def.Rank','Off.TDs.Allowed','Total.TDs.Allowed',\
                         'Yards.Per.Game.Allowed','First.Down.Rank','X4th.Down.Rank','Kickoff.Return.Rank',\
                             'Passing.Off.Rank','Interceptions.Thrown.x','Pass.Yards.Per.Game',\
                                 'Pass.Yards.Per.Game.Allowed','Punt.Return.Rank','Punt.Return.Def.Rank',\
                                     'Redzone.Off.Rank','Redzone.Def.Rank','Rushing.Off.Rank','Rushing.Yards.per.Game',\
                                         'Rushing.Def.Rank','Rush.Yards.Per.Game.Allowed','Sack.Rank','Sacks',\
                                            'Scoring.Def.Rank','Avg.Points.per.Game.Allowed','Points.Per.Game','Turnover.Rank']].values
Y_test = cfp_test[['Rank']].values

#-------------------------KNN--------------------------------------------------
#scale data beofre anything else
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

#And transform the test data according to that scaler
X_test = scaler.transform(X_test)

#look at best accuracy for neighbors
accuracy = []
print("Accuracy in order of k used:")
for k in range(2, 10):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    #print(k)
    knn_classifier.fit(X_train, Y_train.ravel())
    pred_k = knn_classifier.predict(X_test)
    accuracy.append(np.mean(pred_k == Y_test.ravel()))
print(accuracy)
print("\n")

#7 neighbors has the best results  
knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(X_train, Y_train.ravel())
pred_k = knn_classifier.predict(X_test)

#------------------------- 2017 Predicted--------------------------------------
pred_k = pd.DataFrame(pred_k)
prediction_2017 = pd.concat([cfp_test["Team"], pred_k], axis=1)
prediction_2017.rename(columns={0: 'P_Rank'}, inplace=True)
prediction_2017_playoffs = prediction_2017[prediction_2017["P_Rank"] == "CFP"] 
print("2017 Predicted Playoff Teams:\n",prediction_2017_playoffs["Team"])
print("\n")

#------------------------- 2017 Actual-----------------------------------------
real_2017_playoffs = cfp_2017[cfp_2017["Rank"] == "CFP"]
print("2017 Actual Playoff Teams:\n",real_2017_playoffs["Team"])
print("\n")
print("\n")

#Now onto 2018------------------------------------------------------------------------------------------

#-------------------------#read in more data-----------------------------------
cfp_2018 = pd.DataFrame(pd.read_csv("cfb18.csv"))

#-------------------------create train dataframe 2014-2017 & test 2018---------
cfp_train = pd.concat([cfp_2014, cfp_2015, cfp_2016, cfp_2017])
#2018
cfp_test = pd.DataFrame(cfp_2018)

#-------------------------create a winning % for every team--------------------
cfp_test["Winning Percent"] = round(cfp_test["Win"]/cfp_test["Games"],2)
cfp_train["Winning Percent"] = round(cfp_train["Win"]/cfp_train["Games"],2)


#-------------------------feature selection------------------------------------
#train
X_train = cfp_train[['Winning Percent','Off.Rank','Off.TDs',\
                     'Off.Yards.per.Game','Def.Rank','Off.TDs.Allowed','Total.TDs.Allowed',\
                         'Yards.Per.Game.Allowed','First.Down.Rank','X4th.Down.Rank','Kickoff.Return.Rank',\
                             'Passing.Off.Rank','Interceptions.Thrown.x','Pass.Yards.Per.Game',\
                                 'Pass.Yards.Per.Game.Allowed','Punt.Return.Rank','Punt.Return.Def.Rank',\
                                     'Redzone.Off.Rank','Redzone.Def.Rank','Rushing.Off.Rank','Rushing.Yards.per.Game',\
                                         'Rushing.Def.Rank','Rush.Yards.Per.Game.Allowed','Sack.Rank','Sacks',\
                                            'Scoring.Def.Rank','Avg.Points.per.Game.Allowed','Points.Per.Game','Turnover.Rank']].values
Y_train = cfp_train[['Rank']].values

#test
X_test = cfp_test[['Winning Percent','Off.Rank','Off.TDs',\
                     'Off.Yards.per.Game','Def.Rank','Off.TDs.Allowed','Total.TDs.Allowed',\
                         'Yards.Per.Game.Allowed','First.Down.Rank','X4th.Down.Rank','Kickoff.Return.Rank',\
                             'Passing.Off.Rank','Interceptions.Thrown.x','Pass.Yards.Per.Game',\
                                 'Pass.Yards.Per.Game.Allowed','Punt.Return.Rank','Punt.Return.Def.Rank',\
                                     'Redzone.Off.Rank','Redzone.Def.Rank','Rushing.Off.Rank','Rushing.Yards.per.Game',\
                                         'Rushing.Def.Rank','Rush.Yards.Per.Game.Allowed','Sack.Rank','Sacks',\
                                            'Scoring.Def.Rank','Avg.Points.per.Game.Allowed','Points.Per.Game','Turnover.Rank']].values
Y_test = cfp_test[['Rank']].values

#-------------------------KNN--------------------------------------------------
#scale data beofre anything else
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

#And transform the test data according to that scaler
X_test = scaler.transform(X_test)

#train and test
knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(X_train, Y_train.ravel())
pred_k = knn_classifier.predict(X_test)

#------------------------- 2018 Predicted--------------------------------------
pred_k = pd.DataFrame(pred_k)
prediction_2018 = pd.concat([cfp_test["Team"], pred_k], axis=1)
prediction_2018.rename(columns={0: 'P_Rank'}, inplace=True)

prediction_2018_playoffs = prediction_2018[prediction_2018["P_Rank"] == "CFP"] 
print("2018 Predicted Playoff Teams:\n",prediction_2018_playoffs["Team"])
print("\n")

#------------------------- 2018 Actual-----------------------------------------
real_2018_playoffs = cfp_2017[cfp_2018["Rank"] == "CFP"]
print("2018 Actual Playoff Teams:\n",real_2018_playoffs["Team"])

#Review Results