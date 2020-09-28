import xgboost as xgb

#Using the XGBoost Classifier. I have used just a few combinations here and there without GridSearch or RandomSearch because the dataset was pretty small
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=500,seed=42,learning_rate=0.01,max_depth=5,colsample_bytree=0.75,subsample=0.7, 
                          tree_method='exact', min_child_weight=4,reg_alpha=0.005)

#fitting the model
xg_cl.fit(X_train,y_train)
