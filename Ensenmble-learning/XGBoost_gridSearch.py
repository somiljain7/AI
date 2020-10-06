from xgboost.sklearn import XGBRegressor
import datetime
from sklearn.model_selection import GridSearchCV

# Various hyper-parameters to tune
xgb1 = XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07, .01, 0.1], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4, 3, 2],
              'silent': [1],
              'subsample': [0.7, 0.75, 0.8],
              'colsample_bytree': [0.7, 0.75, 0.8],
              'n_estimators': [500, 600, 700, 800, 900, 1000]}

xgb_grid = GridSearchCV(xgb1, parameters, cv = 2, n_jobs = 5, verbose=True)

xgb_grid.fit(X_train, y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

preds = xg_cl.predict(X_test)
