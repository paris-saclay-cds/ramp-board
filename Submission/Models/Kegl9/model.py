from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import Imputer
import numpy as np

def model(X_train, y_train, X_test):
	where_are_NaNs = np.isnan(X_train)
	X_train[where_are_NaNs] = -1
	where_are_NaNs = np.isnan(X_test)
	X_test[where_are_NaNs] = -1

	clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=5, n_estimators=100),
                         n_estimators=20)
 	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_score = clf.predict_proba(X_test)
	return y_pred, y_score
