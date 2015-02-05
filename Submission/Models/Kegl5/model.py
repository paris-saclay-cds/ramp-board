from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import Imputer
import numpy as np

def model(X_train, y_train, X_test):
	where_are_NaNs = np.isnan(X_train)
	X_train[where_are_NaNs] = -1
	where_are_NaNs = np.isnan(X_test)
	X_test[where_are_NaNs] = -1

	clf = GradientBoostingClassifier(n_estimators=100, max_depth=3)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_score = clf.predict_proba(X_test)
	return y_pred, y_score
