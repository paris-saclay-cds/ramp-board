from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

def model(X_train, y_train, X_test):
	clf = Pipeline([('imputer', Imputer(strategy='most_frequent')),
		('rf', ExtraTreesClassifier(n_estimators=1000))])
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_score = clf.predict_proba(X_test)
	return y_pred, y_score
