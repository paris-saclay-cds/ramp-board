from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline

def model(X_train, y_train, X_test):
	clf = Pipeline([('imputer', Imputer()), 
		('rf', RandomForestClassifier(n_estimators=300))])
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	y_score = clf.predict_proba(X_test)
	return y_pred, y_score
