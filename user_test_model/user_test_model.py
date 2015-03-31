import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
import feature_extractor, classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_curve, auc

train_filename = 'train.csv'
vf_train_filename = 'train_varlength_features.csv'
target_column_name = 'type'

def csv_array_to_float_comma(csv_array_string):
    return map(float, csv_array_string[1:-1].split(','))

# X is a column-indexed dict, y is a numpy array
def read_data(df_filename, vf_filename):
    df = pd.read_csv(df_filename, index_col=0)
    y_array = df[target_column_name].values
    X_dict = df.drop(target_column_name, axis=1).to_dict(orient='list')
    vf_raw = pd.read_csv(vf_filename, index_col=0)
    vf_dict = vf_raw.applymap(csv_array_to_float_comma).to_dict(orient='list')
    X_dict = dict(X_dict.items() + vf_dict.items())
    return X_dict, y_array

if __name__ == '__main__':
    print("Reading file ...")
    X_dict, y_array = read_data(train_filename, vf_train_filename)
    fe = feature_extractor.FeatureExtractor()
    X_array = fe.transform(X_dict)
    skf = StratifiedShuffleSplit(y_array, n_iter=2, test_size=0.5, random_state=57)
    print("Training file ...")
    for valid_train_is, valid_test_is in skf:
        X_valid_train = X_array[valid_train_is]
        y_valid_train = y_array[valid_train_is]
        X_valid_test = X_array[valid_test_is]
        y_valid_test = y_array[valid_test_is]
        clf = classifier.Classifier()
        clf_c = CalibratedClassifierCV(clf, cv=2, method='isotonic')
        clf_c = classifier.Classifier()
        clf_c.fit(X_valid_train, y_valid_train)
        y_valid_pred = clf_c.predict(X_valid_test)
        y_valid_proba = clf_c.predict_proba(X_valid_test)
        print y_valid_proba
#        fpr, tpr, _ = roc_curve(y_valid_test, y_valid_proba[:,1])
#        print 'auc = ', auc(fpr, tpr)
        print 'accuracy = ', accuracy_score(y_valid_pred, y_valid_test)
