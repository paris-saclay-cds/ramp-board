import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
import feature_extractor, classifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_curve, auc

train_filename = 'train.csv'
vf_train_filename = 'train_varlength_features.csv.gz'
target_column_name = 'type'

def csv_array_to_float(csv_array_string):
    return map(float, csv_array_string[1:-1].split(','))

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z

# X is a column-indexed dict, y is a numpy array
def read_data(df_filename, vf_filename):
    df = pd.read_csv(df_filename, index_col=0)
    y_array = df[target_column_name].values
    X_dict = df.drop(target_column_name, axis=1).to_dict(orient='records')
    vf_raw = pd.read_csv(vf_filename, index_col=0, compression='gzip')
    vf_dict = vf_raw.applymap(csv_array_to_float).to_dict(orient='records')
    X_dict = [merge_two_dicts(d_inst, v_inst) for d_inst, v_inst in zip(X_dict, vf_dict)]
    return X_dict, y_array

if __name__ == '__main__':
    print("Reading file ...")
    X_dict, y_array = read_data(train_filename, vf_train_filename)
    skf = StratifiedShuffleSplit(y_array, n_iter=2, test_size=0.5, random_state=57)
    print("Training file ...")
    for valid_train_is, valid_test_is in skf:
        X_valid_train_dict = [X_dict[i] for i in valid_train_is]
        y_valid_train = y_array[valid_train_is]
        X_valid_test_dict = [X_dict[i] for i in valid_test_is]
        y_valid_test = y_array[valid_test_is]
        fe = feature_extractor.FeatureExtractor()
        fe.fit(X_valid_train_dict, y_valid_train)
        X_valid_train_array = fe.transform(X_valid_train_dict)
        X_valid_test_array = fe.transform(X_valid_test_dict)

        clf = classifier.Classifier()
        clf_c = CalibratedClassifierCV(clf, cv=2, method='isotonic')
        clf_c.fit(X_valid_train_array, y_valid_train)
        y_valid_pred = clf_c.predict(X_valid_test_array)
        y_valid_proba = clf_c.predict_proba(X_valid_test_array)
        #print y_valid_proba
        print 'accuracy = ', accuracy_score(y_valid_pred, y_valid_test)
