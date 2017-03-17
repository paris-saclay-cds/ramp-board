import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

import feature_extractor
import classifier


train_filename = 'train.csv'


def read_data(filename):
    data = pd.read_csv(filename)
    y_array = data['Survived'].values
    X_df = data.drop(['Survived', 'PassengerId'], axis=1)
    return X_df, y_array


if __name__ == '__main__':

    print("Reading file ...")
    X_df, y_array = read_data(train_filename)
    skf = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=57)

    print("Training file ...")
    scores = []
    for train_is, test_is in skf.split(X_df, y_array):
        print('--------------------------')
        X_train_df = X_df.iloc[train_is]
        y_train_array = y_array[train_is]
        X_test_df = X_df.iloc[test_is]
        y_test_array = y_array[test_is]

        fe = feature_extractor.FeatureExtractor()
        fe.fit(X_train_df, y_train_array)
        X_train_array = fe.transform(X_train_df)
        X_test_array = fe.transform(X_test_df)

        clf = classifier.Classifier()
        clf.fit(X_train_array, y_train_array)
        y_proba = clf.predict_proba(X_test_array)

        print('roc auc score = ', roc_auc_score(y_test_array, y_proba[:, 1]))
