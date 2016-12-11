import pandas as pd
from sklearn.model_selection import ShuffleSplit
import feature_extractor_clf
import feature_extractor_reg
import classifier
import regressor
from sklearn.metrics import accuracy_score
import numpy as np

train_filename = 'train.csv'
target_column_name_clf = 'molecule'
target_column_name_reg = 'concentration'


labels = np.array(['A', 'B', 'Q', 'R'])


def mare_score(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def read_data(filename):
    df = pd.read_csv(filename)
    y_df = df[[target_column_name_clf, target_column_name_reg]]
    X_df = df.drop([target_column_name_clf, target_column_name_reg], axis=1)
    spectra = X_df['spectra'].values
    spectra = np.array(
        [np.array(dd[1:-1].split(',')).astype(float) for dd in spectra])
    X_df['spectra'] = spectra.tolist()
    return X_df, y_df

if __name__ == '__main__':
    print("Reading file ...")
    X_df, y_df = read_data(train_filename)
    skf = ShuffleSplit(n_splits=2, test_size=0.2, random_state=57)
    print("Training file ...")
    for train_is, test_is in skf.split(y_df):
        print('--------------------------')
        X_train_df = X_df.iloc[train_is].copy()
        y_train_df = y_df.iloc[train_is].copy()
        X_test_df = X_df.iloc[test_is].copy()
        y_test_df = y_df.iloc[test_is].copy()
        y_train_clf = y_train_df['molecule'].values
        y_train_reg = y_train_df['concentration'].values
        y_test_clf = y_test_df['molecule'].values
        y_test_reg = y_test_df['concentration'].values

        fe_clf = feature_extractor_clf.FeatureExtractorClf()
        fe_clf.fit(X_train_df, y_train_df)
        X_train_array_clf = fe_clf.transform(X_train_df)
        X_test_array_clf = fe_clf.transform(X_test_df)

        clf = classifier.Classifier()
        clf.fit(X_train_array_clf, y_train_clf)
        y_proba_clf = clf.predict_proba(X_test_array_clf)
        y_pred_clf = labels[np.argmax(y_proba_clf, axis=1)]
        error = 1 - accuracy_score(y_test_clf, y_pred_clf)
        print('error = %s' % error)

        fe_reg = feature_extractor_reg.FeatureExtractorReg()
        for i, label in enumerate(labels):
            X_train_df.loc[:, label] = (y_train_df['molecule'] == label)
            X_test_df.loc[:, label] = y_proba_clf[:, i]
        fe_reg.fit(X_train_df, y_train_reg)
        X_train_array_reg = fe_reg.transform(X_train_df)
        X_test_array_reg = fe_reg.transform(X_test_df)

        reg = regressor.Regressor()
        reg.fit(X_train_array_reg, y_train_reg)
        y_pred_reg = reg.predict(X_test_array_reg)
        mare = mare_score(y_test_reg, y_pred_reg)
        print('mare = ', mare)
        print('combined error = ', 2. / 3 * error + 1. / 3 * mare)
