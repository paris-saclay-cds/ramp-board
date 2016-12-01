import os
from importlib import import_module
import numpy as np
import pandas as pd
from scipy import io
from sklearn.model_selection import ShuffleSplit


problem_name = 'drug_spectra'  # should be the same as the file name

random_state = 57
held_out_test_size = 0.3


raw_filename = 'data.mat'
train_filename = 'train.csv'
test_filename = 'test.csv'

target_column_name_clf = 'molecule'
target_column_name_reg = 'concentration'
workflow_name = 'feature_extractor_classifier_regressor_workflow'
labels = ['A', 'B', 'Q', 'R']


def prepare_data(raw_data_path):
    data = io.loadmat(os.path.join(raw_data_path, raw_filename))
    df = pd.DataFrame(dict(
        spectra=data['Int_ABQR'].tolist(),
        solute=data['Gamme_ABQR'].ravel(),
        vial=data['Vial_ABQR'].ravel(),
        concentration=data['Conc_ABQR'].ravel(),
        molecule=data['Molecule_ABQR'].ravel()))
    skf = ShuffleSplit(n_splits=2, test_size=held_out_test_size,
                       random_state=random_state)
    train_is, test_is = list(skf.split(df))[0]
    df_train = df.iloc[train_is]
    df_test = df.iloc[test_is]
    df_train.to_csv(os.path.join(raw_data_path, train_filename), index=False)
    df_test.to_csv(os.path.join(raw_data_path, test_filename), index=False)


def read_data(filename):
    df = pd.read_csv(filename)
    y_array = df[[target_column_name_clf, target_column_name_reg]].values
    X_df = df.drop([target_column_name_clf, target_column_name_reg], axis=1)
    spectra = X_df['spectra'].values
    spectra = np.array([np.array(
        dd[1:-1].split(',')).astype(float) for dd in spectra])
    X_df['spectra'] = spectra.tolist()
    return X_df, y_array


def get_train_data(raw_data_path):
    X_train_df, y_train_array = read_data(os.path.join(raw_data_path,
                                                       train_filename))
    return X_train_df, y_train_array


def get_test_data(raw_data_path):
    X_test_df, y_test_array = read_data(os.path.join(raw_data_path,
                                                     test_filename))
    return X_test_df, y_test_array


def train_submission(module_path, X_df, y_array, train_is):
    # Preparing the training set
    X_train_df = X_df.iloc[train_is].copy()
    y_train_array = y_array[train_is].copy()
    y_train_clf_array = y_train_array[:, 0]
    y_train_reg_array = y_train_array[:, 1].astype(float)

    # Classifier feature extraction
    feature_extractor_clf = import_module(
        '.feature_extractor_clf', module_path)
    fe_clf = feature_extractor_clf.FeatureExtractorClf()
    fe_clf.fit(X_train_df, y_train_clf_array)
    X_train_array_clf = fe_clf.transform(X_train_df)

    # Classifier
    classifier = import_module('.classifier', module_path)
    clf = classifier.Classifier()
    clf.fit(X_train_array_clf, y_train_clf_array)

    # Concatenating y_proba (derived from labels) to X_train_df
    # TODO: this should change when labels are in the db
    for i, label in enumerate(labels):
        X_train_df.loc[:, label] = (y_train_clf_array == label)

    # Regressor feature extraction
    feature_extractor_reg = import_module(
        '.feature_extractor_reg', module_path)
    fe_reg = feature_extractor_reg.FeatureExtractorReg()
    fe_reg.fit(X_train_df, y_train_reg_array)
    X_train_array_reg = fe_reg.transform(X_train_df)

    # Regressor
    regressor = import_module('.regressor', module_path)
    reg = regressor.Regressor()
    reg.fit(X_train_array_reg, y_train_reg_array)
    return fe_clf, clf, fe_reg, reg


def test_submission(trained_model, X_df, test_is):
    fe_clf, clf, fe_reg, reg = trained_model

    # Preparing the test set
    X_test_df = X_df.iloc[test_is].copy()

    # Classifier feature extraction
    X_test_array_clf = fe_clf.transform(X_test_df)

    # Classifier
    y_proba_clf = clf.predict_proba(X_test_array_clf)

    # Concatenating y_proba (derived from labels) to X_test_df
    # TODO: this should change when labels are in the db
    for i, label in enumerate(labels):
        X_test_df.loc[:, label] = y_proba_clf[:, i]

    # Regressor feature extraction
    X_test_array_reg = fe_reg.transform(X_test_df)

    # Regressor
    y_pred_reg = reg.predict(X_test_array_reg)

    return np.concatenate([y_proba_clf, y_pred_reg.reshape(-1, 1)], axis=1)
