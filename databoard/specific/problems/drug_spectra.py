import os
import sys
import numpy as np
import pandas as pd
from scipy import io
from sklearn.cross_validation import ShuffleSplit
import databoard.mixed_prediction as prediction
from databoard.config import submissions_path, problems_path

sys.path.append(os.path.dirname(os.path.abspath(submissions_path)))

problem_name = 'drug_spectra'  # should be the same as the file name

random_state = 57
held_out_test_size = 0.3


raw_filename = os.path.join(
    problems_path, problem_name, 'data', 'raw', 'data.mat')
train_filename = os.path.join(
    problems_path, problem_name, 'data', 'public', 'train.csv')
test_filename = os.path.join(
    problems_path, problem_name, 'data', 'private', 'test.csv')

prediction.labels = ['A', 'B', 'Q', 'R']
target_column_name_clf = 'molecule'
target_column_name_reg = 'concentration'
workflow_name = 'feature_extractor_classifier_regressor_workflow'
extra_files = [os.path.join(problems_path, problem_name,
                            'drug_spectra_datarun.py')]


def prepare_data():
    data = io.loadmat(raw_filename)
    df = pd.DataFrame(dict(
        spectra=data['Int_ABQR'].tolist(),
        solute=data['Gamme_ABQR'].ravel(),
        vial=data['Vial_ABQR'].ravel(),
        concentration=data['Conc_ABQR'].ravel(),
        molecule=data['Molecule_ABQR'].ravel()))
    skf = ShuffleSplit(
        len(df), n_iter=2, test_size=held_out_test_size,
        random_state=random_state)
    train_is, test_is = list(skf)[0]
    df_train = df.iloc[train_is]
    df_test = df.iloc[test_is]
    df_train.to_csv(train_filename, index=False)
    df_test.to_csv(test_filename, index=False)


def read_data(filename):
    df = pd.read_csv(filename)
    y_array = df[[target_column_name_clf, target_column_name_reg]].values
    X_df = df.drop([target_column_name_clf, target_column_name_reg], axis=1)
    spectra = X_df['spectra'].values
    spectra = np.array([np.array(
        dd[1:-1].split(',')).astype(float) for dd in spectra])
    X_df['spectra'] = spectra.tolist()
    return X_df, y_array


def get_train_data():
    X_train_df, y_train_array = read_data(train_filename)
    return X_train_df, y_train_array


def get_test_data():
    X_test_df, y_test_array = read_data(test_filename)
    return X_test_df, y_test_array
