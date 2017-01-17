import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import databoard.regression_prediction as prediction  # noqa
from databoard.config import submissions_path, problems_path,\
    starting_kit_d_name

sys.path.append(os.path.dirname(os.path.abspath(submissions_path)))

# should be the same as the file name
problem_name = 'epidemium2_cancer_mortality'

random_state = 57
n_CV = 2
public_train_size = 20000
train_size = 20000

raw_filename = os.path.join(
    problems_path, problem_name, 'data', 'raw', 'data.csv.bz2')
public_train_filename = os.path.join(
    problems_path, problem_name, starting_kit_d_name, 'public_train.csv.bz2')
train_filename = os.path.join(
    problems_path, problem_name, 'data', 'private', 'train.csv')
test_filename = os.path.join(
    problems_path, problem_name, 'data', 'private', 'test.csv')

target_column_names = [
    'g_mColon (C18)',
    'g_mLiver (C22)',
    'g_mGallbladder (C23-24)',
    'g_mColon, rectum and anus (C18-21)',
    'g_mIntestine (C17-21)']
workflow_name = 'feature_extractor_regressor_workflow'
prediction_labels = None
extra_files = [os.path.join(problems_path, problem_name,
                            'epidemium2_cancer_mortality_datarun.py')]


def prepare_data():
    df = pd.read_csv(raw_filename)
    countries_encoded = LabelEncoder().fit_transform(df.country)
    skf = StratifiedShuffleSplit(n_splits=2, test_size=None,
                                 train_size=public_train_size,
                                 random_state=random_state)
    gen_skf = skf.split(countries_encoded, countries_encoded)
    public_train_is, private_is = list(gen_skf)[0]
    df_public_train = df.iloc[public_train_is]
    df_private = df.iloc[private_is]
    df_public_train.to_csv(
        public_train_filename, index=False, compression='bz2')

    countries_encoded = LabelEncoder().fit_transform(df_private.country)
    skf = StratifiedShuffleSplit(
        n_splits=2, test_size=None,
        train_size=train_size, random_state=random_state)
    gen_skf = skf.split(countries_encoded, countries_encoded)
    train_is, test_is = list(gen_skf)[0]
    df_train = df_private.iloc[train_is]
    df_test = df_private.iloc[test_is]
    df_train.to_csv(train_filename, index=False)
    df_test.to_csv(test_filename, index=False)


def read_data(filename, index_col=None):
    data = pd.read_csv(filename)
    y_array = data[target_column_names].values
    X_df = data.drop(target_column_names, axis=1)
    return X_df, y_array


def get_train_data():
    X_train_df, y_train_array = read_data(train_filename)
    return X_train_df, y_train_array


def get_test_data():
    X_test_df, y_test_array = read_data(test_filename)
    return X_test_df, y_test_array
