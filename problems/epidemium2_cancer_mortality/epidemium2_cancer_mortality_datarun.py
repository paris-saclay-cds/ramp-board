import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from importlib import import_module

# should be the same as the file name
problem_name = 'epidemium2_cancer_mortality'

random_state = 57
n_CV = 2
public_train_size = 20000
train_size = 20000

raw_filename = 'data.csv.bz2'
public_train_filename = 'public_train.csv.bz2'
train_filename = 'train.csv'
test_filename = 'test.csv'

target_column_names = [
    'g_mColon (C18)',
    'g_mLiver (C22)',
    'g_mGallbladder (C23-24)',
    'g_mColon, rectum and anus (C18-21)',
    'g_mIntestine (C17-21)']
workflow_name = 'feature_extractor_regressor_workflow'
prediction_labels = None


def prepare_data(raw_data_path):
    df = pd.read_csv(os.path.join(raw_data_path, raw_filename))
    countries_encoded = LabelEncoder().fit_transform(df.country)
    skf = StratifiedShuffleSplit(n_splits=2, test_size=None,
                                 train_size=public_train_size,
                                 random_state=random_state)
    gen_skf = skf.split(countries_encoded, countries_encoded)
    public_train_is, private_is = list(gen_skf)[0]
    df_public_train = df.iloc[public_train_is]
    df_private = df.iloc[private_is]
    df_public_train.to_csv(
        os.path.join(raw_data_path, public_train_filename),
        index=False, compression='bz2')

    countries_encoded = LabelEncoder().fit_transform(df_private.country)
    skf = StratifiedShuffleSplit(
        y=countries_encoded, n_iter=2, test_size=None,
        train_size=train_size, random_state=random_state)
    train_is, test_is = list(skf)[0]
    df_train = df_private.iloc[train_is]
    df_test = df_private.iloc[test_is]
    df_train.to_csv(os.path.join(raw_data_path, train_filename), index=False)
    df_test.to_csv(os.path.join(raw_data_path, test_filename), index=False)


def read_data(filename, index_col=None):
    data = pd.read_csv(filename)
    y_array = data[target_column_names].values
    data.drop(target_column_names, axis=1)
    return data, y_array


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
    X_train_df = X_df.iloc[train_is]
    y_train_array = y_array[train_is]

    # Feature extraction
    feature_extractor = import_module('.feature_extractor', module_path)
    fe = feature_extractor.FeatureExtractor()
    fe.fit(X_train_df, y_train_array)
    X_train_array = fe.transform(X_train_df)

    regressor = import_module('.regressor', module_path)
    reg = regressor.Regressor()
    reg.fit(X_train_array, y_train_array)
    return fe, reg


def test_submission(trained_model, X_df, test_is):

    # Preparing the test (or valid) set
    X_test_df = X_df.iloc[test_is]

    fe, reg = trained_model

    # Feature extraction
    X_test_array = fe.transform(X_test_df)

    # Regression
    y_pred = reg.predict(X_test_array)
    return y_pred
