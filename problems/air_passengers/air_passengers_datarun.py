import os
from importlib import import_module
import pandas as pd
from sklearn.cross_validation import train_test_split


# should be the same as the file name
problem_name = 'air_passengers'

random_state = 57
held_out_test_size = 0.2
public_train_size = 0.2

raw_filename = 'data.csv'
public_train_filename = 'public_train.csv'
train_filename = 'train.csv'
test_filename = 'test.csv'

target_column_name = 'log_PAX'
workflow_name = 'feature_extractor_regressor_with_external_data_workflow'


def read_data(df_filename):
    data = pd.read_csv(df_filename)
    return data


def prepare_data(raw_data_path):
    df = read_data(os.path.join(raw_data_path, raw_filename))
    df_train, df_test = train_test_split(
        df, test_size=held_out_test_size, random_state=random_state)
    df_public_train, df_private_train = train_test_split(
        df_train, train_size=public_train_size, random_state=random_state)
    df_public_train.to_csv(os.path.join(raw_data_path, public_train_filename),
                           index=False)
    df_private_train.to_csv(os.path.join(raw_data_path, train_filename),
                            index=False)
    df_test.to_csv(os.path.join(raw_data_path, test_filename), index=False)


def get_train_data(raw_data_path):
    df_train = read_data(os.path.join(raw_data_path, train_filename))
    X_train_df = df_train.drop(target_column_name, axis=1)
    y_train_array = df_train[target_column_name].values
    return X_train_df, y_train_array


def get_test_data(raw_data_path):
    df_test = read_data(os.path.join(raw_data_path, test_filename))
    X_test_df = df_test.drop(target_column_name, axis=1)
    y_test_array = df_test[target_column_name].values
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
