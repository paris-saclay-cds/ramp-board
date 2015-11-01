import os
import hashlib
import logging
from contextlib import contextmanager

from sklearn.externals.joblib import Memory

import config_databoard
import specific

mem = Memory(cachedir=config_databoard.cachedir, verbose=0)
logger = logging.getLogger('databoard')


# TODO: wrap get_train_data here so we can mem_cache it

def get_cv_hash(index_list):
    """We identify files output on cross validation (models, predictions)
    by hashing the point indices coming from an cv object.

    Parameters
    ----------
    index_list : np.array, shape (size_of_set,)

    Returns
    -------
    cv_hash : string
    """
    hasher = hashlib.md5()
    hasher.update(index_list)
    return hasher.hexdigest()


def get_cv_hash_list():
    _, y_train = specific.get_train_data()
    cv = specific.get_cv(y_train)
    train_is_list, _ = zip(*cv)
    return [get_cv_hash(train_is)
            for train_is in train_is_list]


def get_module_path(full_model_path):
    """Computing importable module path (/s replaced by .s) from the full model
    path.

    Parameters
    ----------
    full_model_path : of the form <root_path>/models/<team>/<model_hash>

    Returns
    -------
    module_path
    """
    return full_model_path.lstrip('./').replace('/', '.')


def get_full_model_path(model_hash, model_df):
    """Computing the full model path.

    Parameters
    ----------
    model_hash : the hash string computed on the submission in
        fetch.get_tag_uid. It usually comes from the index of the models table.

    model_df : an entry of the models table.

    Returns
    -------
    full_model_path : of the form
        <root_path>/models/<model_df['team']>/model_hash
    """
    return os.path.join(
        config_databoard.models_path, model_df['team'], model_hash)


def get_f_dir(full_model_path, subdir):
    dir = os.path.join(full_model_path, subdir)
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except OSError, e:
            if not e.errno == 17:  # file exists
                raise
        print dir
    return dir


def get_f_name(full_model_path, subdir, f_name, extension="npy"):
    return os.path.join(get_f_dir(full_model_path, subdir),
                        f_name + '.' + extension)


def get_model_f_name(full_model_path, cv_hash):
    return get_f_name(full_model_path, "model", cv_hash, "p")


def get_valid_f_name(full_model_path, cv_hash):
    return get_f_name(full_model_path, "valid", cv_hash)


def get_test_f_name(full_model_path, cv_hash):
    return get_f_name(full_model_path, "test", cv_hash)


def get_train_time_f_name(full_model_path, cv_hash):
    return get_f_name(full_model_path, "train_time", cv_hash)


def get_valid_time_f_name(full_model_path, cv_hash):
    return get_f_name(full_model_path, "valid_time", cv_hash)


@mem.cache
def get_cv():
    _, y_train = specific.get_train_data()
    return specific.get_cv(y_train)


@mem.cache
def get_train_is_list():
    cv = get_cv()
    return list(zip(*list(cv))[0])


@mem.cache
def get_test_is_list():
    cv = get_cv()
    return list(zip(*list(cv))[1])


@mem.cache
def get_true_predictions_train():
    _, y_train = specific.get_train_data()
    return specific.Predictions(y_true=y_train)


@mem.cache
def get_true_predictions_test():
    _, y_test = specific.get_test_data()
    return specific.Predictions(y_true=y_test)


@mem.cache
def get_true_predictions_valid(test_is):
    _, y_train = specific.get_train_data()
    return specific.Predictions(y_true=y_train[test_is])


@mem.cache
def get_true_predictions_valid_list():
    _, y_train = specific.get_train_data()
    test_is_list = get_test_is_list()
    true_predictions_valid_list = [specific.Predictions(
        y_true=y_train[test_is]) for test_is in test_is_list]
    return true_predictions_valid_list


@contextmanager
def changedir(dir_name):
    current_dir = os.getcwd()
    try:
        os.chdir(dir_name)
        yield
    except Exception as e:
        logger.error(e)
    finally:
        os.chdir(current_dir)
