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

def get_hash_string_from_indices(index_list):
    """We identify files output on cross validation (models, predictions)
    by hashing the point indices coming from an cv object.

    Parameters
    ----------
    index_list : np.array, shape (size_of_set,)

    Returns
    -------
    hash_string
    """
    hasher = hashlib.md5()
    hasher.update(index_list)
    return hasher.hexdigest()


def get_cv_hash_string_list():
    _, y_train = specific.get_train_data()
    cv = specific.get_cv(y_train)
    train_is_list, _ = zip(*cv)
    return [get_hash_string_from_indices(train_is)
            for train_is in train_is_list]


def get_hash_string_from_path(path):
    """When running testing or leaderboard, instead of recreating the hash
    strings from the cv, we just read them from the file names. This is more
    robust: only existing files will be opened when running those functions.
    On the other hand, model directories should be clean otherwise old dangling
    files will also be used. The file names are supposed to be
    <subdir>/<hash_string>.<extension>

    Parameters
    ----------
    path : string with the file name after the last '/'

    Returns
    -------
    hash_string
    """
    return path.split('/')[-1].split('.')[-2]


def get_module_path(full_model_path):
    """Computing importable module path (/s replaced by .s) from the full model
    path.

    Parameters
    ----------
    full_model_path : of the form <root_path>/models/<team>/<tag_name_alias>

    Returns
    -------
    module_path
    """
    return full_model_path.lstrip('./').replace('/', '.')


def get_full_model_path(tag_name_alias, model_df):
    """Computing the full model path.

    Parameters
    ----------
    tag_name_alias : the hash string computed on the submission in
        fetch.get_tag_uid. It usually comes from the index of the models table.

    model_df : an entry of the models table.

    Returns
    -------
    full_model_path : of the form
        <root_path>/models/<model_df['team']>/tag_name_alias
    """
    return os.path.join(
        config_databoard.models_path, model_df['team'], tag_name_alias)


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


def get_f_name(full_model_path, subdir, f_name, extension="csv"):
    return os.path.join(get_f_dir(full_model_path, subdir),
                        f_name + '.' + extension)


def get_model_f_name(full_model_path, hash_string):
    return get_f_name(full_model_path, "model", hash_string, "p")


def get_valid_f_name(full_model_path, hash_string):
    return get_f_name(full_model_path, "valid", hash_string)


def get_test_f_name(full_model_path, hash_string):
    return get_f_name(full_model_path, "test", hash_string)


def get_train_time_f_name(full_model_path, hash_string):
    return get_f_name(full_model_path, "train_time", hash_string)


def get_valid_time_f_name(full_model_path, hash_string):
    return get_f_name(full_model_path, "valid_time", hash_string)


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


@mem.cache
def get_predictions(f_name):
    return specific.Predictions(f_name=f_name)


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
