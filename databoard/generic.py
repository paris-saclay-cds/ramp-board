import os
import sys
import csv
import glob
import pickle
import random
import timeit
import hashlib
import logging
import multiprocessing
import numpy as np
import pandas as pd
from scipy import io
from functools import partial
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals.joblib import Memory

from .model import ModelState
from .config_databoard import (
    root_path, 
    models_path, 
    ground_truth_path, 
    n_processes,
    cachedir,
    private_data_path,
)
import specific

# We needed to make the sets global because Parallel hung
# when we passed a list of dictionaries to the function
# to be dispatched, save_scores()
class DataSets():

    def __init__(self):
        pass

    def set_sets(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_sets(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

    def get_training_sets(self):
        return self.X_train, self.y_train

    def get_test_sets(self):
        return self.X_test, self.y_test

mem = Memory(cachedir=cachedir)
logger = logging.getLogger('databoard')
data_sets = DataSets()

def get_hash_string_from_indices(index_list):
    """We identify files output on cross validation (models, predictions)
    by hashing the point indices coming from an skf object.

    Parameters
    ----------
    test_is : np.array, shape (size_of_set,)

    Returns
    -------
    hash_string
    """
    hasher = hashlib.md5()
    hasher.update(index_list)
    return hasher.hexdigest()

def get_hash_string_from_path(path):
    """When running testing or leaderboard, instead of recreating the hash 
    strings from the skf, we just read them from the file names. This is more
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
    return os.path.join(models_path, model_df['team'], tag_name_alias)

def get_f_dir(full_model_path, subdir):
    dir = os.path.join(full_model_path, subdir)
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir

def get_f_name(full_model_path, subdir, f_name, extension = "csv"):
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

def get_ground_truth_valid_f_name(hash_string):
    return get_f_name(ground_truth_path, "ground_truth_valid", hash_string)

def get_ground_truth_test_f_name():
    return get_f_name(ground_truth_path, '.', "ground_truth_test")

def get_hash_strings_from_ground_truth():
    ground_truth_f_names = glob.glob(ground_truth_path + "/ground_truth_valid/*")
    hash_strings = [get_hash_string_from_path(path) 
                    for path in ground_truth_f_names]
    return hash_strings

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


def setup_ground_truth():
    """Setting up the GroundTruth subdir, saving y_test for each fold in skf. 
    File names are valid_<hash of the train index vector>.csv.

    Parameters
    ----------
    gt_paths : ground truth path
    y : array-like, shape = [n_instances]
        the label vector
    """
    os.rmdir(ground_truth_path)  # cleanup the ground_truth
    os.mkdir(ground_truth_path)
    _, y_train, _, y_test, skf = specific.split_data()
    f_name_test = get_ground_truth_test_f_name()
    np.savetxt(f_name_test, y_test, delimiter="\n", fmt='%s')

    logger.debug('Ground truth files...')
    scores = []
    for train_is, test_is in skf:
        hash_string = get_hash_string_from_indices(train_is)
        f_name_valid = get_ground_truth_valid_f_name(hash_string)
        logger.debug(f_name_valid)
        np.savetxt(f_name_valid, y_train[test_is], delimiter="\n", fmt='%s')

def test_trained_model(trained_model, X, skf_is = None):
    if skf_is == None:
        skf_is = ([], range(len(X))) # test on all point
    start = timeit.default_timer()
    test_model_output = specific.test_model(trained_model, X, skf_is)
    end = timeit.default_timer()
    return test_model_output, end - start

def train_on_fold(skf_is, full_model_path):
    """Trains the model on a single fold. It requires specific to contain
    a train_model function that takes the module_path, X_train, y_train, and
    an skf_is containing the train_train and valid_train indices. Most of the
    time it will simply train on X_train[valid_train] but in the case of time
    series it may do feature extraction on the full file (using always the past).

    Parameters
    ----------
    skf_is : a pair of indices (train_train_is, valid_train_is)
    full_model_path : of the form <root_path>/models/<team>/<tag_name_alias>
    """
    valid_train_is, _ = skf_is
    # We had a bug when several threads wanted to access the same variable
    # Plus splitting is a much faster operation, not sure why (get makes copies?)
    #X_train, y_train = data_sets.get_training_sets()
    X_train, y_train, X_test, y_test, skf = specific.split_data()

    hash_string = get_hash_string_from_indices(valid_train_is)
    logger.info("Training : %s" % hash_string)
    start = timeit.default_timer()
    open(os.path.join(full_model_path, "__init__.py"), 'a').close()  # so to make it importable
    module_path = get_module_path(full_model_path)

    trained_model = specific.train_model(module_path, X_train, y_train, skf_is)
    end = timeit.default_timer()
    with open(get_train_time_f_name(full_model_path, hash_string), 'w') as f:
        f.write(str(end - start)) # saving running time

    #try:
    #    with open(get_model_f_name(full_model_path, hash_string), 'w') as f:
    #        pickle.dump(trained_model, f) # saving the model
    #except Exception as e:
    #    logger.error("Cannot pickle trained model\n{}".format(e))
    #    os.remove(get_model_f_name(full_model_path, hash_string))


    # in case we want to train and test without going through pickling, for
    # example, because pickling doesn't work, we return the model
    return trained_model

def test_on_fold(skf_is, full_model_path):
    valid_train_is, _ = skf_is
    hash_string = get_hash_string_from_indices(valid_train_is)

    logger.info("Testing : %s" % hash_string)
    try:
        logger.info("Loading from pickle")
        with open(get_model_f_name(full_model_path, hash_string), 'r') as f:
            trained_model = pickle.load(f)
    except IOError, e: # no pickled model, retrain
        logger.info("No pickle, retraining")
        trained_model = train_on_fold(skf_is, full_model_path)

    #X_test, _ = data_sets.get_test_sets()
    X_train, y_train, X_test, y_test, skf = specific.split_data()
    test_model_output, _ = test_trained_model(trained_model, X_test)
    test_f_name = get_test_f_name(full_model_path, hash_string)
    test_model_output.save_predictions(test_f_name)

# TODO: Should be factorized
def train_valid_and_test_on_fold(skf_is, full_model_path):
    trained_model = train_on_fold(skf_is, full_model_path)

    valid_train_is, valid_test_is = skf_is
    #X_train, y_train = data_sets.get_training_sets()
    X_train, y_train, X_test, y_test, skf = specific.split_data()
    hash_string = get_hash_string_from_indices(valid_train_is)

    logger.info("Validating : %s" % hash_string)
    valid_model_output, test_time = test_trained_model(
        trained_model, X_train, skf_is)
    valid_f_name = get_valid_f_name(full_model_path, hash_string)
    valid_model_output.save_predictions(valid_f_name)
    with open(get_valid_time_f_name(full_model_path, hash_string), 'w') as f:
        f.write(str(test_time))  # saving running time

    logger.info("Testing : %s" % hash_string)
    #X_test, _ = data_sets.get_test_sets()
    X_train, y_train, X_test, y_test, skf = specific.split_data()
    test_model_output, _ = test_trained_model(trained_model, X_test)
    test_f_name = get_test_f_name(full_model_path, hash_string)
    test_model_output.save_predictions(test_f_name)

def train_and_valid_on_fold(skf_is, full_model_path):
    trained_model = train_on_fold(skf_is, full_model_path)

    valid_train_is, valid_test_is = skf_is
    #X_train, y_train = data_sets.get_training_sets()
    X_train, y_train, X_test, y_test, skf = specific.split_data()
    hash_string = get_hash_string_from_indices(valid_train_is)

    logger.info("Validating : %s" % hash_string)
    valid_model_output, test_time = test_trained_model(
        trained_model, X_train, skf_is)
    valid_f_name = get_valid_f_name(full_model_path, hash_string)
    valid_model_output.save_predictions(valid_f_name)
    with open(get_valid_time_f_name(full_model_path, hash_string), 'w') as f:
        f.write(str(test_time))  # saving running time


def train_valid_and_test_model(full_model_path, skf):
    Parallel(n_jobs=n_processes, verbose=5)\
        (delayed(train_valid_and_test_on_fold)(skf_is, full_model_path)
        for skf_is in skf)

#@mem.cache
def train_and_valid_model(full_model_path, skf):
    #Uncomment this and comment out the follwing two lines if
    #parallel training is not working
    #for skf_is in skf:
    #    train_and_valid_on_fold(skf_is, full_model_path)

    # from joblib.pool import has_shareable_memory

    Parallel(n_jobs=n_processes, verbose=5)\
        (delayed(train_and_valid_on_fold)(skf_is, full_model_path)
        for skf_is in skf)
    
    #partial_train = partial(train_and_valid_on_fold, full_model_path=full_model_path)
    #pool = multiprocessing.Pool(processes=n_processes)
    #pool.map(partial_train, skf)
    #pool.close()

    #import pprocess
    #import time
    ##pprocess.pmap(partial_train, skf, limit=n_processes)

    #class MyExchange(pprocess.Exchange):

    #    "Parallel convenience class containing the array assignment operation."

    #    def store_data(self, ch):
    #        r = ch.receive()
    #        print "*****************************************************"
    #        print r

    #exchange = MyExchange(limit=n_processes)

    ## Wrap the calculate function and manage it.

    #calc = exchange.manage(pprocess.MakeParallel(train_and_valid_on_fold))

    ## Perform the work.

    #for skf_is, i in zip(skf, range(n_processes)):
    #    calc(skf_is, full_model_path)
    #    time.sleep(5)

    #exchange.finish()
 
def train_and_valid_models(orig_models_df, last_time_stamp=None):
    models_df = orig_models_df.sort("timestamp")
        
    if models_df.shape[0] == 0:
        logger.info("No models to train.")
        return

    logger.info("Reading data")
    X_train, y_train, X_test, y_test, skf = specific.split_data()
    data_sets.set_sets(X_train, y_train, X_test, y_test)

    for idx, model_df in models_df.iterrows():
        if model_df['state'] in ["ignore"]:
            continue

        full_model_path = get_full_model_path(idx, model_df)

        logger.info("Training : {}/{}".format(model_df['team'], model_df['model']))

        try:
            train_and_valid_model(full_model_path, skf)
            # failed_models.drop(idx, axis=0, inplace=True)
            orig_models_df.loc[idx, 'state'] = "trained"
        except Exception, e:
            orig_models_df.loc[idx, 'state'] = "error"
            logger.error("Training failed with exception: \n{}".format(e))

            # TODO: put the error in the database instead of a file
            # Keep the model folder clean.
            with open(get_f_name(full_model_path, '.', "error", "txt"), 'w') as f:
                error_msg = str(e)
                cut_exception_text = error_msg.rfind('--->')
                if cut_exception_text > 0:
                    error_msg = error_msg[cut_exception_text:]
                f.write("{}".format(error_msg))

def train_valid_and_test_models(orig_models_df, last_time_stamp=None):
    models_df = orig_models_df.sort("timestamp")
        
    if models_df.shape[0] == 0:
        logger.info("No models to train.")
        return

    logger.info("Reading data")
    X_train, y_train, X_test, y_test, skf = specific.split_data()
    data_sets.set_sets(X_train, y_train, X_test, y_test)

    for idx, model_df in models_df.iterrows():
        if model_df['state'] in ["ignore"]:
            continue

        full_model_path = get_full_model_path(idx, model_df)

        logger.info("Training : {}/{}".format(model_df['team'], model_df['model']))

        try:
            train_valid_and_test_model(full_model_path, skf)
            # failed_models.drop(idx, axis=0, inplace=True)
            orig_models_df.loc[idx, 'state'] = "tested"
        except Exception, e:
            orig_models_df.loc[idx, 'state'] = "error"
            logger.error("Training/testing failed with exception: \n{}".format(e))

            # TODO: put the error in the database instead of a file
            # Keep the model folder clean.
            with open(get_f_name(full_model_path, '.', "error", "txt"), 'w') as f:
                error_msg = str(e)
                cut_exception_text = error_msg.rfind('--->')
                if cut_exception_text > 0:
                    error_msg = error_msg[cut_exception_text:]
                f.write("{}".format(error_msg))

#@mem.cache
def test_model(full_model_path, skf):

    Parallel(n_jobs=n_processes, verbose=5)\
        (delayed(test_on_fold)(skf_is, full_model_path) for skf_is in skf)

    #partial_test = partial(test_on_fold, full_model_path=full_model_path)
    #pool = multiprocessing.Pool(processes=n_processes)
    #pool.map(partial_test, skf)
    #pool.close()
    #pprocess.pmap(partial_test, skf, limit=n_processes)


def test_models(orig_models_df, last_time_stamp=None):
    models_df = orig_models_df.sort("timestamp")
        
    if models_df.shape[0] == 0:
        logger.info("No models to test.")
        return

    logger.info("Reading data")
    X_train, y_train, X_test, y_test, skf = specific.split_data()
    data_sets.set_sets(X_train, y_train, X_test, y_test)

    for idx, model_df in models_df.iterrows():
        if model_df['state'] in ["ignore"]:
            continue

        full_model_path = get_full_model_path(idx, model_df)

        logger.info("Testing : {}/{}".format(model_df['team'], model_df['model']))
        logger.info("Testing : " + idx)

        try:
            test_model(full_model_path, skf)
            # failed_models.drop(idx, axis=0, inplace=True)
            orig_models_df.loc[idx, 'state'] = "tested"
        except Exception, e:
            orig_models_df.loc[idx, 'state'] = "test_error"
            # trained_models.drop(idx, axis=0, inplace=True)
            logger.error("Testing failed with exception: \n{}".format(e))

            # TODO: put the error in the database instead of a file
            # Keep the model folder clean.
            with open(get_f_name(full_model_path, '.', "test_error", "txt"), 'w') as f:
                error_msg = str(e)
                cut_exception_text = error_msg.rfind('--->')
                if cut_exception_text > 0:
                    error_msg = error_msg[cut_exception_text:]
                f.write("{}".format(error_msg))

def get_predictions_list(models_df, subdir, hash_string):
    predictions_list = [] 
    for idx, model_df in models_df.iterrows():
        full_model_path = get_full_model_path(idx, model_df)
        predictions_path = get_f_name(full_model_path, subdir, hash_string)
        # When the prediction file is not there (eg because the model has
        # not yet been tested), we set the preicition to None. The list can
        # be still used if that model is not needed (for example, not in 
        # best_index_list). TODO: handle this more intellingently
        try:
            predictions = specific.prediction_type.PredictionArrayType(
                f_name=predictions_path)
        except IOError:
            predictions = None
        predictions_list.append(predictions)
    return predictions_list

def leaderboard_execution_times(models_df):
    hash_strings = get_hash_strings_from_ground_truth()
    num_cv_folds = len(hash_strings)
    leaderboard = pd.DataFrame(index=models_df.index)
    leaderboard['train time'] = np.zeros(models_df.shape[0])
    # we name it "test" (not "valid") bacause this is what it is from the 
    # participant's point of view (ie, "public test")
    leaderboard['test time'] = np.zeros(models_df.shape[0])

    if models_df.shape[0] != 0:
        for hash_string in hash_strings:
            for idx, model_df in models_df.iterrows():
                full_model_path = get_full_model_path(idx, model_df)
                try: 
                    with open(get_train_time_f_name(full_model_path, hash_string), 'r') as f:
                        leaderboard.loc[idx, 'train time'] += abs(float(f.read()))
                except IOError:
                    logger.debug("Can't open %s, setting training time to 0" % 
                        get_train_time_f_name(full_model_path, hash_string))
                try: 
                    with open(get_valid_time_f_name(full_model_path, hash_string), 'r') as f:
                        leaderboard.loc[idx, 'test time'] += abs(float(f.read()))
                except IOError:
                    logger.debug("Can't open %s, setting testing time to 0" % 
                        get_test_time_f_name(full_model_path, hash_string))

    leaderboard['train time'] = map(int, leaderboard['train time'] / num_cv_folds)
    leaderboard['test time'] = map(int, leaderboard['test time'] / num_cv_folds)
    logger.info("Classical leaderboard train times = {}".
        format(leaderboard['train time'].values))
    logger.info("Classical leaderboard valid times = {}".
        format(leaderboard['test time'].values))
    return leaderboard

# TODO: should probably go to specific, or even output_type
def get_ground_truth(ground_truth_filename):        
    return pd.read_csv(
        ground_truth_filename, names=['ground_truth']).values.flatten()

def get_scores(models_df, hash_string, subdir = "valid", calibrate=False):
    logger.info("Evaluating : {}".format(hash_string))
    if subdir == "valid":
        ground_truth_filename = get_ground_truth_valid_f_name(hash_string)
    else: #subdir == "test"
        ground_truth_filename = get_ground_truth_test_f_name()
    ground_truth = get_ground_truth(ground_truth_filename)
    specific.score.set_eps(0.01 / len(ground_truth))
 
    if calibrate:
        # Calibrating predictions
        if subdir == "valid":
            uncalibrated_predictions_list = get_predictions_list(
                models_df, "valid", hash_string)
            predictions_list = calibrate_predictions(
                uncalibrated_predictions_list, ground_truth)

        else: #subdir == "test":
            uncalibrated_test_predictions_list = get_predictions_list(
                models_df, "test", hash_string)
            valid_predictions_list = get_predictions_list(
                models_df, "valid", hash_string)
            ground_truth_valid_filename = get_ground_truth_valid_f_name(hash_string)
            ground_truth_valid = get_ground_truth(ground_truth_valid_filename)
            predictions_list = calibrate_test_predictions(
                uncalibrated_test_predictions_list, valid_predictions_list, 
                ground_truth_valid)
    else:
        predictions_list = get_predictions_list(models_df, subdir, hash_string)

    scores = np.array([specific.score(ground_truth, predictions) 
                       for predictions in predictions_list])
    return scores

def calibrate(y_probas_array, ground_truth):
    skf = StratifiedShuffleSplit(ground_truth, n_iter=1, test_size=0.5, 
                                 random_state=specific.random_state)
    calibrated_proba_array = np.empty(y_probas_array.shape)
    fold1_is, fold2_is = list(skf)[0]
    folds = [(fold1_is, fold2_is), (fold2_is, fold1_is)]
    calibrator = IsotonicCalibrator()
    #calibrator = NolearnCalibrator()
    for fold_train_is, fold_test_is in folds:
        calibrator.fit(
            np.nan_to_num(y_probas_array[fold_train_is]), ground_truth[fold_train_is])
        calibrated_proba_array[fold_test_is] = calibrator.predict_proba(
            np.nan_to_num(y_probas_array[fold_test_is]))
    return calibrated_proba_array

def calibrate_predictions(uncalibrated_predictions_list, ground_truth):
    predictions_list = []
    for uncalibrated_predictions in uncalibrated_predictions_list:
        y_pred_array, y_probas_array = uncalibrated_predictions.get_predictions()
        calibrated_y_probas_array = calibrate(y_probas_array, ground_truth)
        calibrated_y_pred_array = get_y_pred_array(calibrated_y_probas_array)
        predictions = specific.prediction_type.PredictionArrayType(
            y_pred_array=calibrated_y_pred_array, 
            y_probas_array=calibrated_y_probas_array)
        predictions_list.append(predictions)
    return predictions_list

def leaderboard_classicial_mean_of_scores(models_df, subdir = "valid", calibrate = False):
    hash_strings = get_hash_strings_from_ground_truth()
    mean_scores = np.array([specific.score.zero() 
                            for _ in range(models_df.shape[0])])
    mean_scores_calibrated = np.array([specific.score.zero() 
                            for _ in range(models_df.shape[0])])

    if models_df.shape[0] != 0:
        scores_list = Parallel(n_jobs=n_processes, verbose=0)\
            (delayed(get_scores)(models_df, hash_string, subdir)
             for hash_string in hash_strings)
        mean_scores = np.mean(np.array(scores_list), axis=0)

        #for hash_string in hash_strings:
        #    scores = get_scores(models_df, hash_string, subdir)
        #    mean_scores += scores
        #    mean_scores /= len(hash_strings)

        if calibrate:
            scores_list = Parallel(n_jobs=n_processes, verbose=0)\
                (delayed(get_scores)(models_df, hash_string, subdir, calibrate)
                 for hash_string in hash_strings)
            mean_scores_calibrated = np.mean(np.array(scores_list), axis=0)
        
            #for hash_string in hash_strings:
            #    scores_list = get_scores(models_df, hash_string, subdir, calibrate)
            #    mean_scores_calibrated += scores_list
            #mean_scores_calibrated /= len(hash_strings)


    logger.info("classical leaderboard mean {} scores = {}".
        format(subdir, mean_scores))
    leaderboard = pd.DataFrame({'score': mean_scores}, index=models_df.index)
    if calibrate:
        logger.info("classical leaderboard calibrated mean {} scores = {}".
            format(subdir, mean_scores_calibrated))
        leaderboard['calib score'] = mean_scores_calibrated
    return leaderboard.sort(columns=['score'])

def best_combine(predictions_list, ground_truth, best_indexes):
    """Finds the model that minimizes the score if added to y_preds[indexes].

    Parameters
    ----------
    y_preds : array-like, shape = [k_models, n_instances], binary
    best_indexes : array-like, shape = [max k_models], a set of indices of 
        the current best model
    Returns
    -------
    best_indexes : array-like, shape = [max k_models], a list of indices. If 
    no model imporving the input combination, the input index set is returned. 
    otherwise the best model is added to the set. We could also return the 
    combined prediction (for efficiency, so the combination would not have to 
    be done each time; right now the algo is quadratic), but first I don't think
    any meaningful rules will be associative, in which case we should redo the
    combination from scratch each time the set changes.
    """
    best_predictions = specific.prediction_type.combine(predictions_list, best_indexes)
    best_index = -1
    # Combination with replacement, what Caruana suggests. Basically, if a model
    # added several times, it's upweighted, leading to integer-weighted ensembles
    for i in range(len(predictions_list)):
        combined_predictions = specific.prediction_type.combine(
            predictions_list, np.append(best_indexes, i))
        if specific.score(ground_truth, combined_predictions) > \
           specific.score(ground_truth, best_predictions):
            best_predictions = combined_predictions
            best_index = i
            #print i, specific.score(ground_truth, combined_predictions)
    if best_index > -1:
        return np.append(best_indexes, best_index)
    else:
        return best_indexes

# TODO: this shuold be an inner consistency operation in OutputType
def get_y_pred_array(y_probas_array):
    return np.array([specific.prediction_type.labels[y_probas.argmax()] 
                     for y_probas in y_probas_array])

from .isotonic import IsotonicRegression
from sklearn.cross_validation import StratifiedShuffleSplit

class IsotonicCalibrator():
    def __init__(self):
        pass
    
    def fit(self, X_array, y_list, plot=False):
        labels = np.sort(np.unique(y_list))
        self.calibrators = []
        for class_index in range(X_array.shape[1]):
            calibrator = IsotonicRegression(
                y_min=0., y_max=1., out_of_bounds='clip')
            class_indicator = np.array([1 if y == labels[class_index] else 0
                                        for y in y_list])
            #print "before"
            #np.save("x", X_array[:,class_index])
            #np.save("y", class_indicator)
            #x = np.load("x.npy")
            #y = np.load("y.npy")
            calibrator = IsotonicRegression(
                y_min=0, y_max=1, out_of_bounds='clip')
            #print x
            #print y
            #calibrator.fit(x, y)
            calibrator.fit(np.nan_to_num(X_array[:,class_index]), 
                class_indicator)            
            #print "after"
            self.calibrators.append(calibrator)
            
    def predict_proba(self, y_probas_array_uncalibrated):
        num_classes = y_probas_array_uncalibrated.shape[1]
        y_probas_array_transpose = np.array(
            [self.calibrators[class_index].predict(
                np.nan_to_num(y_probas_array_uncalibrated[:,class_index]))
             for class_index in range(num_classes)])
        sum_rows = np.sum(y_probas_array_transpose, axis=0)
        y_probas_array_normalized_transpose = np.divide(
            y_probas_array_transpose, sum_rows)
        return y_probas_array_normalized_transpose.T

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
import os
os.environ["OMP_NUM_THREADS"] = "1"

import theano
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

class NolearnCalibrator(BaseEstimator):
    def __init__(self):
        self.net = None
        self.label_encoder = None
    
    def fit(self, X_array, y_pred_array):
        labels = np.sort(np.unique(y_pred_array))
        num_classes = X_array.shape[1]
        layers0 = [('input', InputLayer),
                   ('dense', DenseLayer),
                   ('output', DenseLayer)]
        X = X_array.astype(theano.config.floatX)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        self.label_encoder = LabelEncoder()
        #y = class_indicators.astype(np.int32)
        y = self.label_encoder.fit_transform(y_pred_array).astype(np.int32)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        num_features = X.shape[1]
        self.net = NeuralNet(layers=layers0,
                             input_shape=(None, num_features),
                             dense_num_units=num_features,
                             output_num_units=num_classes,
                             output_nonlinearity=softmax,

                             update=nesterov_momentum,
                             update_learning_rate=0.01,
                             update_momentum=0.9,

                             eval_size=0.2,
                             verbose=0,
                             max_epochs=100)
        self.net.fit(X, y)

    def predict_proba(self, y_probas_array_uncalibrated):
        num_classes = y_probas_array_uncalibrated.shape[1]
        X = y_probas_array_uncalibrated.astype(theano.config.floatX)
        X = self.scaler.fit_transform(X)
        return self.net.predict_proba(X)

def calibrate_test(y_probas_array, ground_truth, test_y_probas_array):
    calibrator = IsotonicCalibrator()
    #calibrator = NolearnCalibrator()
    calibrator.fit(np.nan_to_num(y_probas_array), ground_truth)
    calibrated_test_y_probas_array = calibrator.predict_proba(np.nan_to_num(test_y_probas_array))
    return calibrated_test_y_probas_array


def calibrate_test_predictions(uncalibrated_test_predictions_list, 
                               valid_predictions_list, ground_truth_valid,
                               best_index_list = []):
    if len(best_index_list) == 0:
        best_index_list = range(len(uncalibrated_test_predictions_list))
    test_predictions_list = []
    for best_index in best_index_list:
        _, uncalibrated_test_y_probas_array = \
            uncalibrated_test_predictions_list[best_index].get_predictions()
        _, valid_y_probas_array = valid_predictions_list[best_index].get_predictions()
        test_y_probas_array = calibrate_test(
            valid_y_probas_array, ground_truth_valid, 
            uncalibrated_test_y_probas_array)
        test_y_pred_array = get_y_pred_array(test_y_probas_array)
        test_predictions = specific.prediction_type.PredictionArrayType(
            y_pred_array=test_y_pred_array, 
            y_probas_array=test_y_probas_array)
        test_predictions_list.append(test_predictions)
    return test_predictions_list

def get_calibrated_predictions_list(models_df, hash_string, ground_truth = []): 
    if len(ground_truth) == 0:
        ground_truth_filename = get_ground_truth_valid_f_name(hash_string)
        ground_truth = get_ground_truth(ground_truth_filename)

    uncalibrated_predictions_list = get_predictions_list(
        models_df, "valid", hash_string)

    logger.info("Evaluating : {}".format(hash_string))
    specific.score.set_eps(0.01 / len(ground_truth))

    # TODO: make calibration optional, output-type or score dependent
    # this doesn't work for RMSE
    #predictions_list = calibrate_predictions(
    #    uncalibrated_predictions_list, ground_truth)

    predictions_list = uncalibrated_predictions_list
    return predictions_list

# TODO: This is completely ouput_type dependent, so will probably go there.
def get_best_index_list(models_df, hash_string, selected_index_list=[]):
    if len(selected_index_list) == 0:
        selected_index_list = np.arange(len(models_df))
    ground_truth_valid_filename = get_ground_truth_valid_f_name(hash_string)
    ground_truth_valid = get_ground_truth(ground_truth_valid_filename)

    predictions_list = get_calibrated_predictions_list(
        models_df, hash_string, ground_truth_valid)
    predictions_list = [predictions_list[i] for i in selected_index_list]

    valid_scores = [specific.score(ground_truth_valid, predictions) 
                    for predictions in predictions_list]
    best_prediction_index = np.argmax(valid_scores)

    best_index_list = np.array([best_prediction_index])
    foldwise_best_predictions = predictions_list[best_prediction_index]

    improvement = True
    max_len_best_index_list = 80
    while improvement and len(best_index_list) < max_len_best_index_list:
        old_best_index_list = best_index_list
        best_index_list = best_combine(
            predictions_list, ground_truth_valid, best_index_list)
        improvement = len(best_index_list) != len(old_best_index_list)
    logger.info("best indices = {}".format(selected_index_list[best_index_list]))
    combined_predictions = specific.prediction_type.combine(
        predictions_list, best_index_list)
    return selected_index_list[best_index_list], foldwise_best_predictions, \
        combined_predictions, ground_truth_valid

def get_combined_test_predictions(models_df, hash_string, best_index_list):
    assert(len(best_index_list) > 0)
    logger.info("Evaluating combined test: {}".format(hash_string))
    ground_truth_valid_filename = get_ground_truth_valid_f_name(hash_string)
    ground_truth_valid = get_ground_truth(ground_truth_valid_filename)

    #Calibrating predictions
    uncalibrated_test_predictions_list = get_predictions_list(
        models_df, "test", hash_string)
    valid_predictions_list = get_predictions_list(
        models_df, "valid", hash_string)
    test_predictions_list = calibrate_test_predictions(
        uncalibrated_test_predictions_list, valid_predictions_list, 
        ground_truth_valid, best_index_list)

    combined_test_predictions = specific.prediction_type.combine(
        test_predictions_list, range(len(test_predictions_list)))
    foldwise_best_test_predictions = test_predictions_list[0]
    return combined_test_predictions, foldwise_best_test_predictions

#TODO: make an actual prediction
def make_combined_test_prediction(best_index_lists, models_df, hash_strings):
    if models_df.shape[0] != 0:
        ground_truth_test_filename = get_ground_truth_test_f_name()
        ground_truth_test = get_ground_truth(ground_truth_test_filename)

        list_of_tuples = Parallel(n_jobs=n_processes, verbose=0)\
            (delayed(get_combined_test_predictions)(
                         models_df, hash_string, best_index_list)
             for hash_string, best_index_list 
             in zip(hash_strings, best_index_lists))
        combined_test_predictions_list, foldwise_best_test_predictions_list = \
            zip(*list_of_tuples)

        #combined_test_predictions_list = []
        #foldwise_best_test_predictions_list = []
        #for hash_string, best_index_list in zip(hash_strings, best_index_lists):
        #    combined_test_predictions, foldwise_best_test_predictions = \
        #        get_combined_test_predictions(
        #            models_df, hash_string, best_index_list)
        #    combined_test_predictions_list.append(combined_test_predictions)
        #    # best in the fold
        #    foldwise_best_test_predictions_list.append(foldwise_best_test_predictions)

        combined_combined_test_predictions = specific.prediction_type.combine(
                combined_test_predictions_list)
        logger.info("foldwise combined test score = {}".format(specific.score(
                ground_truth_test, combined_combined_test_predictions)))
        combined_combined_test_predictions.save_predictions(
            os.path.join(private_data_path, "foldwise_combined.csv"))
        combined_foldwise_best_test_predictions = specific.prediction_type.combine(
                foldwise_best_test_predictions_list)
        combined_foldwise_best_test_predictions.save_predictions(
            os.path.join(private_data_path, "foldwise_best.csv"))
        logger.info("foldwise best test score = {}".format(specific.score(
            ground_truth_test, combined_foldwise_best_test_predictions)))

def leaderboard_combination_mean_of_scores(orig_models_df, test=False):
    models_df = orig_models_df.sort(columns='timestamp')
    hash_strings = get_hash_strings_from_ground_truth()
    counts = np.zeros(models_df.shape[0], dtype=int)

    if models_df.shape[0] != 0:
        random_index_lists = np.array([random.sample(
            range(len(models_df)), int(0.9*models_df.shape[0]))
            for _ in range(2)])
        #print random_index_lists

        list_of_tuples = Parallel(n_jobs=n_processes, verbose=0)\
            (delayed(get_best_index_list)(models_df, hash_string, random_index_list)
             for hash_string in hash_strings 
             for random_index_list in random_index_lists)
        best_index_lists, foldwise_best_predictions_list, \
            combined_predictions_list, ground_truth_valid_list = \
            zip(*list_of_tuples)

        foldwise_best_scores = [specific.score(ground_truth, foldwise_best_predictions_list)
                                for ground_truth, foldwise_best_predictions_list
                                in zip(ground_truth_valid_list, foldwise_best_predictions_list)]
        combined_scores = [specific.score(ground_truth, combined_predictions)
                           for ground_truth, combined_predictions
                           in zip(ground_truth_valid_list, combined_predictions_list)]

        logger.info("foldwise best validation score = {}".format(np.mean(foldwise_best_scores)))
        logger.info("foldwise combined validation score = {}".format(np.mean(combined_scores)))
        for best_index_list in best_index_lists:
            counts += np.histogram(best_index_list, 
                                   bins=range(models_df.shape[0] + 1))[0]
        
        #for hash_string in hash_strings:
        #    best_index_list, best_score, combined_score = get_best_index_list(models_df, hash_string)
        #    # adding 1 each time a model appears in best_indexes, with 
        #    # replacement (so counts[best_indexes] += 1 did not work)
        #    counts += np.histogram(
        #        best_index_list, bins=range(models_df.shape[0] + 1))[0]

        if test:
            make_combined_test_prediction(
                best_index_lists, models_df, hash_strings)

    leaderboard = pd.DataFrame({'contributivity': counts}, index=models_df.index)
    return leaderboard.sort(columns=['contributivity'], ascending=False)

def get_ground_truth_valid_list(hash_strings):
    return [get_ground_truth(get_ground_truth_valid_f_name(hash_string))
            for hash_string in hash_strings]

def get_Gabor_combined_mean_score(predictions_list, skf, hash_strings, 
                                  num_points, verbose=True):
    """Input is a list of predictions of a single model on a list of folds."""
    sum_y_pred_array = np.zeros(num_points, dtype=float)
    count_predictions = np.zeros(num_points, dtype=int)
    ground_truth_valid_list = get_ground_truth_valid_list(hash_strings)
    ground_truth_full = ['' for _ in range(num_points)]
    y_pred_array_list = np.array([predictions.get_predictions()
        for predictions in predictions_list])
    for (train_is, test_is), y_pred_array, hash_string, ground_truth_valid \
            in zip(skf, y_pred_array_list, hash_strings, ground_truth_valid_list):
        sum_y_pred_array[test_is] += y_pred_array
        count_predictions[test_is] += 1
        for test_i, i in zip(test_is, range(len(test_is))):
            ground_truth_full[test_i] = ground_truth_valid[i]
        mean_y_pred_array = sum_y_pred_array / count_predictions
        # we may not have any predictions for certain data points.
        valid_point_indexes = np.argwhere(count_predictions != 0).flatten()
        predictions = specific.prediction_type.PredictionArrayType(
            y_pred_array=mean_y_pred_array[valid_point_indexes])
        ground_truth_full_valid = [ground_truth_full[i] for i in valid_point_indexes]
        score = specific.score(ground_truth_full_valid, predictions)
        if verbose:
            print score
    if verbose:
        print "-------------"
    return score

# for multiclass this should be renamed get_Gabor_combined_mean_score
def get_Gabor_combined_mean_score_multiclass(predictions_list, skf, hash_strings, 
                                  num_points, verbose=True):
    """Input is a list of predictions of a single model on a list of folds."""
    # TODO: this has to be made more generic, now it works only for 
    # multiclass proba arrays
    sum_y_probas_array = np.zeros([num_points, len(specific.prediction_type.labels)], dtype=float)
    count_predictions = np.zeros([num_points, len(specific.prediction_type.labels)], dtype=int)
    ground_truth_valid_list = get_ground_truth_valid_list(hash_strings)
    ground_truth_full = ['' for _ in range(num_points)]
    y_probas_array_list = np.array([predictions.get_predictions()[1]
        for predictions in predictions_list])
    for (train_is, test_is), y_probas_array, hash_string, ground_truth_valid \
            in zip(skf, y_probas_array_list, hash_strings, ground_truth_valid_list):
        sum_y_probas_array[test_is] += y_probas_array
        count_predictions[test_is] += 1
        for test_i, i in zip(test_is, range(len(test_is))):
            ground_truth_full[test_i] = ground_truth_valid[i]
        mean_y_probas_array = sum_y_probas_array / count_predictions
        mean_y_pred_array = get_y_pred_array(mean_y_probas_array)
        # we may not have any predictions for certain data points.
        valid_point_indexes = np.argwhere(count_predictions[:,0] != 0).flatten()
        predictions = specific.prediction_type.PredictionArrayType(
            y_pred_array=mean_y_pred_array[valid_point_indexes], 
            y_probas_array=mean_y_probas_array[valid_point_indexes])
        ground_truth_full_valid = [ground_truth_full[i] for i in valid_point_indexes]
        score = specific.score(ground_truth_full_valid, predictions)
        if verbose:
            print score
    if verbose:
        print "-------------"
    return score

def leaderboard_classical(models_df, subdir = "valid", calibrate = False):
    mean_scores = []

    if models_df.shape[0] != 0:
        _, y_train_array, _, _, skf = specific.split_data()
        num_train = len(y_train_array)
        # we need the hash strings in the same order as train/test_is
        hash_strings = [get_hash_string_from_indices(train_is) 
                        for train_is, test_is in skf]
        predictions_lists = Parallel(n_jobs=n_processes, verbose=0)\
            (delayed(get_calibrated_predictions_list)(models_df, hash_string)
             for hash_string in hash_strings)

        # predictions_lists is a matrix of predictions, size
        # num_folds x num_models. We call mean_score per model, that is,
        # for every column
        mean_scores = [get_Gabor_combined_mean_score(
            predictions_list, skf, hash_strings, num_train)
            for predictions_list in zip(*predictions_lists)]

    logger.info("classical leaderboard mean {} scores = {}".
        format(subdir, mean_scores))
    leaderboard = pd.DataFrame({'score': mean_scores}, index=models_df.index)
    return leaderboard.sort(columns=['score'])


def leaderboard_combination(orig_models_df, test=False):
    models_df = orig_models_df.sort(columns='timestamp')
    counts = np.zeros(models_df.shape[0], dtype=int)

    if models_df.shape[0] != 0:
        _, y_train_array, _, _, skf = specific.split_data()
        num_train = len(y_train_array)
        num_bags = 1
        # One of Caruana's trick: bag the models
        #selected_index_lists = np.array([random.sample(
        #    range(len(models_df)), int(0.8*models_df.shape[0]))
        #    for _ in range(num_bags)])
        # Or you can select a subset
        #selected_index_lists = np.array([[24, 26, 28, 31]])
        # Or just take everybody
        selected_index_lists = np.array([range(len(models_df))])
        logger.info("Combining models {}".format(selected_index_lists))

        # we need the hash strings in the same order as train/test_is, can't
        # get them from the file names
        hash_strings = [get_hash_string_from_indices(train_is) 
                        for train_is, test_is in skf]
        list_of_tuples = Parallel(n_jobs=n_processes, verbose=0)\
            (delayed(get_best_index_list)(models_df, hash_string, selected_index_list)
             for hash_string in hash_strings 
             for selected_index_list in selected_index_lists)

        best_index_lists, foldwise_best_predictions_list, \
            combined_predictions_list, ground_truth_valid_list = \
            zip(*list_of_tuples)
        foldwise_best_scores = [specific.score(ground_truth, predictions)
                                for ground_truth, predictions
                                in zip(ground_truth_valid_list, foldwise_best_predictions_list)]
        combined_scores = [specific.score(ground_truth, predictions)
                           for ground_truth, predictions
                           in zip(ground_truth_valid_list, combined_predictions_list)]

        logger.info("Mean of scores")
        logger.info("foldwise best validation score = {}".format(np.mean(foldwise_best_scores)))
        logger.info("foldwise combined validation score = {}".format(np.mean(combined_scores)))
        # contributivity counts
        for best_index_list in best_index_lists:
            counts += np.histogram(best_index_list, 
                                   bins=range(models_df.shape[0] + 1))[0]

        combined_score = get_Gabor_combined_mean_score(
            list(combined_predictions_list), np.repeat(list(skf), num_bags, axis=0), 
            np.repeat(hash_strings, num_bags), num_train)
        foldwise_best_score = get_Gabor_combined_mean_score(
            list(foldwise_best_predictions_list), np.repeat(list(skf), num_bags, axis=0), 
            np.repeat(hash_strings, num_bags), num_train)
        logger.info("Score of \"means\" (Gabor's formula)")
        logger.info("foldwise best validation score = {}".format(foldwise_best_score))
        logger.info("foldwise combined validation score = {}".format(combined_score))
         
        #for hash_string in hash_strings:
        #    best_index_list, best_score, combined_score = get_best_index_list(models_df, hash_string)
        #    # adding 1 each time a model appears in best_indexes, with 
        #    # replacement (so counts[best_indexes] += 1 did not work)
        #    counts += np.histogram(
        #        best_index_list, bins=range(models_df.shape[0] + 1))[0]

        if test:
            make_combined_test_prediction(
                best_index_lists, models_df, hash_strings)

    leaderboard = pd.DataFrame({'contributivity': counts}, index=models_df.index)
    return leaderboard.sort(columns=['contributivity'], ascending=False)

