import os
import sys
import csv
import glob
import pickle
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
    np.savetxt(f_name_test, y_test, delimiter="\n", fmt='%d')

    logger.debug('Ground truth files...')
    scores = []
    for train_is, test_is in skf:
        hash_string = get_hash_string_from_indices(train_is)
        f_name_valid = get_ground_truth_valid_f_name(hash_string)
        logger.debug(f_name_valid)
        np.savetxt(f_name_valid, y_train[test_is], delimiter="\n", fmt='%d')

def test_trained_model(trained_model, X_test):
    start = timeit.default_timer()
    test_model_output = specific.test_model(trained_model, X_test)
    end = timeit.default_timer()
    return test_model_output, end - start

def train_on_fold(skf_is, full_model_path):
    valid_train_is, _ = skf_is
    X_train, y_train = data_sets.get_training_sets()
    hash_string = get_hash_string_from_indices(valid_train_is)
    # the current paradigm is that X is a list of things (not necessarily np.array)
    X_valid_train = [X_train[i] for i in valid_train_is]
    # but y should be an np.array just in case no one touches it in the fe
    y_valid_train = np.array([y_train[i] for i in valid_train_is])

    open(os.path.join(full_model_path, "__init__.py"), 'a').close()  # so to make it importable
    module_path = get_module_path(full_model_path)

    logger.info("Training : %s" % hash_string)
    start = timeit.default_timer()
    trained_model = specific.train_model(module_path, X_valid_train, y_valid_train)
    end = timeit.default_timer()
    with open(get_train_time_f_name(full_model_path, hash_string), 'w') as f:
        f.write(str(end - start)) # saving running time

    try:
        with open(get_model_f_name(full_model_path, hash_string), 'w') as f:
            pickle.dump(trained_model, f) # saving the model
    except pickle.PicklingError, e:
        logger.error("Cannot pickle trained model\n{}".format(e))
        os.remove(get_model_f_name(full_model_path, hash_string))


    # in case we want to train and test without going through pickling, for
    # example, because pickling doesn't work, we return the model
    return trained_model

def test_on_fold(skf_is, full_model_path):
    valid_train_is, _ = skf_is
    hash_string = get_hash_string_from_indices(valid_train_is)
    X_test, _ = data_sets.get_test_sets()

    logger.info("Testing : %s" % hash_string)
    try:
        logger.info("Loading from pickle")
        with open(get_model_f_name(full_model_path, hash_string), 'r') as f:
            trained_model = pickle.load(f)
    except IOError, e: # no pickled model, retrain
        logger.info("No pickle, retraining")
        trained_model = train_on_fold(skf_is, full_model_path)

    test_model_output, _ = test_trained_model(trained_model, X_test)

    with open(get_test_f_name(full_model_path, hash_string), 'w') as f:
        # saving test set predictions
        specific.save_model_predictions(test_model_output, f)

def train_and_valid_on_fold(skf_is, full_model_path):
    trained_model = train_on_fold(skf_is, full_model_path)

    valid_train_is, valid_test_is = skf_is
    X_train, y_train = data_sets.get_training_sets()
    hash_string = get_hash_string_from_indices(valid_train_is)
    X_valid_test = [X_train[i] for i in valid_test_is]
    y_valid_test = np.array([y_train[i] for i in valid_test_is])

    logger.info("Validating : %s" % hash_string)
    valid_model_output, test_time = test_trained_model(trained_model, X_valid_test)
    with open(get_valid_time_f_name(full_model_path, hash_string), 'w') as f:
        f.write(str(test_time))  # saving running time
    with open(get_valid_f_name(full_model_path, hash_string), 'w') as f:
        # saving validation set predictions
        specific.save_model_predictions(valid_model_output, f)

#@mem.cache
def train_and_valid_model(full_model_path, skf):
    #Uncomment this and comment out the follwing two lines if
    #parallel training is not working
    #for skf_is in skf:
    #    train_and_valid_on_fold(skf_is, full_model_path)
    
    Parallel(n_jobs=n_processes, verbose=5)\
        (delayed(train_and_valid_on_fold)(skf_is, full_model_path)
         for skf_is in skf)
    
    #partial_train = partial(train_and_valid_on_fold, full_model_path=full_model_path)
    #pool = multiprocessing.Pool(processes=n_processes)
    #pool.map(partial_train, skf)
    #pool.close()

    #import pprocess
    #import time
    #pprocess.pmap(partial_train, skf, limit=n_processes)

    #class MyExchange(pprocess.Exchange):

    #    "Parallel convenience class containing the array assignment operation."

    #    def store_data(self, ch):
    #        r = ch.receive()
    #        print "*****************************************************"
    #        print r

    #exchange = MyExchange(limit=n_processes)

    # Wrap the calculate function and manage it.

    #calc = exchange.manage(pprocess.MakeParallel(train_and_valid_on_fold))

    # Perform the work.
    #X_train, y_train = data_sets.get_training_sets()

    #for skf_is, i in zip(skf, range(n_processes)):
    #    calc(skf_is, full_model_path, X_train, y_train, i)
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

        logger.info("Training : %s/%s" % model_df['team'], model_df['tag'])

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

        logger.info("Testing : %s/%s" % model_df['team'], model_df['tag'])

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

def get_predictions_lists(models_df, subdir, hash_string):
    predictions_list = []
    for idx, model_df in models_df.iterrows():
        full_model_path = get_full_model_path(idx, model_df)
        predictions_path = get_f_name(full_model_path, subdir, hash_string)
        predictions = specific.load_model_predictions(predictions_path)
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

def leaderboard_classical(models_df, subdir = "valid"):
    hash_strings = get_hash_strings_from_ground_truth()
    mean_valid_scores = np.zeros(models_df.shape[0])

    if models_df.shape[0] != 0:
        for hash_string in hash_strings:
            if subdir == "valid":
                ground_truth_filename = get_ground_truth_valid_f_name(hash_string)
            else: #subdir = "test"
                ground_truth_filename = get_ground_truth_test_f_name()
            ground_truth = get_ground_truth(ground_truth_filename)
            predictions_lists = get_predictions_lists(
                models_df, subdir, hash_string)
            valid_scores = [specific.Score().score(ground_truth, predictions_list) 
                            for predictions_list in predictions_lists]
            mean_valid_scores += valid_scores

    mean_valid_scores /= len(hash_strings)
    logger.info("classical leaderboard mean valid scores = {}".
        format(mean_valid_scores))
    leaderboard = pd.DataFrame({'score': mean_valid_scores}, 
                               index=models_df.index)
    return leaderboard.sort(
        columns=['score'], ascending=not specific.Score().higher_the_better)



def better_score(score1, score2, eps):
    if specific.Score().higher_the_better:
        return score1 > score2 + eps
    else:
        return score1 < score2 - eps

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
    best_predictions = combine_models_using_probas(predictions_list, best_indexes)
    best_index = -1
    # FIXME: I don't remember why we need eps, but without it the results are
    # very different. In any case, the Score class should take care of its eps 
    eps = 0.01/len(ground_truth)
    # Combination with replacement, what Caruana suggests. Basically, if a model
    # added several times, it's upweighted.
    for i in range(len(predictions_list)):
        combined_predictions = combine_models_using_probas(
            predictions_list, np.append(best_indexes, i))
        if better_score(specific.Score().score(ground_truth, combined_predictions), 
                        specific.Score().score(ground_truth, best_predictions),
                        eps):
            best_predictions = combined_predictions
            best_index = i
    if best_index > -1:
        return np.append(best_indexes, best_index)
    else:
        return best_indexes

# FIXME: This is really dependent on the output_type and the score at the same 
# time
def combine_models_using_probas(predictions_list, indexes):
    """Combines the predictions y_preds[indexes] by "proba"
    voting. I'll detail it once you verify that it makes sense (see my mail)

    Parameters
    ----------
    y_preds : array-like, shape = [k_models, n_instances], binary
    y_probas : array-like, shape = [k_models, n_instances], permutation of [0,...,n_instances]
    indexes : array-like, shape = [max k_models], a set of indices of 
        models to combine
    Returns
    -------
    com_y_pred : array-like, shape = [n_instances], a list of (combined) 
        binary predictions.
    """
    #k = len(indexes)
    #n = len(y_preds[0])
    #n_ones = n * k - y_preds[indexes].sum() # number of zeros
    y_probas = np.array([[predictions_list[i][j][1] 
                 for j in range(len(predictions_list[i]))
                ]
                for i in indexes
               ])
    # We do mean probas because sum(log(probas)) have problems at zero
    means_y_probas = y_probas.mean(axis=0)
    # don't really have to convert it back to list, just to stay consistent
    predictions = [[specific.labels[y_proba.argmax()], y_proba.tolist()]
                   for y_proba in means_y_probas]
    #print predictions
    return predictions

from sklearn.isotonic import IsotonicRegression
from sklearn.cross_validation import StratifiedShuffleSplit

class IsotonicCalibrator():
    def __init__(self):
        pass
    
    def fit(self, X, y):
        labels = np.sort(np.unique(y))
        self.calibrators = []
        for class_index in range(X.shape[1]):
            calibrator = IsotonicRegression(
                y_min=0., y_max=1., out_of_bounds='clip')
            class_indicator = np.zeros(X.shape[0])
            class_indicator[y == labels[class_index]] = 1
            calibrator.fit(X[:,class_index], class_indicator)
            self.calibrators.append(calibrator)
            
    def predict(self, X):
        X_calibrated_transpose = np.array(
            [self.calibrators[class_index].predict(X[:,class_index])
             for class_index in range(X.shape[1])])
        sum_rows = np.sum(X_calibrated_transpose, axis=0)
        X_calibrated_normalized_transpose = np.divide(
            X_calibrated_transpose, sum_rows)
        return X_calibrated_normalized_transpose.T

def calibrate_test(predict_proba_array, ground_truth, test_predict_proba_array):
    isotonic_calibrator = IsotonicCalibrator()
    isotonic_calibrator.fit(predict_proba_array, ground_truth)
    calibrated_test_predict_proba_array = isotonic_calibrator.predict(
        test_predict_proba_array)
    return calibrated_test_predict_proba_array

def calibrate(predict_proba_array, ground_truth, test_predict_proba_array = []):
    skf = StratifiedShuffleSplit(ground_truth, n_iter=1, test_size=0.5, 
                                 random_state=specific.random_state)
    calibrated_proba_array = np.empty(predict_proba_array.shape)
    fold1_is, fold2_is = list(skf)[0]
    folds = [(fold1_is, fold2_is), (fold2_is, fold1_is)]
    isotonic_calibrator = IsotonicCalibrator()
    for fold_train_is, fold_test_is in folds:
        isotonic_calibrator.fit(predict_proba_array[fold_train_is], 
                                ground_truth[fold_train_is])
        calibrated_proba_array[fold_test_is] = isotonic_calibrator.predict(
            predict_proba_array[fold_test_is])
    return calibrated_proba_array

# It's a bit messy now. Calibration takes the proba array whereas 
# predictions_list also contains the label. In any case, this is completely
# ouput_type dependent, so will probably go there.
def get_best_index_list(models_df, hash_string):
    ground_truth_filename = get_ground_truth_valid_f_name(hash_string)
    ground_truth = get_ground_truth(ground_truth_filename)

    predictions_lists_orig = get_predictions_lists(models_df, "valid", hash_string)
    predictions_lists = []
    for predictions_list_orig in predictions_lists_orig:
        # getting rid of the predicted labels predictions_list[:,0]
        predict_proba_list = [
            predictions[1] for predictions in predictions_list_orig]
        predict_proba_array = np.array(predict_proba_list)
        calibrated_proba_array = calibrate(predict_proba_array, ground_truth)
        predictions_list = [[predictions[0], calibrated_proba_list]
            for predictions, calibrated_proba_list
            in zip(predictions_list_orig, calibrated_proba_array.tolist())]
        predictions_lists.append(predictions_list)

    valid_scores = [specific.Score().score(ground_truth, predictions_list) 
                    for predictions_list in predictions_lists]
    best_index_list = np.array([np.argmax(valid_scores)])

    improvement = True
    while improvement:
        old_best_index_list = best_index_list
        best_index_list = best_combine(
            predictions_lists, ground_truth, best_index_list)
        improvement = len(best_index_list) != len(old_best_index_list)
    logger.info("best indices = {}".format(best_index_list))
    best_score = np.max(valid_scores)
    combined_predictions = combine_models_using_probas(
        predictions_lists, best_index_list)
    combined_score = specific.Score().score(ground_truth, combined_predictions)
    return best_index_list, best_score, combined_score

#TODO: fix foldwise best
#TODO: make an actual prediction
#TODO: parallelize
def make_combined_test_prediction(best_index_lists, models_df, hash_strings):
    # We shouod use it to calibrate only the needed models, only once
    unique_best_index_list = np.unique([best_index for best_index_list in best_index_lists
                                 for best_index in best_index_list])

    if models_df.shape[0] != 0:
        ground_truth_test_filename = get_ground_truth_test_f_name()
        ground_truth_test = get_ground_truth(ground_truth_test_filename)
        combined_test_predictions_list = []
        for hash_string, best_index_list in zip(hash_strings, best_index_lists):
            print hash_string
            ground_truth_valid_filename = get_ground_truth_valid_f_name(hash_string)
            ground_truth_valid = get_ground_truth(ground_truth_valid_filename)
            test_predictions_lists_orig = get_predictions_lists(
                models_df, "test", hash_string)
            valid_predictions_lists_orig = get_predictions_lists(
                models_df, "valid", hash_string)
            test_predictions_lists = []
            for best_index in best_index_list:
                test_predictions_list_orig = test_predictions_lists_orig[best_index]
                valid_predictions_list_orig = valid_predictions_lists_orig[best_index]

                valid_predict_proba_list = [
                    predictions[1] for predictions in valid_predictions_list_orig]
                valid_predict_proba_array = np.array(valid_predict_proba_list)

                test_predict_proba_list = [
                    predictions[1] for predictions in test_predictions_list_orig]
                test_predict_proba_array = np.array(test_predict_proba_list)

                calibrated_test_predict_proba_array = calibrate_test(
                    valid_predict_proba_array, ground_truth_valid, 
                    test_predict_proba_array)
                test_predictions_list = [[predictions[0], calibrated_proba_list]
                    for predictions, calibrated_proba_list
                    in zip(test_predictions_list_orig, 
                        calibrated_test_predict_proba_array.tolist())]
                test_predictions_lists.append(test_predictions_list)

            combined_test_predictions = combine_models_using_probas(
                test_predictions_lists, range(len(test_predictions_lists)))
            combined_test_predictions_list.append(combined_test_predictions)
            #foldwise_best_test_predictions = \
            #    predictions_lists['test'][best_indexes[0]]
            #foldwise_best_test_predictions_list.append(foldwise_best_test_predictions)


        combined_combined_test_predictions = combine_models_using_probas(
                combined_test_predictions_list, 
                range(len(combined_test_predictions_list)))
        print "foldwise combined test score = ", \
            specific.Score().score(
                ground_truth_test, combined_combined_test_predictions)
        #combined_foldwise_best_test_predictions = combine_models_using_probas(
        #        foldwise_best_test_predictions_list, 
        #        range(len(foldwise_best_test_predictions_list)))
        #print "foldwise best test score = ", \
        #    specific.Score().score(ground_truth_test, combined_foldwise_best_test_predictions)

def leaderboard_combination(orig_models_df, test=False):
    models_df = orig_models_df.sort(columns='timestamp')
    hash_strings = get_hash_strings_from_ground_truth()
    counts = np.zeros(models_df.shape[0], dtype=int)

    if models_df.shape[0] != 0:
        list_of_tuples = Parallel(n_jobs=n_processes, verbose=0)\
            (delayed(get_best_index_list)(models_df, hash_string)
             for hash_string in hash_strings)
        best_index_lists, best_scores, combined_scores = zip(*list_of_tuples)
        logger.info("foldwise best validation score = %.4f" % np.mean(best_scores))
        logger.info("foldwise combined validation score = %.4f" % np.mean(combined_scores))
        for best_index_list in best_index_lists:
            counts += np.histogram(best_index_list, 
                                   bins=range(models_df.shape[0] + 1))[0]
        if test:
            make_combined_test_prediction(
                best_index_lists, models_df, hash_strings)
        #for hash_string in hash_strings:
        #    best_index_list = get_best_index_list(models_df, hash_string)
        #    # adding 1 each time a model appears in best_indexes, with 
        #    # replacement (so counts[best_indexes] += 1 did not work)
        #    counts += np.histogram(
        #        best_index_list, bins=range(models_df.shape[0] + 1))[0]

    leaderboard = pd.DataFrame({'contributivity': counts}, index=models_df.index)
    return leaderboard.sort(columns=['contributivity'], ascending=False)

