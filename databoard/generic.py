import os
import sys
import csv
import glob
import pickle
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
    <prefix>_<hash_string>.<extension>

    Parameters
    ----------
    path : string with the file name after the last '/'

    Returns
    -------
    hash_string
    """
    return path.split('/')[-1].split('.')[-2].split('_')[-1]

def get_module_path(m_path):
    return m_path.lstrip('./').replace('/', '.')

def get_model_f_name(m_path, hash_string):
    return os.path.join(m_path, "model_" + hash_string + ".p")

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
    ground_truth_path = os.path.join(root_path, 'ground_truth')
    os.rmdir(ground_truth_path)  # cleanup the ground_truth
    os.mkdir(ground_truth_path)
    _, y_train, _, y_test, skf = specific.split_data()
    f_name_test = ground_truth_path + "/ground_truth_test.csv"
    np.savetxt(f_name_test, y_test, delimiter="\n", fmt='%d')

    logger.debug('Ground truth files...')
    scores = []
    for train_is, test_is in skf:
        hash_string = get_hash_string_from_indices(train_is)
        f_name_valid = os.path.join(
            ground_truth_path, "ground_truth_valid_" + hash_string + ".csv")
        logger.debug(f_name_valid)
        np.savetxt(f_name_valid, y_train[test_is], delimiter="\n", fmt='%d')

def train_and_valid_on_fold(skf_is, m_path):
    valid_train_is, valid_test_is = skf_is
    X_train, y_train = data_sets.get_training_sets()
    hash_string = get_hash_string_from_indices(valid_train_is)
    # the current paradigm is that X is a list of things (not necessarily np.array)
    X_valid_train = [X_train[i] for i in valid_train_is]
    y_valid_train = [y_train[i] for i in valid_train_is]
    X_valid_test = [X_train[i] for i in valid_test_is]
    y_valid_test = [y_train[i] for i in valid_test_is]

    open(os.path.join(m_path, "__init__.py"), 'a').close()  # so to make it importable
    module_path = get_module_path(m_path)

    logger.info("Training : %s" % hash_string)
    trained_model = specific.train_model(module_path, X_valid_train, y_valid_train)
    with open(get_model_f_name(m_path, hash_string), 'w') as f:
        pickle.dump(trained_model, f)

    logger.info("Validating : %s" % hash_string)
    valid_model_output = specific.test_model(trained_model, X_valid_test)
    f_name_valid = os.path.join(m_path, "valid_" + hash_string + ".csv")
    specific.save_model_predictions(valid_model_output, f_name_valid)

def test_on_fold(skf_is, m_path):
    valid_train_is, valid_test_is = skf_is
    X_test, y_test = data_sets.get_test_sets()
    hash_string = get_hash_string_from_indices(valid_train_is)
    f_name_test = m_path + "/test_" + hash_string + ".csv"
    logger.info("Testing : %s" % hash_string)

    open(os.path.join(m_path, '/__init__.py'), 'a').close()  # so to make it importable
    module_path = get_module_path(m_path)

    with open(get_model_f_name(m_path, hash_string), 'r') as f:
        trained_model = pickle.load(f)
    test_model_output = specific.test_model(trained_model, X_test)
    specific.save_model_predictions(test_model_output, f_name_test)

def train_and_valid_models(models, last_time_stamp=None):

    models_sorted = models.sort("timestamp")
        
    if len(models_sorted) == 0:
        logger.info("No models to train.")
        return

    logger.info("Reading data")
    X_train, y_train, X_test, y_test, skf = specific.split_data()
    data_sets.set_sets(X_train, y_train, X_test, y_test)

    for idx, m in models_sorted.iterrows():

        team = m['team']
        model = m['model']
        timestamp = m['timestamp']
        path = m['path']
        m_path = os.path.join(models_path, path)

        logger.info("Training : %s" % m_path)

        try:
            train_and_valid_model(m_path, skf)
            # failed_models.drop(idx, axis=0, inplace=True)
            models.loc[idx, 'state'] = "trained"
        except Exception, e:
            models.loc[idx, 'state'] = "error"
            # trained_models.drop(idx, axis=0, inplace=True)
            logger.error("Training failed with exception: \n{}".format(e))

            # TODO: put the error in the database instead of a file
            # Keep the model folder clean.
            with open(os.path.join(m_path, 'error.txt'), 'w') as f:
                error_msg = str(e)
                cut_exception_text = error_msg.rfind('--->')
                if cut_exception_text > 0:
                    error_msg = error_msg[cut_exception_text:]
                f.write("{}".format(error_msg))


@mem.cache
def train_and_valid_model(m_path, skf):
    """Train a model on all folds and save the valid predictions and the 
    model pickle. All model file names use the format
    <hash of the np.array(train index vector)>

    Parameters
    ----------
    m_paths : list, shape (k_models,)
    skf : object a cross_validation object with N folds
    """

    #Uncomment this and comment out the follwing two lines if
    #parallel training is not working
    #for skf_is in skf:
    #    train_and_valid_on_fold(skf_is, m_path)
    
    Parallel(n_jobs=n_processes, verbose=5)\
        (delayed(train_and_valid_on_fold)(skf_is, m_path)
         for skf_is in skf)
    
    #partial_train = partial(train_and_valid_on_fold, m_path=m_path)
    #pool = multiprocessing.Pool(processes=n_processes)
    #pool.map(partial_train, skf)
    #pool.close()

def test_models(models, last_time_stamp=None):

    models_sorted = models.sort("timestamp")
        
    if len(models_sorted) == 0:
        logger.info("No models to test.")
        return

    logger.info("Reading data")
    X_train, y_train, X_test, y_test, skf = specific.split_data()
    data_sets.set_sets(X_train, y_train, X_test, y_test)

    for idx, m in models_sorted.iterrows():

        team = m['team']
        model = m['model']
        timestamp = m['timestamp']
        path = m['path']
        m_path = os.path.join(models_path, path)

        logger.info("Testing : %s" % m_path)

        if models.loc[idx, 'state'] not in ["trained", "test_error"]:
            logger.error("Only trained models can be tested")
            logger.error(m_path + "is untrained")
            return

        try:
            test_model(m_path, skf)
            # failed_models.drop(idx, axis=0, inplace=True)
            models.loc[idx, 'state'] = "tested"
        except Exception, e:
            models.loc[idx, 'state'] = "test_error"
            # trained_models.drop(idx, axis=0, inplace=True)
            logger.error("Testing failed with exception: \n{}".format(e))

            # TODO: put the error in the database instead of a file
            # Keep the model folder clean.
            with open(os.path.join(m_path, 'test_error.txt'), 'w') as f:
                error_msg = str(e)
                cut_exception_text = error_msg.rfind('--->')
                if cut_exception_text > 0:
                    error_msg = error_msg[cut_exception_text:]
                f.write("{}".format(error_msg))

@mem.cache
def test_model(m_path, skf):
    """Test a model on all folds and save the test predictions. All model file 
    names use the format
    <hash of the np.array(train index vector)>

    Parameters
    ----------
    m_paths : list, shape (k_models,)
    """

    Parallel(n_jobs=n_processes, verbose=5)\
        (delayed(test_on_fold)(skf_is, m_path)
         for skf_is in skf)

def get_predictions_lists(model_paths, hash_string):
    predictions_lists = {'valid' : [], 'test' : []}
    for model_path in model_paths:
        for test_set in ['valid', 'test']:
            predictions_path = os.path.join(
                root_path, "models", model_path, 
                test_set + "_" + hash_string + ".csv")
            predictions = specific.load_model_predictions(predictions_path)
            predictions_lists[test_set].append(predictions)
    return predictions_lists

def get_predictions_lists_no_test(model_paths, hash_string):
    predictions_lists = {'valid' : []}
    for model_path in model_paths:
        for test_set in ['valid']:
            predictions_path = os.path.join(
                root_path, "models", model_path, 
                test_set + "_" + hash_string + ".csv")
            predictions = specific.load_model_predictions(predictions_path)
            predictions_lists[test_set].append(predictions)
    return predictions_lists

def leaderboard_classical(ground_truth_path, orig_models):
    models = orig_models.sort(columns='timestamp')
    models_paths = [os.path.join(models_path, path)
                    for path in models['path']]
    ground_truth_filanames = glob.glob(ground_truth_path + "/ground_truth_valid*")
    hash_strings = [get_hash_string_from_path(path) 
                    for path in ground_truth_filanames]
    mean_valid_scores = np.zeros(len(models_paths))

    if models.shape[0] != 0:
        for hash_string in hash_strings:
            ground_truth_filename = os.path.join(
                ground_truth_path, "ground_truth_valid_" + hash_string + ".csv")
            ground_truth = pd.read_csv(ground_truth_filename, 
                names=['ground_truth']).values.flatten()
            predictions_lists = get_predictions_lists_no_test(
                models['path'], hash_string)
            valid_scores = [specific.Score().score(ground_truth, predictions) 
                            for predictions in predictions_lists['valid']]
            mean_valid_scores += valid_scores

    # TODO: add a score column to the models df

    mean_valid_scores /= len(hash_strings)
    logger.info("classical leaderboard mean valid scores = {}".
        format(mean_valid_scores))
    leaderboard = pd.DataFrame({'score': mean_valid_scores}, index=models.index)
    return leaderboard.sort(
        columns=['score'], ascending=not specific.Score().higher_the_better)


def leaderboard_classical_with_test(ground_truth_path, orig_models):
    """Output classical leaderboard (sorted in increasing order by score).

    Parameters
    ----------
    m_paths : list, shape (k_models,)
        A list of paths, each containing a "score.csv" file with three columns:
            1) the file prefix (hash of the test index set the model was tested on)
            2) the number of test points
            3) the test score (the lower the better; error)

    Returns
    -------
    leaderboard : DataFrame
        a pandas DataFrame with two columns:
            1) The model name (the name of the subdir)
            2) the mean score
        example:
            model     error
        0   Kegl9  0.232260
        1  Kegl10  0.232652
        2   Kegl7  0.234174
        3   Kegl1  0.237330
        4   Kegl2  0.238945
        5   Kegl4  0.242344
        6   Kegl5  0.243212
        7   Kegl6  0.243380
        8   Kegl8  0.244015
        9   Kegl3  0.251158

    """

    # FIXME:
    # we should not modify the original models df by 
    # sorting it or explicitly modifying its index

    models = orig_models.sort(columns='timestamp')
    models_paths = [os.path.join(models_path, path)
                    for path in models['path']]
    ground_truth_filanames = glob.glob(ground_truth_path + "/ground_truth_valid*")
    ground_truth_test = pd.read_csv(
        os.path.join(ground_truth_path, "ground_truth_test.csv"),
        names=['ground_truth']).values.flatten()
    hash_strings = [get_hash_string_from_path(path)
                    for path in ground_truth_filanames]
    mean_valid_scores = np.zeros(len(models_paths))
    mean_test_scores = np.zeros(len(models_paths))

    if models.shape[0] != 0:
        for hash_string in hash_strings:
            ground_truth_filename = os.path.join(
                ground_truth_path, "ground_truth_valid_" + hash_string + ".csv")
            ground_truth = pd.read_csv(ground_truth_filename, 
                names=['ground_truth']).values.flatten()
            predictions_lists = get_predictions_lists(
                models['path'], hash_string)
            valid_scores = [specific.Score().score(ground_truth, predictions) 
                            for predictions in predictions_lists['valid']]
            mean_valid_scores += valid_scores
            test_scores = [specific.Score().score(ground_truth_test, predictions) 
                           for predictions in predictions_lists['test']]
            mean_test_scores += test_scores

    # TODO: add a score column to the models df

    mean_valid_scores /= len(hash_strings)
    logger.info("classical leaderboard mean valid scores = {}".
        format(mean_valid_scores))
    mean_test_scores /= len(hash_strings)
    logger.info("classical leaderboard mean test scores = {}".
        format(mean_test_scores))
    leaderboard = pd.DataFrame({'score': mean_valid_scores}, index=models.index)
    return leaderboard.sort(
        columns=['score'], ascending=not specific.Score().higher_the_better)

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

def leaderboard_combination(ground_truth_path, orig_models):
    models = orig_models.sort(columns='timestamp')
    counts = np.zeros(len(models), dtype=int)
    if models.shape[0] != 0:
        models_paths = [os.path.join(models_path, path)
                        for path in models['path']]
        ground_truth_filanames = glob.glob(ground_truth_path + "/ground_truth_valid*")
        hash_strings = [get_hash_string_from_path(path) 
                        for path in ground_truth_filanames]

        for hash_string in hash_strings:
            ground_truth_filename = os.path.join(
                ground_truth_path, "ground_truth_valid_" + hash_string + ".csv")
            ground_truth = pd.read_csv(ground_truth_filename, 
                names=['ground_truth']).values.flatten()
            predictions_lists = get_predictions_lists_no_test(
                models['path'], hash_string)
            valid_scores = [specific.Score().score(ground_truth, predictions) 
                            for predictions in predictions_lists['valid']]
            best_indexes = np.array([np.argmax(valid_scores)])

            improvement = True
            while improvement:
                old_best_indexes = best_indexes
                best_indexes = best_combine(
                    predictions_lists['valid'], ground_truth, best_indexes)
                improvement = len(best_indexes) != len(old_best_indexes)
            logger.info("best indices = {}".format(best_indexes))
            # adding 1 each time a model appears in best_indexes, with 
            # replacement (so counts[best_indexes] += 1 did not work)
            counts += np.histogram(best_indexes, bins = range(len(models) + 1))[0]

    # leaderboard = models.copy()
    leaderboard = pd.DataFrame({'contributivity': counts}, index=models.index)
    return leaderboard.sort(columns=['contributivity'],  ascending=False)


def leaderboard_combination_with_test(ground_truth_path, orig_models):
    """Output combined leaderboard (sorted in decreasing order by score). We use
    Caruana's greedy combination
    http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf
    on each fold, and cound how many times each model is chosen.

    Parameters
    ----------
    groundtruth_paths : ground truth path
    m_paths : array-like, shape = [k_models]
        A list of paths, each containing len(skf) csv files with y_test.
    Returns
    -------
    leaderboard : a pandas DataFrame with two columns:
            1) The model name (the name of the subdir)
            2) the count score
        example:
         model  count
        0  Kegl10     45
        2   Kegl9     38
        3   Kegl1     26
        4   Kegl2     13
        5   Kegl6     13
        6   Kegl5     12
        7   Kegl8     10
        8   Kegl4      4
        9   Kegl3      3

    """

    models = orig_models.sort(columns='timestamp')
    counts = np.zeros(len(models), dtype=int)
    if models.shape[0] != 0:
        models_paths = [os.path.join(models_path, path)
                        for path in models['path']]
        ground_truth_filanames = glob.glob(ground_truth_path + "/ground_truth_valid*")
        hash_strings = [get_hash_string_from_path(path) 
                        for path in ground_truth_filanames]
        ground_truth_test = pd.read_csv(
            os.path.join(ground_truth_path, "ground_truth_test.csv"),
            names=['ground_truth']).values.flatten()

        combined_test_predictions_list = []
        foldwise_best_test_predictions_list = []
        for hash_string in hash_strings:
            ground_truth_filename = os.path.join(
                ground_truth_path, "ground_truth_valid_" + hash_string + ".csv")
            ground_truth = pd.read_csv(ground_truth_filename, 
                names=['ground_truth']).values.flatten()
            predictions_lists = get_predictions_lists(
                models['path'], hash_string)
            valid_scores = [specific.Score().score(ground_truth, predictions) 
                            for predictions in predictions_lists['valid']]
            best_indexes = np.array([np.argmax(valid_scores)])

            improvement = True
            while improvement:
                old_best_indexes = best_indexes
                best_indexes = best_combine(
                    predictions_lists['valid'], ground_truth, best_indexes)
                improvement = len(best_indexes) != len(old_best_indexes)
            logger.info("best indices = {}".format(best_indexes))
            # adding 1 each time a model appears in best_indexes, with 
            # replacement (so counts[best_indexes] += 1 did not work)
            counts += np.histogram(best_indexes, bins = range(len(models) + 1))[0]

            combined_test_predictions = combine_models_using_probas(
                predictions_lists['test'], best_indexes)
            combined_test_predictions_list.append(combined_test_predictions)
            foldwise_best_test_predictions = \
                predictions_lists['test'][best_indexes[0]]
            foldwise_best_test_predictions_list.append(foldwise_best_test_predictions)


        combined_combined_test_predictions = combine_models_using_probas(
                combined_test_predictions_list, 
                range(len(combined_test_predictions_list)))
        print "foldwise combined test score = ", \
            specific.Score().score(ground_truth_test, combined_combined_test_predictions)
        combined_foldwise_best_test_predictions = combine_models_using_probas(
                foldwise_best_test_predictions_list, 
                range(len(foldwise_best_test_predictions_list)))
        print "foldwise best test score = ", \
            specific.Score().score(ground_truth_test, combined_foldwise_best_test_predictions)

    # leaderboard = models.copy()
    leaderboard = pd.DataFrame({'contributivity': counts}, index=models.index)
    return leaderboard.sort(columns=['contributivity'],  ascending=False)

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
