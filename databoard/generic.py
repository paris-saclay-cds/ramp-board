import os
import sys
import csv
import glob
import hashlib
import logging
import multiprocessing
import numpy as np
import pandas as pd
from scipy import io
from functools import partial
from importlib import import_module
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
from .specific import (
    test_filename, 
    read_data, 
    split_data, 
    run_model, 
    Score,
)

sys.path.append(os.path.dirname(os.path.abspath(models_path)))

mem = Memory(cachedir=cachedir)
logger = logging.getLogger('databoard')

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
    """Setting up the GroundTruth subdir, saving y_test for each fold in skf. File
    names are pred_<hash of the test index vector>.csv.

    Parameters
    ----------
    gt_paths : ground truth path
    y : array-like, shape = [n_instances]
        the label vector
    """
    gt_path = os.path.join(root_path, 'ground_truth')
    os.rmdir(gt_path)  # cleanup the ground_truth
    os.mkdir(gt_path)
    _, y_train, _, _, skf = split_data()

    logger.debug('Ground truth files...')
    scores = []
    for train_is, test_is in skf:
        hasher = hashlib.md5()
        hasher.update(test_is)
        h_str = hasher.hexdigest()
        f_name_pred = gt_path + "/pred_" + h_str + ".csv"
        logger.debug(f_name_pred)
        np.savetxt(f_name_pred, y_train[test_is], delimiter="\n", fmt='%d')


def save_scores(skf_is, m_path, X_train, y_train, X_test, y_test, f_name_score):
    valid_train_is, valid_test_is = skf_is
    hasher = hashlib.md5()
    hasher.update(valid_test_is)
    h_str = hasher.hexdigest()
    f_name_pred = m_path + "/pred_" + h_str + ".csv"
    f_name_test = m_path + "/test_" + h_str + ".csv"
    X_valid_train = X_train[valid_train_is]
    y_valid_train = y_train[valid_train_is]
    X_valid_test = X_train[valid_test_is]
    y_valid_test = y_train[valid_test_is]

    open(m_path + "/__init__.py", 'a').close()  # so to make it importable
    module_path = '.'.join(m_path.lstrip('./').split('/'))
    model = import_module('.model', module_path)

    y_valid_pred, y_valid_proba, y_test_pred, y_test_proba = run_model(model, 
        X_valid_train, y_valid_train, X_valid_test, X_test)

    assert len(y_valid_pred) == len(y_valid_proba) == len(X_valid_test)
    assert len(y_test_pred) == len(y_test_proba) == len(X_test)
    
    # y_proba[i] is the the proba of the ith element of y_proba
    #y_valid_proba = y_valid_proba[:,1].argsort().argsort()
    #y_test_proba = y_test_proba[:,1].argsort().argsort()
    #output_valid = np.transpose(np.array([y_valid_pred, y_valid_proba]))
    output_valid = np.transpose(np.array([y_valid_pred, y_valid_proba[:,1]]))
    np.savetxt(f_name_pred, output_valid, fmt='%d,%lf')
    output_test = np.transpose(np.array([y_test_pred, y_test_proba[:,1]]))
    np.savetxt(f_name_test, output_test, fmt='%d,%lf')
    print f_name_pred


def train_models(models, last_time_stamp=None):
    models_sorted = models[models['state'] == 'new'].sort("timestamp")

    # FIXME: should not modify the index like this
    # models_sorted.index = range(1, len(models_sorted) + 1)

    X_train, y_train, X_test, y_test, skf = split_data()

    # models_sorted = models_sorted[models_sorted.index < 50]  # XXX to make things fast

    for idx, m in models_sorted.iterrows():

        team = m['team']
        model = m['model']
        timestamp = m['timestamp']
        path = m['path']
        m_path = os.path.join(root_path, 'models', path)

        logger.info("Training : %s" % m_path)

        try:
            train_model(m_path, X_train, y_train, X_test, y_test, skf)
            # failed_models.drop(idx, axis=0, inplace=True)
            models.loc[idx, 'state'] = "trained"
        except Exception, e:
            models.loc[idx, 'state'] = "error"
            # trained_models.drop(idx, axis=0, inplace=True)
            logger.error("Training failed with exception: \n{}".format(e))

            # TODO: put the error in the database instead of a file
            # Keep the model folder clean.
            with open(os.path.join(m_path, 'error.txt'), 'w') as f:
                cut_exception_text = str(e).find(path)
                if cut_exception_text > 0:
                    a = e[cut_exception_text:]
                f.write("{}".format(e))


@mem.cache
def train_model(m_path, X_train, y_train, X_test, y_test, skf):
    """Training a model on all folds and saving the predictions and proba order. The latter we can
    use for computing ROC or cutting ties.

    m_path/model.py 
    should contain the model function, for example

    def model(X_train, y_train, X_test):
    clf = Pipeline([('imputer', Imputer(strategy='most_frequent')),
        ('rf', AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=5, n_estimators=100),
                         n_estimators=20))])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    return y_pred, y_proba

    File names are pred_<hash of the np.array(test index vector)>.csv. A score.csv file is also
    saved with k_model lines of header
    <hash of the test index vector>, <number of test instances>, <error>
    evaluations are parallel, so the order in score.csv is undefined.

    Parameters
    ----------
    m_paths : list, shape (k_models,)
    X : array-like, shape (n_instances, d_features)
    y : array-like, shape (n_instances,)
        the label vector
    skf : object
        a cross_validation object with n_folds
    """
    print m_path

    f_name_score = m_path + "/score.csv"
    scores = []
    open(f_name_score, "w").close()

    Parallel(n_jobs=n_processes)(delayed(save_scores)
        (skf_is, m_path, X_train, y_train, X_test, y_test, f_name_score) for skf_is in skf)
    
    # partial_save_scores = partial(save_scores, m_path=m_path, X=X, y=y, f_name_score=f_name_score)
    # pool = multiprocessing.Pool(processes=n_processes)
    # pool.map(partial_save_scores, skf)
    # pool.close()

def leaderboard_classical(groundtruth_path, orig_models):
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
    models_paths = [os.path.join(root_path, 'models', path) for path in models['path']]
    pred_paths = glob.glob(groundtruth_path + "/pred_*")
    pred_paths = np.array([pred_path.split('/')[-1] for pred_path in pred_paths])

    print pred_paths

    mean_scores = np.zeros(len(models_paths))

    if models.shape[0] != 0:
        for pred_path in pred_paths:
            y_preds = []
            y_probas = []
            y_test = pd.read_csv(groundtruth_path + '/' + pred_path, names=['pred']).values.flatten()
            for model_path in models['path']:
                predictions_path = os.path.join(root_path, 'models', model_path, pred_path)
                predictions = pd.read_csv(predictions_path, names=['pred', 'proba'])
                y_preds.append(predictions['pred'].values) # use m_path as db key
                y_probas.append(predictions['proba'].values)

            y_preds = np.array(y_preds)
            y_probas = np.array(y_probas)

            try:
                #scores = [score(y_pred, y_test) for y_pred in y_preds]
                scores = [Score().score(y_test, y_proba) for y_proba in y_probas]
            except Exception as e:
                print 'FAILED in one fold (%s)' % pred_path
                print '++++++++'
                print e
                print '++++++++'
                scores = [0.] * len(models_paths)
            # print scores
            mean_scores += scores

    # TODO: add a score column to the models df

    mean_scores /= len(pred_paths)
    leaderboard = pd.DataFrame({'score': mean_scores}, index=models.index)
    return leaderboard.sort(
        columns=['score'], ascending=not Score().higher_the_better)

def private_leaderboard_classical(orig_models):
    models = orig_models.sort(columns='timestamp')
    m_paths = [os.path.join(root_path, 'models', path) for path in models['path']]
    _, _, _, y_test, _ = split_data()
    leaderboard = models.copy()
    leaderboard['score'] = 0.0
    mean_scores = np.zeros(len(m_paths))
    # get model file names
    for mi, m_path in zip(range(len(models)), models['path']):
        pr_paths = glob.glob(os.path.join(models_path, m_path, 'test_*'))
        sum_proba = np.zeros(len(y_test))
        for pr_path in pr_paths:
             inp = pd.read_csv(pr_path, names=['pred', 'proba'])
             sum_proba += inp['proba'].values
        y_proba = sum_proba.argsort().argsort()
        leaderboard.loc[mi, 'score'] = Score().score(y_test, y_proba)
    return leaderboard.sort(
        columns=['score'], ascending=not Score().higher_the_better)


def combine_models_using_probas(y_preds, y_probas, indexes):
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
    return np.log(y_probas[indexes]).sum(axis=0)


def leaderboard_combination(groundtruth_path, orig_models):
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


    # get model file names
    counts = np.zeros(len(models), dtype=int)
    if models.shape[0] != 0:
        # get prediction file names
        pred_paths = glob.glob(os.path.join(
            models_path, models['path'][0], 'pred_*'))
        pred_paths = np.array([pred_path.split('/')[-1] for pred_path in pred_paths])
        test_paths = glob.glob(os.path.join(
            models_path, models['path'][0], 'test_*'))
        _, y_test = read_data(test_filename)
        y_test_combined_log_probas = np.zeros(len(y_test), dtype=float)
        y_test_best_log_probas = np.zeros(len(y_test), dtype=float)
        for pred_path in pred_paths:
            # probably an overshoot to use dataframes here, but slightly simpler code
            # to be simplified perhaps
            y_preds = pd.DataFrame()
            y_probas = pd.DataFrame()
            y_valid_test = pd.read_csv(
                os.path.join(groundtruth_path, pred_path), names=['pred']).values.flatten()

            for model_path in models['path']:
                predictions_path = os.path.join(root_path, 'models', model_path, pred_path)
                predictions = pd.read_csv(predictions_path, names=['pred', 'proba'])
                y_preds[model_path] = predictions['pred'] # use m_path as db key
                y_probas[model_path] = predictions['proba']

            # y_preds: k vectors of length n
            y_preds = np.transpose(y_preds.values)
            y_probas = np.transpose(y_probas.values)
            scores = [Score().score(y_valid_test, y_proba) for y_proba in y_probas]
            #print scores
            #best_indexes = np.array([np.argmin(scores)])
            best_indexes = np.array([np.argmax(scores)])

            improvement = True
            while improvement:
                old_best_indexes = best_indexes
                best_indexes = best_combine(
                    y_preds, y_probas, y_valid_test, best_indexes)
                improvement = len(best_indexes) != len(old_best_indexes)
            print best_indexes
            counts[best_indexes] += 1

            for index in best_indexes:
                test_predictions = pd.read_csv(
                    test_paths[index], names=['pred', 'proba'])
                # We divide by len(best_indexes)to avoid over-biasing by 
                # larger combinations
                y_test_combined_log_probas += \
                    np.log(test_predictions['proba'].values) / len(best_indexes)
            test_predictions = pd.read_csv(
                test_paths[best_indexes[0]], names=['pred', 'proba'])
            y_test_best_log_probas += np.log(test_predictions['proba'].values)
        print "foldwise combined test score = ", Score().score(y_test, y_test_combined_log_probas)
        print "foldwise best test score = ", Score().score(y_test, y_test_best_log_probas)

    # leaderboard = models.copy()
    leaderboard = pd.DataFrame({'score': counts}, index=models.index)
    return leaderboard.sort(columns=['score'],  ascending=False)


def best_combine(y_preds, y_probas, y_test, best_indexes):
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
    y_pred = combine_models_using_probas(y_preds, y_probas, best_indexes)
    best_index = -1
    # FIXME: I don't remember why we need eps, but without it the results are
    # very different. In any case, the Score class should take care of its eps 
    eps = 0.01/len(y_preds)
    # Combination with replacement, what Caruana suggests. Basically, if a model
    # added several times, it's upweighted.
    for i in range(len(y_preds)):
        com_y_pred = combine_models_using_probas(y_preds, y_probas, np.append(best_indexes, i))
        if Score().score(y_test, y_pred) < Score().score(y_test, com_y_pred) - eps:
            y_pred = com_y_pred
            best_index = i
    if best_index > -1:
        return np.append(best_indexes, best_index)
    else:
        return best_indexes
