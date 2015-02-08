import os
import hashlib, imp, glob, csv
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from scipy import io
import multiprocessing
from functools import partial
from config import root_path, n_CV, test_size, n_processes, random_state

def read_data():
    data = io.loadmat('dataMarathon.mat')
    Z = np.c_[data['data_target'].astype(np.int), data['X']]
    label_col = u'TARGET'
    columns = [label_col] + [d[0] for d in data['header'].ravel()]
    df = pd.DataFrame(Z, columns=columns)
    Z = df.values
    y = Z[:, 0]
    X = Z[:, 1:]
    return X, y

def setup_ground_truth(gt_path, y, skf):
    """Setting up the GroundTruth subdir, saving y_test for each fold in skf. File
    names are pred_<hash of the test index vector>.csv.

    Parameters
    ----------
    gt_paths : ground truth path
    y : array-like, shape = [n_instances]
        the label vector
    """
    print gt_path
    scores = []
    for train_is, test_is in skf:
        hasher = hashlib.md5()
        hasher.update(test_is)
        h_str = hasher.hexdigest()
        f_name_pred = gt_path + "/pred_" + h_str + ".csv"
        print f_name_pred
        np.savetxt(f_name_pred, y[test_is], delimiter="\n", fmt='%d')

def save_scores(skf_is, m_path, X, y, f_name_score):
    hasher = hashlib.md5()
    train_is, test_is = skf_is
    hasher.update(test_is)
    h_str = hasher.hexdigest()
    f_name_pred = m_path + "/pred_" + h_str + ".csv"
    X_train = X[train_is]
    y_train = y[train_is]
    X_test = X[test_is]
    y_test = y[test_is]
    model = imp.load_source('model',m_path + "/model.py")
    y_pred, y_score = model.model(X_train, y_train, X_test)
    # y_rank[i] is the the rank of the ith element of y_score
    y_rank = y_score[:,1].argsort().argsort()
    output = np.transpose(np.array([y_pred, y_rank]))
    np.savetxt(f_name_pred, output, fmt='%d,%d')
    acc = accuracy_score(y_test, y_pred)
    score = str(1 - acc)
    #scores.append([h_str, len(test_is), score]) # error
    csv.writer(open(f_name_score, "a")).writerow([h_str, len(test_is), score])
    print f_name_pred, acc


def train_model(m_path, X, y, skf):
    """Training a model on all folds and saving the predictions and rank order. The latter we can
    use for computing ROC or cutting ties.

    m_path/model.py 
    should contain the model function, for example

    def model(X_train, y_train, X_test):
    clf = Pipeline([('imputer', Imputer(strategy='most_frequent')),
        ('rf', AdaBoostClassifier(base_estimator=RandomForestClassifier(max_depth=5, n_estimators=100),
                         n_estimators=20))])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    return y_pred, y_score

    File names are pred_<hash of the np.array(test index vector)>.csv. A score.csv file is also
    saved with k_model lines of header
    <hash of the test index vector>, <number of test instances>, <error>
    evaluations are parallel, so the order in score.csv is undefined.

    Parameters
    ----------
    m_paths : array-like, shape = [k_models]
    X : array-like, shape = [n_instances, d_features]
    y : array-like, shape = [n_instances]
        the label vector
    skf : array-like, shape = [N_folds], a cross_validation object
    """

    print m_path
    model = imp.load_source('model',m_path + "/model.py")
    f_name_score = m_path + "/score.csv"
    scores = []
    f = open(f_name_score, "w")
    f.close()
    partial_save_scores = partial(save_scores, m_path=m_path, X=X, y=y, f_name_score=f_name_score)
    pool = multiprocessing.Pool(processes=n_processes)
    pool.map(partial_save_scores, skf)
    pool.close()

def leaderboard_to_html(leaderboard):
    return leaderboard.to_html()

def leaderboard_classical(models):
    """Output classical leaderboard (sorted in increasing order by score).

    Parameters
    ----------
    m_paths : array-like, shape = [k_models]
        A list of paths, each containing a "score.csv" file with three columns:
            1) the file prefix (hash of the test index set the model was tested on)
            2) the number of test points
            3) the test score (the lower the better; error)
    Returns
    -------
    leaderboard : a pandas DataFrame with two columns:
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
    
    mean_scores = []
    m_paths = [os.path.join(root_path, 'Submission', 'Models', path) for path in models['path']]
    for m_path in m_paths:
        #print m_path
        scores = pd.read_csv(m_path + "/score.csv", names=["h_str", "n", "score"])
        mean_scores = np.append(mean_scores, scores['score'].mean())
    #print mean_scores[0]
    ordering = mean_scores.argsort() # error: increasing order
    #print mean_scores[ordering]
    leaderboard = models.copy()
    leaderboard['score'] = mean_scores[ordering.argsort()] # argsort of argsort gives rank of entry
    return leaderboard.sort(columns=['score'],  ascending=True)

def combine_models(y_preds, y_ranks, indexes):
    """Combines the predictions y_preds[indexes] by "rank"
    voting. I'll detail it once you verify that it makes sense (see my mail)

    Parameters
    ----------
    y_preds : array-like, shape = [k_models, n_instances], binary
    y_ranks : array-like, shape = [k_models, n_instances], permutation of [0,...,n_instances]
    indexes : array-like, shape = [max k_models], a set of indices of 
        models to combine
    Returns
    -------
    com_y_pred : array-like, shape = [n_instances], a list of (combined) 
        binary predictions.
    """
    k = len(indexes)
    n = len(y_preds[0])
    n_ones = n * k - y_preds[indexes].sum() # number of zeros
    sum_y_ranks = y_ranks[indexes].sum(axis=0) + k #sum of ranks \in [1,n]
    com_y_pred = np.zeros(n, dtype=int)
    com_y_pred[np.greater(sum_y_ranks, n_ones)] = 1
    return com_y_pred

def leaderboard_combination(models, gt_path):
    """Output combined leaderboard (sorted in decreasing order by score). We use
    Caruana's greedy combination
    http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf
    on each fold, and cound how many times each model is chosen.

    Parameters
    ----------
    gt_paths : ground truth path
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
    def score(y_pred, y_test):
        return 1 - accuracy_score(y_pred, y_test)

    def best_combine(y_preds, y_ranks, best_indexes):
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
        eps = 0.01/len(y_preds)
        y_pred = combine_models(y_preds, y_ranks, best_indexes)
        best_index = -1
        # Combination with replacement, what Caruana suggests. Basically, if a model
        # added several times, it's upweighted.
        for i in range(len(y_preds)):
            com_y_pred = combine_models(y_preds, y_ranks, np.append(best_indexes, i))
            #print score(y_pred, y_test), score(com_y_pred, y_test)
            if score(y_pred, y_test) > score(com_y_pred, y_test) + eps:
                y_pred = com_y_pred
                best_index = i
        if best_index > -1:
            return np.append(best_indexes, best_index)
        else:
            return best_indexes

    # get prediction file names
    pr_paths = glob.glob(gt_path + "/pred_*")[:n_CV]
    pr_names = np.array([pr_path.split('/')[-1] for pr_path in pr_paths])
    # get model file names
    counts = np.zeros(len(models), dtype=int)
    for pr_name in pr_names:
        # probab;y an overshoot to use dataframes here, but slightly simpler code
        # to be simplified perhaps
        y_preds = pd.DataFrame()
        y_ranks = pd.DataFrame()
        y_test = pd.read_csv(gt_path + '/' + pr_name, names=['pred']).values.flatten()
        for m_path in models['path']:
            pr_path = os.path.join(root_path, 'Submission', 'Models', m_path, pr_name)
            print pr_path
            inp = pd.read_csv(pr_path, names=['pred', 'rank'])
            y_preds[m_path] = inp['pred'] # use m_path as db key
            y_ranks[m_path] = inp['rank']
        # y_preds: k vectors of length n
        y_preds = np.transpose(y_preds.values)
        y_ranks = np.transpose(y_ranks.values)
        scores = [score(y_pred, y_test) for y_pred in y_preds]
        #print scores
        best_indexes = np.array([np.argmin(scores)])
        #print best_indexes
        improvement = True
        while improvement:
            old_best_indexes = best_indexes
            best_indexes = best_combine(y_preds, y_ranks, best_indexes)
            improvement = len(best_indexes) != len(old_best_indexes)
            #print best_indexes
        counts[best_indexes] += 1
        #print counts
    ordering = (-counts).argsort() # count: decreasing order
    #print ordering
    leaderboard = models.copy()
    leaderboard['score'] = counts[ordering.argsort()] # argsort of argsort gives rank of entry
    return leaderboard.sort(columns=['score'],  ascending=False)
 