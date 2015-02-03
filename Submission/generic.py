import hashlib, imp, glob, csv
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def train_model(m_path, X, y, skf):
    print m_path
    model = imp.load_source('model',m_path + "/model.py")
    hasher = hashlib.md5()
    f_name_score = m_path + "/score.csv"
    scores = []
    for train_is, test_is in skf:
        hasher.update(test_is)
        h_str = hasher.hexdigest()
        f_name_pred = m_path + "/pred_" + h_str + ".csv"
        print f_name_pred
        X_train = X[train_is]
        y_train = y[train_is]
        X_test = X[test_is]
        y_test = y[test_is]
        y_pred = model.model(X_train, y_train, X_test)
        np.savetxt(f_name_pred, y_pred, delimiter="\n", fmt='%d')
        acc = accuracy_score(y_test, y_pred)
        score = str(1 - acc)
        scores.append([h_str, len(test_is), score]) # error
        csv.writer(open(f_name_score, "w")).writerows(scores)
        print acc

def leaderboard_classical(m_paths):
    mean_scores = []
    m_names = []
    for m_path in m_paths:
        print m_path
        m_name = m_path.split("/")[-1]
        print m_name
        scores = pd.read_csv(m_path + "/score.csv", names=["h_str", "n", "score"])
        mean_scores = np.append(mean_scores, scores['score'].mean())
        m_names = np.append(m_names, m_name)
    print mean_scores[0]
    ordering = mean_scores.argsort() # error: decreasing order
    print mean_scores[ordering]
    leaderboard = pd.DataFrame()
    leaderboard['model'] = m_names[ordering]
    leaderboard['error'] = mean_scores[ordering]
    return leaderboard

