# Author: Balazs Kegl
# License: BSD 3 clause
import os
import numpy as np
import pandas as pd

from sklearn.externals.joblib import Parallel, delayed

import config_databoard
import generic
import specific

n_processes = config_databoard.get_ramp_field('num_cpus')


def get_predictions_list(models_df, train_is, subdir, index_list=None):
    """Constructs a matrix of predictions (list of a vector of predictions, of
    type specific.Predictions) by reading predictions
    from the models directory, using the model indices stored in the models_df
    data frame (filled in fatch.fetch_models), the subdir (e.g. "test",
    "valid"), and the test indices, representing the cv fold. When the
    prediction file is not there (eg because the model has not yet been
    tested), we set predictions to None. The list can be still used if that
    model is not needed (for example, not in best_index_list).


    Parameters
    ----------
    models_df :
        the data frame containing the model indices.
    subdir :
        the subdirectory that contains the predictions (e.g. "test", "valid")
    test_is : represents the cv fold

    index_list : the subset of predictions to be returned. If None, the full
        set is combined.
    Returns
    -------
    predictions_list : a list of prediction arrays (of type
        specific.Predictions). Each element of the list
        is a array of predictions of a given model on the same data points.
    """
    if index_list is None:
        index_list = range(len(models_df))
    predictions_list = []
    selected_models_df = models_df.iloc[index_list]
    for tag_name_alias, model_df in selected_models_df.iterrows():
        full_model_path = generic.get_full_model_path(tag_name_alias, model_df)
        hash_string = generic.get_hash_string_from_indices(train_is)
        f_name = generic.get_f_name(full_model_path, subdir, hash_string)
        try:
            predictions = generic.get_predictions(f_name)
        except IOError as e:
            print "WARNING: ", e
            predictions = None
            raise e
        predictions_list.append(predictions)
    return predictions_list


def combine_predictions_list(predictions_list, index_list=None):
    """Combines predictions by taking the mean of their
    get_combineable_predictions views. E.g. for regression it is the actual
    predictions, and for classification it is the probability array (which
    should be calibrated if we want the best performance). Called by
    best_combine (to construct ensembles using greedy forward selection),
    get_best_index_list (which re-does the combination after the best subset
    has been found), get_combined_test_predictions,
    make_combined_test_predictions

    Parameters
    ----------
    predictions_list : a list of prediction arrays (of type
        specific.Predictions). Each element of the list
        is a array of predictions of a given model on the same data points.
    index_list : the subset of models to be combined. If None, the full set is
        combined.

    Returns
    -------
    combined_predictions : a prediction array containing the combined
        predictions
    """

    if index_list is None:  # we combine the full list
        index_list = range(len(predictions_list))
    y_comb_list = np.array(
        [predictions_list[i].y_pred_comb for i in index_list])
    y_comb = np.nanmean(y_comb_list, axis=0)
    combined_predictions = specific.Predictions(y_pred=y_comb)
    return combined_predictions


def get_bagging_score(predictions_list, fast=False):
    """Takes a list of predictions on the hold out test instances, bags them
    using combine_predictions_list, and returns the score of the bagged
    predictor. If fast is false, it logs the score of subsequent combinations.
    This is for assessing the bagging learning curve, which is useful for
    setting the number of cv folds to its optimal value (in case the RAMP is
    competitive, say, to win a Kaggle challenge; although it's kinda stupid
    since in those RAMPs we don't have a test file, so the learning curves
    should be assessed in get_cv_bagging_score on the (cross-)validation sets).
    When the database is set up, this function should be cut into two
    functions, one filling the score field, the other filling the (optional)
    learning curve field which we can then use in an optional diagnostics view
    or a GP-based test of whether the score is still improving at the last
    fold.

    Parameters
    ----------
    predictions_list : a list of specific.Predictions,
        representing predictions on the test set.

    fast : True means no construction of the bagging learning curve, only final
        score

    Returns
    -------
    score : the final bagging score
    """
    true_predictions = generic.get_true_predictions_test()
    fold_scores = []
    if fast:
        combine_range = [len(predictions_list) - 1]
    else:
        combine_range = range(len(predictions_list))
    for i in combine_range:
        combined_predictions = combine_predictions_list(
            predictions_list[:i + 1])
        fold_score = specific.score(true_predictions, predictions_list[i])
        fold_scores.append(fold_score)
        score = specific.score(true_predictions, combined_predictions)
        generic.logger.info(
            "Fold {}: score on fold = {}, combined score after fold = {}".
            format(i, fold_score, score))
    fold_scores = np.array(fold_scores, dtype=float)
    generic.logger.info("Mean of scores = {0:.4f}".format(fold_scores.mean()))
    generic.logger.info("Std of scores = {0:.4f}".format(fold_scores.std()))
    generic.logger.info("Score of \"means\" (cv bagging) = {}".format(score))
    generic.logger.info("------------")
    return score


def get_cv_bagging_score(predictions_list, test_is_list):
    """Takes a list of predictions on the (cross-)validation instances, cv-bags
    them using combine_predictions_list, and returns the score of the bagged
    predictor. The predictions in predictions_list[i] belong to those indicated
    by test_is_list[i].

    Parameters
    ----------
    predictions_list : a list of specific.Predictions,
        representing predictions on the (cross-)validation sets
    test_is_list : a list of index lists, representing the indices of
        validation points in each fold

    Returns
    -------
    score : the cv-bagged score
    """

    true_predictions_train = generic.get_true_predictions_train()
    true_predictions_valid_list = generic.get_true_predictions_valid_list()
    n_samples = true_predictions_train.n_samples
    y_comb_array = np.array(
        [specific.Predictions(n_samples=n_samples) for _ in predictions_list])
    fold_scores = []
    # We crashed here because smebody output a matrix in predict proba with
    # 4 times more rows. We should check this in train_test
    for i, test_is in enumerate(test_is_list):
        y_comb_array[i].set_valid_predictions(predictions_list[i], test_is)
        combined_predictions = combine_predictions_list(y_comb_array[:i + 1])
        valid_indexes = combined_predictions.valid_indexes
        fold_score = specific.score(
            true_predictions_valid_list[i], predictions_list[i])
        fold_scores.append(fold_score)
        score = specific.score(
            true_predictions_train, combined_predictions, valid_indexes)
        coverage = 1.0 * np.count_nonzero(valid_indexes) / n_samples
        generic.logger.info("Fold {}: score on fold = {}, combined score after"
                            " fold = {}, coverage = {:>3}%".format(
                                i, fold_score, score,
                                int(round(100 * coverage))))
    fold_scores = np.array(fold_scores, dtype=float)
    generic.logger.info("Mean of scores = {0:.4f}".format(fold_scores.mean()))
    generic.logger.info("Std of scores = {0:.4f}".format(fold_scores.std()))
    generic.logger.info("Score of \"means\" (cv bagging) = {}".format(score))
    generic.logger.info("------------")
    return score


def best_combine(predictions_list, true_predictions, best_index_list):
    """Finds the model that minimizes the score if added to
    predictions_list[best_index_list]. If there is no model improving the input
    combination, the input best_index_list is returned. Otherwise the best
    model is added to the list. We could also return the combined prediction
    (for efficiency, so the combination would not have to be done each time;
    right now the algo is quadratic), but I don't think any meaningful
    rule will be associative, in which case we should redo the combination from
    scratch each time the set changes. Since now combination = mean, we could
    maintain the sum and the number of models, but it would be a bit bulky.
    We'll see how this evolves.

    Parameters
    ----------
    predictions_list : a list of specific.Predictions
    true_predictions : the ground truth of type specific.Predictions
    best_index_list : a list of indices of the current best model

    Returns
    -------
    best_index_list : the new combined list.
    """
    best_predictions = combine_predictions_list(
        predictions_list, best_index_list)
    best_score = specific.score(true_predictions, best_predictions)
    best_index = -1
    # Combination with replacement, what Caruana suggests. Basically, if a
    # model is added several times, it's upweighted, leading to
    # integer-weighted ensembles
    for i in range(len(predictions_list)):
        combined_predictions = combine_predictions_list(
            predictions_list, np.append(best_index_list, i))
        new_score = specific.score(true_predictions, combined_predictions)
        # '>' is overloaded in score, so 'x > y' means 'x is better than y'
        if new_score > best_score:
            best_predictions = combined_predictions
            best_index = i
            best_score = new_score
    if best_index > -1:
        return np.append(best_index_list, best_index)
    else:
        return best_index_list


def get_best_index_list(models_df, train_is, test_is,
                        selected_index_list=None):
    """Constructs the best model combination on a single fold, using greedy
    forward selection. See http://www.cs.cornell.edu/~caruana/ctp/ct.papers/
    caruana.icml04.icdm06long.pdf. Also returns the single best prediction

    Parameters
    ----------
    models_df : the models to combine
    train_is : the list of training indices
    test_is : the list of test indices
    selected_index_list : a list of model indices to combine (in case of
        bagging we may ignore some models here)

    Returns
    -------
    best_selected_index_list : the index list of models of the best ensemble
    best_predictions : the predictions of the single best model
    combined_predictions : the predictions of the combined model
    """
    if selected_index_list is None:
        selected_index_list = np.arange(len(models_df))
    true_predictions = generic.get_true_predictions_valid(test_is)

    # careful: we only fetch here the selected predictions
    # best_index_list will be the index list within the selected list
    # best_selected_index_list will be the index list within the full
    # model list
    predictions_list = get_predictions_list(
        models_df, train_is, subdir="valid", index_list=selected_index_list)

    valid_scores = [specific.score(true_predictions, predictions)
                    for predictions in predictions_list]
    best_prediction_index = np.argmax(valid_scores)
    best_predictions = predictions_list[best_prediction_index]

    best_index_list = np.array([best_prediction_index])

    improvement = True
    max_len_best_index_list = 80  # should be a config parameter
    while improvement and len(best_index_list) < max_len_best_index_list:
        old_best_index_list = best_index_list
        best_index_list = best_combine(
            predictions_list, true_predictions, best_index_list)
        improvement = len(best_index_list) != len(old_best_index_list)
    generic.logger.info("best indices = {}".format(
        selected_index_list[best_index_list]))
    best_selected_index_list = selected_index_list[best_index_list]
    combined_predictions = combine_predictions_list(
        predictions_list, best_index_list)
    return best_selected_index_list, best_predictions, combined_predictions


def get_combined_test_predictions(models_df, train_is, best_index_list):
    assert(len(best_index_list) > 0)

    test_predictions_list = get_predictions_list(
        models_df, train_is, "test", best_index_list)

    combined_test_predictions = combine_predictions_list(
        test_predictions_list, range(len(test_predictions_list)))
    foldwise_best_test_predictions = test_predictions_list[0]
    return combined_test_predictions, foldwise_best_test_predictions


# TODO: make an actual prediction, for example, for submitting to Kaggle
# TODO: right now we just print the score on the screen, should be saved
# somewhere
def make_combined_test_prediction(best_index_lists, models_df):
    if models_df.shape[0] != 0:
        true_predictions = generic.get_true_predictions_test()
        train_is_list = generic.get_train_is_list()
        if config_databoard.is_parallelize:
            list_of_tuples = Parallel(n_jobs=n_processes, verbose=0)(delayed(
                get_combined_test_predictions)(
                    models_df, train_is, best_index_list)
                for best_index_list, train_is
                in zip(best_index_lists, train_is_list))
            combined_test_predictions_list, foldwise_best_test_predictions_list = \
                zip(*list_of_tuples)
        else:
            combined_test_predictions_list = []
            foldwise_best_test_predictions_list = []
            for best_index_list, train_is in zip(
                    best_index_lists, train_is_list):
                combined_test_predictions, foldwise_best_test_predictions = \
                    get_combined_test_predictions(
                        models_df, train_is, best_index_list)
                combined_test_predictions_list.append(
                    combined_test_predictions)
                # best in the fold
                foldwise_best_test_predictions_list.append(
                    foldwise_best_test_predictions)

        combined_combined_test_predictions = combine_predictions_list(
            combined_test_predictions_list)
        generic.logger.info("foldwise combined test score = {}".format(
            specific.score(
                true_predictions, combined_combined_test_predictions)))
        combined_combined_test_predictions.save_predictions(
            os.path.join(config_databoard.private_data_path,
                         "foldwise_combined.csv"))
        combined_foldwise_best_test_predictions = combine_predictions_list(
            foldwise_best_test_predictions_list)
        combined_foldwise_best_test_predictions.save_predictions(
            os.path.join(config_databoard.private_data_path,
                         "foldwise_best.csv"))
        generic.logger.info("foldwise best test score = {}".format(
            specific.score(
                true_predictions, combined_foldwise_best_test_predictions)))


def leaderboard_classical(models_df, subdir="valid", calibrate=False):
    mean_scores = []

    if models_df.shape[0] != 0:
        train_is_list = generic.get_train_is_list()
        test_is_list = generic.get_test_is_list()

        if config_databoard.is_parallelize:
            predictions_lists = Parallel(n_jobs=n_processes, verbose=0)(
                delayed(get_predictions_list)(
                    models_df, train_is, subdir=subdir)
                for train_is in train_is_list)
        else:
            predictions_lists = [
                get_predictions_list(models_df, train_is, subdir=subdir)
                for train_is in train_is_list]

        if subdir == "valid":
            generic.logger.info("Combining models on validation")
            mean_scores = [get_cv_bagging_score(predictions_list, test_is_list)
                           for predictions_list in zip(*predictions_lists)]
        else:  # subdir == "test"
            generic.logger.info("Combining models on test")
            mean_scores = [get_bagging_score(predictions_list)
                           for predictions_list in zip(*predictions_lists)]

    generic.logger.info("classical leaderboard mean {} scores = {}".
                        format(subdir, mean_scores))
    leaderboard = pd.DataFrame({'score': mean_scores}, index=models_df.index)
    return leaderboard.sort(columns=['score'])


def leaderboard_combination(orig_models_df, test=False):
    models_df = orig_models_df.sort(columns='timestamp')
    # contributivity counts
    integer_precentage_counts = []

    if models_df.shape[0] != 0:
        # The following should go into config, we'll get there when we have a
        # lot of models.
        # One of Caruana's trick: bag the models
        # selected_index_lists = np.array([random.sample(
        #    range(len(models_df)), int(0.8*models_df.shape[0]))
        #    for _ in range(num_bags)])
        # Or you can select a subset
        # selected_index_lists = np.array([[24, 26, 28, 31]])
        # Or just take everybody
        selected_index_lists = np.array([range(len(models_df))])
        cv = generic.get_cv()
        generic.logger.info("Combining models {}".format(selected_index_lists))
        if config_databoard.is_parallelize:
            list_of_tuples = Parallel(n_jobs=n_processes, verbose=0)(delayed(
                get_best_index_list)(
                    models_df, train_is, test_is, selected_index_list)
                for train_is, test_is in cv
                for selected_index_list in selected_index_lists)
        else:
            list_of_tuples = [
                get_best_index_list(
                    models_df, train_is, test_is, selected_index_list)
                for train_is, test_is in cv
                for selected_index_list in selected_index_lists]

        best_index_lists, foldwise_best_predictions_list, \
            combined_predictions_list = zip(*list_of_tuples)

        # contributivity counts
        normalized_counts = np.zeros(models_df.shape[0], dtype=float)
        for best_index_list in best_index_lists:
            fold_counts = np.histogram(
                best_index_list, bins=range(models_df.shape[0] + 1))[0]
            normalized_counts += 1.0 * fold_counts / fold_counts.sum()
        normalized_counts = 100. * normalized_counts / normalized_counts.sum()
        # we have 1 for every model picked at least once
        normalized_counts[normalized_counts > 0] = np.maximum(
            1.0, normalized_counts[normalized_counts > 0])
        normalized_counts += 0.4999
        integer_precentage_counts = normalized_counts.astype(int)

        if test:
            make_combined_test_prediction(best_index_lists, models_df)

    leaderboard = pd.DataFrame(
        {'contributivity': integer_precentage_counts}, index=models_df.index)
    return leaderboard.sort(columns=['contributivity'], ascending=False)


def leaderboard_execution_times(models_df):
    hash_strings = generic.get_cv_hash_string_list()
    num_cv_folds = len(hash_strings)
    leaderboard = pd.DataFrame(index=models_df.index)
    leaderboard['train time'] = np.zeros(models_df.shape[0])
    # we name it "test" (not "valid") bacause this is what it is from the
    # participant's point of view (ie, "public test")
    leaderboard['test time'] = np.zeros(models_df.shape[0])

    if models_df.shape[0] != 0:
        for hash_string in hash_strings:
            for idx, model_df in models_df.iterrows():
                full_model_path = generic.get_full_model_path(idx, model_df)
                try:
                    with open(generic.get_train_time_f_name(
                            full_model_path, hash_string), 'r') as f:
                        leaderboard.loc[
                            idx, 'train time'] += abs(float(f.read()))
                except IOError:
                    generic.logger.debug(
                        "Can't open {}, setting training time to 0".format(
                            generic.get_train_time_f_name(
                                full_model_path, hash_string)))
                try:
                    with open(generic.get_valid_time_f_name(
                            full_model_path, hash_string), 'r') as f:
                        leaderboard.loc[
                            idx, 'test time'] += abs(float(f.read()))
                except IOError:
                    generic.logger.debug(
                        "Can't open {}, setting testing time to 0".format(
                            generic.get_valid_time_f_name(
                                full_model_path, hash_string)))

    leaderboard['train time'] = map(
        int, leaderboard['train time'] / num_cv_folds)
    leaderboard['test time'] = map(
        int, leaderboard['test time'] / num_cv_folds)
    generic.logger.info("Classical leaderboard train times = {}".
                        format(leaderboard['train time'].values))
    generic.logger.info("Classical leaderboard valid times = {}".
                        format(leaderboard['test time'].values))
    return leaderboard
