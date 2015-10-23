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


def get_predictions_list(models_df, hash_string, subdir, index_list=None):
    """Constructs a matrix of predictions (list of a vector of predictions, of
    type specific.prediction_type.PredictionArrayType) by reading predictions
    from the models directory, using the model indices stored in the models_df
    data frame (filled in fatch.fetch_models), the subdir (e.g. "test",
    "valid"), and the hash string, representing the cv fold. When the
    prediction file is not there (eg because the model has not yet been
    tested), we set predictions to None. The list can be still used if that
    model is not needed (for example, not in best_index_list).


    Parameters
    ----------
    models_df :
        the data frame containing the model indices.
    subdir :
        the subdirectory that contains the predictions (e.g. "test", "valid")
    hash_string : represents the cv fold

    Returns
    -------
    predictions_list : a list of prediction arrays (of type
        specific.prediction_type.PredictionArrayType). Each element of the list
        is a array of predictions of a given model on the same data points.
    """
    if index_list is None:
        index_list = range(len(models_df))
    predictions_list = []
    selected_models_df = models_df.iloc[index_list]
    for tag_name_alias, model_df in selected_models_df.iterrows():
        full_model_path = generic.get_full_model_path(tag_name_alias, model_df)
        predictions_f_name = generic.get_f_name(
            full_model_path, subdir, hash_string)
        try:
            predictions = specific.prediction_type.PredictionArrayType(
                predictions_f_name=predictions_f_name)
        except IOError as e:
            print "WARNING: ", e
            predictions = None
        predictions_list.append(predictions)
    return predictions_list


def combine_predictions_list(predictions_list, indexes=[]):
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
        specific.prediction_type.PredictionArrayType). Each element of the list
        is a array of predictions of a given model on the same data points.
    indexes : the subset of models to be combined. If [], the full set is
        combined.

    Returns
    -------
    combined_predictions : a prediction array containing the combined
        predictions
    """

    if len(indexes) == 0:  # we combine the full list
        indexes = range(len(predictions_list))

    y_combineable_array_list = \
        np.array([predictions_list[i].get_combineable_predictions()
                  for i in indexes])
    y_combined_array = y_combineable_array_list.mean(axis=0)
    combined_predictions = specific.prediction_type.PredictionArrayType(
        y_prediction_array=y_combined_array)
    return combined_predictions


def get_ground_truth(ground_truth_f_name):
    return specific.prediction_type.PredictionArrayType(
        ground_truth_f_name=ground_truth_f_name)


def get_ground_truth_valid_list(hash_strings):
    return [get_ground_truth(generic.get_ground_truth_valid_f_name(hash_string))
            for hash_string in hash_strings]


def get_bagging_score(predictions_list):
    ground_truth = get_ground_truth(generic.get_ground_truth_test_f_name())
    fold_scores = []
    for i in range(len(predictions_list)):
        combined_predictions = combine_predictions_list(
            predictions_list[:i + 1])
        fold_score = specific.score(ground_truth, predictions_list[i])
        fold_scores.append(fold_score)
        score = specific.score(ground_truth, combined_predictions)
        generic.logger.info(
            "Fold {}: score on fold = {}, combined score after fold = {}".
            format(i, fold_score, score))
    fold_scores = np.array(fold_scores, dtype=float)
    generic.logger.info("Mean of scores = {0:.4f}".format(fold_scores.mean()))
    generic.logger.info("Std of scores = {0:.4f}".format(fold_scores.std()))
    generic.logger.info("Score of \"means\" (cv bagging) = {}".format(score))
    generic.logger.info("------------")
    return score


def get_cv_bagging_score(predictions_list, cv, hash_strings, num_points):
    """Input is a list of predictions of a single model on a list of folds."""
    # nan array of the shape of the final combined predictions
    y_combineable_array_list = np.array(
        [specific.prediction_type.get_nan_combineable_predictions(num_points)
         for _ in predictions_list])
    ground_truth_valid_list = get_ground_truth_valid_list(hash_strings)
    ground_truth_full_prediction_array = \
        specific.prediction_type.get_nan_combineable_predictions(num_points)
    fold_scores = []
    # We crashed here because smebody output a matrix in predict proba with
    # 4 times more rows. We should check this in train_test
    for i in range(len(cv)):
        _, test_is = list(cv)[i]
        y_combineable_array_list[i][test_is] = \
            predictions_list[i].get_combineable_predictions()
        # this should maybe handled in the prediction_type
        ground_truth_full_prediction_array[test_is] = \
            ground_truth_valid_list[i].get_prediction_array()
        # indices of points which appear at least in one test set
        # TODO: this is definitely multiclass only
        if len(ground_truth_full_prediction_array.shape) == 1:
            valid_indexes = ~np.isnan(ground_truth_full_prediction_array)
        else:
            valid_indexes = ~np.isnan(ground_truth_full_prediction_array)[:, 0]
        # computing score after each fold for info
        y_combined_array = np.nanmean(y_combineable_array_list[:i + 1], axis=0)
        combined_predictions = specific.prediction_type.PredictionArrayType(
            y_prediction_array=y_combined_array[valid_indexes])
        fold_score = specific.score(
            ground_truth_valid_list[i], predictions_list[i])
        fold_scores.append(fold_score)
        ground_truth_full = specific.prediction_type.PredictionArrayType(
            y_prediction_array=ground_truth_full_prediction_array[valid_indexes])
        score = specific.score(ground_truth_full, combined_predictions)
        coverage = 1.0 * \
            len(ground_truth_full_prediction_array[valid_indexes]) / num_points
        generic.logger.info("Fold {}: score on fold = {}, combined score after fold = {}, coverage = {:>4}%".
                            format(hash_strings[i], fold_score, score, int(round(100 * coverage))))
    fold_scores = np.array(fold_scores, dtype=float)
    generic.logger.info("Mean of scores = {0:.4f}".format(fold_scores.mean()))
    generic.logger.info("Std of scores = {0:.4f}".format(fold_scores.std()))
    generic.logger.info("Score of \"means\" (cv bagging) = {}".format(score))
    generic.logger.info("------------")
    return score


def leaderboard_execution_times(models_df):
    hash_strings = generic.get_hash_strings_from_ground_truth()
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
                        "Can't open %s, setting training time to 0" %
                        generic.get_train_time_f_name(full_model_path,
                                                      hash_string))
                try:
                    with open(generic.get_valid_time_f_name(
                            full_model_path, hash_string), 'r') as f:
                        leaderboard.loc[
                            idx, 'test time'] += abs(float(f.read()))
                except IOError:
                    generic.logger.debug(
                        "Can't open %s, setting testing time to 0" %
                        generic.get_valid_time_f_name(full_model_path,
                                                      hash_string))

    leaderboard['train time'] = map(
        int, leaderboard['train time'] / num_cv_folds)
    leaderboard['test time'] = map(
        int, leaderboard['test time'] / num_cv_folds)
    generic.logger.info("Classical leaderboard train times = {}".
                        format(leaderboard['train time'].values))
    generic.logger.info("Classical leaderboard valid times = {}".
                        format(leaderboard['test time'].values))
    return leaderboard


def get_scores(models_df, hash_string, subdir="valid"):
    generic.logger.info("Evaluating : {}".format(hash_string))
    if subdir == "valid":
        ground_truth_filename = generic.get_ground_truth_valid_f_name(
            hash_string)
    else:  # subdir == "test"
        ground_truth_filename = generic.get_ground_truth_test_f_name()
    ground_truth = get_ground_truth(ground_truth_filename)
    specific.score.set_eps(0.01 / len(ground_truth.get_prediction_array()))

    predictions_list = get_predictions_list(models_df, hash_string, subdir)

    scores = np.array([specific.score(ground_truth, predictions)
                       for predictions in predictions_list])
    return scores


def best_combine(predictions_list, ground_truth, best_indexes):
    """Finds the model that minimizes the score if added to 
    predictions_list[best_indexes].

    Parameters
    ----------
    predictions_list : array-like, shape = (k_models, n_instances), binary
    best_indexes : array-like, shape = [max k_models]
        A set of indices of the current best model
    Returns
    -------
    best_indexes : array-like, shape = [max k_models], a list of indices. If
    no model improving the input combination, the input index set is returned.
    otherwise the best model is added to the set. We could also return the
    combined prediction (for efficiency, so the combination would not have to
    be done each time; right now the algo is quadratic), but first I don't think
    any meaningful rules will be associative, in which case we should redo the
    combination from scratch each time the set changes.
    """
    best_predictions = combine_predictions_list(predictions_list, best_indexes)
    best_index = -1
    # Combination with replacement, what Caruana suggests. Basically, if a model
    # added several times, it's upweighted, leading to integer-weighted
    # ensembles
    for i in range(len(predictions_list)):
        combined_predictions = combine_predictions_list(
            predictions_list, np.append(best_indexes, i))
        if specific.score(ground_truth, combined_predictions) > \
           specific.score(ground_truth, best_predictions):
            best_predictions = combined_predictions
            best_index = i
            # print i, specific.score(ground_truth, combined_predictions)
    if best_index > -1:
        return np.append(best_indexes, best_index)
    else:
        return best_indexes

# TODO: this shuold be an inner consistency operation in OutputType


def get_y_pred_array(y_probas_array):
    return np.array([specific.prediction_type.labels[y_probas.argmax()]
                     for y_probas in y_probas_array])



# TODO: This is completely ouput_type dependent, so will probably go there.


def get_best_index_list(models_df, hash_string, selected_index_list=[]):
    if len(selected_index_list) == 0:
        selected_index_list = np.arange(len(models_df))
    ground_truth_valid_filename = generic.get_ground_truth_valid_f_name(
        hash_string)
    ground_truth_valid = get_ground_truth(ground_truth_valid_filename)

    predictions_list = get_predictions_list(
        models_df, hash_string, subdir="valid")
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
    generic.logger.info("best indices = {}".format(
        selected_index_list[best_index_list]))
    combined_predictions = combine_predictions_list(
        predictions_list, best_index_list)
    return selected_index_list[best_index_list], foldwise_best_predictions, \
        combined_predictions, ground_truth_valid


def get_combined_test_predictions(models_df, hash_string, best_index_list):
    assert(len(best_index_list) > 0)
    generic.logger.info("Evaluating combined test: {}".format(hash_string))

    test_predictions_list = get_predictions_list(
        models_df, hash_string, "test", best_index_list)
    test_predictions_list = [test_predictions_list[i]
                             for i in best_index_list]

    combined_test_predictions = combine_predictions_list(
        test_predictions_list, range(len(test_predictions_list)))
    foldwise_best_test_predictions = test_predictions_list[0]
    return combined_test_predictions, foldwise_best_test_predictions

# TODO: make an actual prediction


def make_combined_test_prediction(best_index_lists, models_df, hash_strings):
    if models_df.shape[0] != 0:
        ground_truth_test_filename = generic.get_ground_truth_test_f_name()
        ground_truth_test = get_ground_truth(ground_truth_test_filename)

        if config_databoard.is_parallelize:
            list_of_tuples = Parallel(n_jobs=n_processes, verbose=0)\
                (delayed(get_combined_test_predictions)(
                    models_df, hash_string, best_index_list)
                 for hash_string, best_index_list
                 in zip(hash_strings, best_index_lists))
            combined_test_predictions_list, foldwise_best_test_predictions_list = \
                zip(*list_of_tuples)
        else:
            combined_test_predictions_list = []
            foldwise_best_test_predictions_list = []
            for hash_string, best_index_list in zip(hash_strings, best_index_lists):
                combined_test_predictions, foldwise_best_test_predictions = \
                    get_combined_test_predictions(
                        models_df, hash_string, best_index_list)
                combined_test_predictions_list.append(
                    combined_test_predictions)
                # best in the fold
                foldwise_best_test_predictions_list.append(
                    foldwise_best_test_predictions)

        combined_combined_test_predictions = combine_predictions_list(
            combined_test_predictions_list)
        generic.logger.info("foldwise combined test score = {}".format(specific.score(
            ground_truth_test, combined_combined_test_predictions)))
        combined_combined_test_predictions.save_predictions(
            os.path.join(config_databoard.private_data_path, "foldwise_combined.csv"))
        combined_foldwise_best_test_predictions = combine_predictions_list(
            foldwise_best_test_predictions_list)
        combined_foldwise_best_test_predictions.save_predictions(
            os.path.join(config_databoard.private_data_path, "foldwise_best.csv"))
        generic.logger.info("foldwise best test score = {}".format(specific.score(
            ground_truth_test, combined_foldwise_best_test_predictions)))


def leaderboard_classical(models_df, subdir="valid", calibrate=False):
    mean_scores = []

    if models_df.shape[0] != 0:
        _, y_train_array = specific.get_train_data()
        cv = specific.get_cv(y_train_array)
        num_train = len(y_train_array)
        # we need the hash strings in the same order as train/test_is
        hash_strings = [generic.get_hash_string_from_indices(train_is)
                        for train_is, test_is in cv]
        if config_databoard.is_parallelize:
            predictions_lists = Parallel(n_jobs=n_processes, verbose=0)\
                (delayed(get_predictions_list)(models_df, hash_string,
                                               subdir=subdir)
                 for hash_string in hash_strings)
        else:
            predictions_lists = [
                get_predictions_list(models_df, hash_string, subdir=subdir)
                for hash_string in hash_strings]

        # predictions_lists is a matrix of predictions, size
        # num_folds x num_models. We call mean_score per model, that is,
        # for every column
        if subdir == "valid":
            generic.logger.info("Combining models on validation")
            mean_scores = [get_cv_bagging_score(
                predictions_list, cv, hash_strings, num_train)
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
    normalized_counts = np.zeros(models_df.shape[0], dtype=float)
    integer_precentage_counts = []

    if models_df.shape[0] != 0:
        _, y_train_array = specific.get_train_data()
        cv = specific.get_cv(y_train_array)
        # One of Caruana's trick: bag the models
        # selected_index_lists = np.array([random.sample(
        #    range(len(models_df)), int(0.8*models_df.shape[0]))
        #    for _ in range(num_bags)])
        # Or you can select a subset
        #selected_index_lists = np.array([[24, 26, 28, 31]])
        # Or just take everybody
        selected_index_lists = np.array([range(len(models_df))])
        generic.logger.info("Combining models {}".format(selected_index_lists))

        # we need the hash strings in the same order as train/test_is, can't
        # get them from the file names
        hash_strings = [generic.get_hash_string_from_indices(train_is)
                        for train_is, test_is in cv]
        if config_databoard.is_parallelize:
            list_of_tuples = Parallel(n_jobs=n_processes, verbose=0)\
                (delayed(get_best_index_list)(models_df, hash_string, selected_index_list)
                 for hash_string in hash_strings
                 for selected_index_list in selected_index_lists)
        else:
            list_of_tuples = [
                get_best_index_list(
                    models_df, hash_string, selected_index_list)
                for hash_string in hash_strings
                for selected_index_list in selected_index_lists]

        best_index_lists, foldwise_best_predictions_list, \
            combined_predictions_list, ground_truth_valid_list = \
            zip(*list_of_tuples)

        # contributivity counts
        for best_index_list in best_index_lists:
            fold_counts = np.histogram(
                best_index_list, bins=range(models_df.shape[0] + 1))[0]
            normalized_counts += 1.0 * fold_counts / fold_counts.sum()
        normalized_counts = 100. * normalized_counts / normalized_counts.sum()
        normalized_counts[normalized_counts > 0] = np.maximum(
            1.0, normalized_counts[normalized_counts > 0])  # we have 1 for every model picked at least once
        normalized_counts += 0.4999
        integer_precentage_counts = normalized_counts.astype(int)

        if test:
            make_combined_test_prediction(
                best_index_lists, models_df, hash_strings)

    leaderboard = pd.DataFrame(
        {'contributivity': integer_precentage_counts}, index=models_df.index)
    return leaderboard.sort(columns=['contributivity'], ascending=False)
