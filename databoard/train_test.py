# Author: Balazs Kegl
# License: BSD 3 clause

import os
import pickle
import timeit

from sklearn.externals.joblib import Parallel, delayed

# import config_databoard
from databoard.config import config_object
import databoard.config as config
import databoard.generic as generic
import databoard.machine_parallelism as machine_parallelism
from databoard.db.model import SubmissionOnCVFold

n_processes = config_object.n_cpus


def run_method_on_folds(submission, method, cv_folds, **kwargs):
    """Runs various combinations of train, validate, and test on all folds.
    If is_parallelize is True, it will launch the jobs in parallel on different
    cores.

    Parameters
    ----------
    method : the method to be run (train_valid_and_test_on_fold,
        train_and_valid_on_fold, test_on_fold)
    submission_path : of the form <root_path>/models/<team>/<submission_hash>
    cv : a list of pairs of training and validation indices, identifying the
        folds
    """
    title = '{0}_{1}'.format(submission.team.name, submission.name)

    n_processes = config.config_object.num_cpus

    if config.is_parallelize:
        # Should be re-tested
        if config.is_parallelize_across_machines:
            job_ids = set()
            for i, cv_fold in enumerate(cv_folds):
                job_id = machine_parallelism.put_job(
                    method, (cv_fold, submission.path),
                    title=title + "_cv{0}".format(i))
                job_ids.add(job_id)
            try:
                job_status = machine_parallelism.wait_for_jobs_and_get_status(
                    job_ids,
                    timeout=config.timeout_parallelize_across_machines,
                    finish_if_exception=True)
            except machine_parallelism.TimeoutError:
                raise
            for status in job_status.values():
                if isinstance(status, Exception):
                    raise status
        else:
            Parallel(n_jobs=n_processes, verbose=5)(
                delayed(method)(cv_fold, submission) for cv_fold in cv_folds)
    else:
        for cv_fold in cv_folds:
            method(cv_fold, submission)


def write_execution_time(f_name, time):
    with open(f_name, 'w') as f:
        f.write(str(time))  # writing running time


def pickle_trained_submission(f_name, trained_submission):
    try:
        with open(f_name, 'w') as f:
            pickle.dump(trained_submission, f)  # saving the submission
    except Exception as e:
        generic.logger.error("Cannot pickle trained submission\n{}".format(e))
        os.remove(f_name)


def train_on_fold(X_train, y_train, cv_fold, submission):
    """Trains the submission on a single fold. Wrapper around
    specific.train_submission().
    It requires specific to contain a train_submission function that takes the
    module_path, X_train, y_train, and a cv_is containing the train_train and
    valid_train indices. Most of the time it will simply train on
    X_train[valid_train] but in the case of time series it may do feature
    extraction on the full file (using always the past). Training time is
    measured and returned.

    Parameters
    ----------
    X_train, y_train: training input data and labels
    cv_is : a pair of indices (train_train_is, valid_train_is)
    submission_path : of the form <root_path>/submissions/<team>/<submission_hash>

    Returns
    -------
    trained_submission : the trained submission, to be fed to specific.test_submission
    train_time : the wall clock time of the train
    """
    specific = config.config_object.specific

    cv_hash = generic.get_cv_hash(cv_fold.test_is)

    generic.logger.info("Training on fold : %s" % cv_hash)

    # so to make it importable
    open(os.path.join(submission.path, "__init__.py"), 'a').close()
    module_path = generic.get_module_path(submission.path)

    start = timeit.default_timer()
    trained_submission = specific.train_submission(
        module_path, X_train, y_train, cv_fold.train_is)
    end = timeit.default_timer()
    train_time = end - start

    return trained_submission, train_time


def train_measure_and_pickle_on_fold(X_train, y_train, cv_fold, submission):
    """Calls train_on_fold() to train on fold, writes execution time in
    <submission_path>/train_time/<cv_hash>.npy and pickles the submission
    (if is_pickle_trained_submission) into submission_path>/submission/<cv_hash>.p

    Parameters
    ----------
    X_train, y_train: training input data and labels
    cv_is : a pair of indices (train_train_is, valid_train_is)
    submission_path : of the form <root_path>/submissions/<team>/<submission_hash>

    Returns
    -------
    trained_submission : the trained submission, to be fed to specific.test_submission
     """
    cv_hash = generic.get_cv_hash(cv_fold.train_is)

    trained_submission, train_time = train_on_fold(
        X_train, y_train, cv_fold, submission)
    write_execution_time(generic.get_train_time_f_name(
        submission.path, cv_hash), train_time)
    if config.is_pickle_trained_submission:
        pickle_trained_submission(generic.get_submission_f_name(
            submission.path, cv_hash), trained_submission)
    return trained_submission


def test_trained_submission(trained_submission, X, cv_fold=None):
    """Tests and times a trained submission on a fold. If cv_is is None, tests
    on the whole (holdout) set. Wrapper around specific.test_submission()

    Parameters
    ----------
    trained_submission : a trained submission, returned by specific.train_submission()
    X : input data
    cv_is : a pair of indices (train_train_is, valid_train_is)

    Returns
    -------
    test_submission_output : the output of the tested submission, returned by
        specific.test_submission
    test_time : the wall clock time of the test
    """
    specific = config.config_object.specific
    if cv_fold is None:  # test on all points
        from databoard.db.model import CVFold
        _, y_test = specific.get_test_data()
        cv_fold = CVFold(train_is=[], test_is=range(len(y_test)))
    start = timeit.default_timer()
    test_submission_output = specific.test_submission(
        trained_submission, X, cv_fold.test_is)
    end = timeit.default_timer()
    test_time = end - start
    return test_submission_output, test_time


def test_trained_submission_on_test(
    trained_submission, X_test, cv_hash, submission):
    """Tests a trained submission on (holdout) X_test and outputs
    the predictions into <submission_path>/test/<cv_hash>.npy.

    Parameters
    ----------
    trained_submission : a trained submission, returned by specific.train_submission()
    X_test : input (holdout test) data
    cv_hash : the fold identifier
    submission_path : of the form <root_path>/submissions/<team>/<submission_hash>
    """
    generic.logger.info("Testing on fold : %s" % cv_hash)
    # We ignore test time, it is measured when validating
    test_submission_output, _ = test_trained_submission(
        trained_submission, X_test)
    test_f_name = generic.get_test_f_name(submission.path, cv_hash)
    test_submission_output.save(test_f_name)


def test_trained_submission_and_measure_on_valid(
    trained_submission, X_train, cv_fold, submission):
    """Tests a trained submission on a validation fold represented by cv_is,
    outputs the predictions into <submission_path>/valid/<cv_hash>.npy and
    the validation time into <submission_path>/valid_time/<cv_hash>.npy.

    Parameters
    ----------
    trained_submission : a trained submission, returned by specific.train_submission()
    X_train : input (training) data
    cv_is : a pair of indices (train_train_is, valid_train_is)
    submission_path : of the form <root_path>/submissions/<team>/<submission_hash>
    """
    cv_hash = generic.get_cv_hash(cv_fold.train_is)

    generic.logger.info("Validating on fold : %s" % cv_hash)

    valid_submission_output, valid_time = test_trained_submission(
        trained_submission, X_train, cv_fold)
    valid_f_name = generic.get_valid_f_name(submission.path, cv_hash)
    valid_submission_output.save(valid_f_name)
    write_execution_time(generic.get_valid_time_f_name(
        submission.path, cv_hash), valid_time)


def test_on_fold(cv_fold, submission):
    """Tests a trained submission on a validation fold represented by cv_is.
    Reloads the data so safe in each thread. Tries to unpickle the trained
    submission, if can't, retrains. Called using run_on_folds().

    Parameters
    ----------
    cv_is : a pair of indices (train_train_is, valid_train_is)
    submission_path : of the form <root_path>/submissions/<team>/<submission_hash>
    """
    specific = config.config_object.specific

    X_train, y_train = specific.get_train_data()
    X_test, _ = specific.get_test_data()
    cv_hash = generic.get_cv_hash(cv_fold.train_is)

    try:
        generic.logger.info("Loading from pickle on fold : %s" % cv_hash)
        with open(generic.get_submission_f_name(
                submission.path, cv_hash), 'r') as f:
            trained_submission = pickle.load(f)
    except IOError:  # no pickled submission, retrain
        generic.logger.info("No pickle, retraining on fold : %s" % cv_hash)
        trained_submission = train_measure_and_pickle_on_fold(
            X_train, y_train, cv_fold, submission)

    test_trained_submission_on_test(
        trained_submission, X_test, cv_hash, submission)


def train_valid_and_test_on_fold(cv_fold, submission):
    specific = config.config_object.specific

    X_train, y_train = specific.get_train_data()
    X_test, _ = specific.get_test_data()

    trained_submission = SubmissionOnCVFold(
        submission=submission, cv_fold=cv_fold)
    from databoard.db.model import db
    db.session.add(trained_submission)
    trained_submission.train(X_train, y_train)


def train_valid_and_test_on_fold_old(cv_fold, submission):
    """Trains and validates a submission on a validation fold represented by
    cv_is, then tests it. Reloads the data so safe in each thread.  Called
    using run_on_folds().

    Parameters
    ----------
    cv_is : a pair of indices (train_train_is, valid_train_is)
    submission_path : of the form <root_path>/submissions/<team>/<submission_hash>
    """
    specific = config.config_object.specific

    X_train, y_train = specific.get_train_data()
    X_test, _ = specific.get_test_data()
    cv_hash = generic.get_cv_hash(cv_fold.train_is)

    trained_submission = train_measure_and_pickle_on_fold(
        X_train, y_train, cv_fold, submission)

    test_trained_submission_and_measure_on_valid(
        trained_submission, X_train, cv_fold, submission)

    test_trained_submission_on_test(
        trained_submission, X_test, cv_hash, submission)


def train_and_valid_on_fold(cv_fold, submission):
    """Trains and validates a submission on a validation fold represented by
    cv_is. Reloads the data so safe in each thread.  Called using
    run_on_folds().

    Parameters
    ----------
    cv_is : a pair of indices (train_train_is, valid_train_is)
    submission_path : of the form <root_path>/submissions/<team>/<submission_hash>
    """
    specific = config.config_object.specific

    X_train, y_train = specific.get_train_data()

    trained_submission = train_measure_and_pickle_on_fold(
        X_train, y_train, cv_fold, submission)

    test_trained_submission_and_measure_on_valid(
        trained_submission, X_train, cv_fold, submission)


def check_on_fold(cv_fold, submission):
    """Checks a submission. Called using run_on_folds().

    Parameters
    ----------
    cv_is : a pair of indices (train_train_is, valid_train_is)
    submission_path : of the form <root_path>/submissions/<team>/<submission_hash>
    """
    specific = config.config_object.specific

    X_check, y_check = specific.get_check_data()
    module_path = generic.get_module_path(submission.path)
    specific.check_submission(
        module_path, X_check, y_check, (cv_fold.train_is, cv_fold.test_is))
