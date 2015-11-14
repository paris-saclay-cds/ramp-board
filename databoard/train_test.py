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

n_processes = config_object.num_cpus


def run_on_folds_new(method, submission_path, cv, **kwargs):
    """Runs various combinations of train, validate, and test on all folds.
    If is_parallelize is True, it will launch the jobs in parallel on different
    cores.

    Parameters
    ----------
    method : the method to be run (train_valid_and_test_on_fold,
        train_and_valid_on_fold, test_on_fold)
    submission_path : of the form <root_path>/models/<team>/<model_hash>
    cv : a list of pairs of training and validation indices, identifying the
        folds
    """

    if 'team_name' in kwargs and 'submission_name' in kwargs:
        title = '{0}_{1}'.format(
            kwargs['team_name'], kwargs['submission_name'])
    else:
        title = submission_path.replace("/", "_")

    n_processes = config.config_object.num_cpus

    if config.is_parallelize:
        # Should be re-tested
        if config.is_parallelize_across_machines:
            job_ids = set()
            for i, cv_is in enumerate(cv):
                job_id = machine_parallelism.put_job(
                    method, (cv_is, submission_path),
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
                delayed(method)(cv_is, submission_path) for cv_is in cv)
    else:
        for cv_is in cv:
            method(cv_is, submission_path)


def run_on_folds(method, full_model_path, cv, **kwargs):
    """Runs various combinations of train, validate, and test on all folds.
    If is_parallelize is True, it will launch the jobs in parallel on different
    cores.

    Parameters
    ----------
    method : the method to be run (train_valid_and_test_on_fold,
        train_and_valid_on_fold, test_on_fold)
    full_model_path : of the form <root_path>/models/<team>/<model_hash>
    cv : a list of pairs of training and validation indices, identifying the
        folds
    """

    if "team" in kwargs and "tag" in kwargs:
        title = "{0}_{1}".format(kwargs["team"], kwargs["tag"])
    else:
        title = full_model_path.replace("/", "_")

    if config.is_parallelize:
        if config.is_parallelize_across_machines:
            job_ids = set()
            for i, cv_is in enumerate(cv):
                job_id = machine_parallelism.put_job(
                    method, (cv_is, full_model_path),
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
                delayed(method)(cv_is, full_model_path) for cv_is in cv)
    else:
        for cv_is in cv:
            method(cv_is, full_model_path)


def write_execution_time(f_name, time):
    with open(f_name, 'w') as f:
        f.write(str(time))  # writing running time


def pickle_trained_model(f_name, trained_model):
    try:
        with open(f_name, 'w') as f:
            pickle.dump(trained_model, f)  # saving the model
    except Exception as e:
        generic.logger.error("Cannot pickle trained model\n{}".format(e))
        os.remove(f_name)


def train_on_fold(X_train, y_train, cv_is, full_model_path):
    """Trains the model on a single fold. Wrapper around
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
    full_model_path : of the form <root_path>/models/<team>/<model_hash>

    Returns
    -------
    trained_model : the trained model, to be fed to specific.test_submission
    train_time : the wall clock time of the train
    """
    specific = config.config_object.specific

    valid_train_is, _ = cv_is
    cv_hash = generic.get_cv_hash(valid_train_is)

    generic.logger.info("Training on fold : %s" % cv_hash)

    # so to make it importable
    open(os.path.join(full_model_path, "__init__.py"), 'a').close()
    module_path = generic.get_module_path(full_model_path)

    start = timeit.default_timer()
    trained_model = specific.train_submission(
        module_path, X_train, y_train, cv_is)
    end = timeit.default_timer()
    train_time = end - start

    return trained_model, train_time


def train_measure_and_pickle_on_fold(X_train, y_train, cv_is, full_model_path):
    """Calls train_on_fold() to train on fold, writes execution time in
    <full_model_path>/train_time/<cv_hash>.npy and pickles the model
    (if is_pickle_trained_model) into full_model_path>/model/<cv_hash>.p

    Parameters
    ----------
    X_train, y_train: training input data and labels
    cv_is : a pair of indices (train_train_is, valid_train_is)
    full_model_path : of the form <root_path>/models/<team>/<model_hash>

    Returns
    -------
    trained_model : the trained model, to be fed to specific.test_submission
     """
    valid_train_is, _ = cv_is
    cv_hash = generic.get_cv_hash(valid_train_is)

    trained_model, train_time = train_on_fold(
        X_train, y_train, cv_is, full_model_path)
    write_execution_time(generic.get_train_time_f_name(
        full_model_path, cv_hash), train_time)
    if config.is_pickle_trained_model:
        pickle_trained_model(generic.get_model_f_name(
            full_model_path, cv_hash), trained_model)
    return trained_model


def test_trained_model(trained_model, X, cv_is=None):
    """Tests and times a trained model on a fold. If cv_is is None, tests
    on the whole (holdout) set. Wrapper around specific.test_submission()

    Parameters
    ----------
    trained_model : a trained model, returned by specific.train_submission()
    X : input data
    cv_is : a pair of indices (train_train_is, valid_train_is)

    Returns
    -------
    test_submission_output : the output of the tested model, returned by
        specific.test_submission
    test_time : the wall clock time of the test
    """
    specific = config.config_object.specific
    if cv_is is None:
        _, y_test = specific.get_test_data()
        cv_is = ([], range(len(y_test)))  # test on all points
    start = timeit.default_timer()
    test_submission_output = specific.test_submission(trained_model, X, cv_is)
    end = timeit.default_timer()
    test_time = end - start
    return test_submission_output, test_time


def test_trained_model_on_test(trained_model, X_test, cv_hash,
                               full_model_path):
    """Tests a trained model on (holdout) X_test and outputs
    the predictions into <full_model_path>/test/<cv_hash>.npy.

    Parameters
    ----------
    trained_model : a trained model, returned by specific.train_submission()
    X_test : input (holdout test) data
    cv_hash : the fold identifier
    full_model_path : of the form <root_path>/models/<team>/<model_hash>
    """
    generic.logger.info("Testing on fold : %s" % cv_hash)
    # We ignore test time, it is measured when validating
    test_submission_output, _ = test_trained_model(trained_model, X_test)
    test_f_name = generic.get_test_f_name(full_model_path, cv_hash)
    test_submission_output.save(test_f_name)


def test_trained_model_and_measure_on_valid(trained_model, X_train,
                                            cv_is, full_model_path):
    """Tests a trained model on a validation fold represented by cv_is,
    outputs the predictions into <full_model_path>/valid/<cv_hash>.npy and
    the validation time into <full_model_path>/valid_time/<cv_hash>.npy.

    Parameters
    ----------
    trained_model : a trained model, returned by specific.train_submission()
    X_train : input (training) data
    cv_is : a pair of indices (train_train_is, valid_train_is)
    full_model_path : of the form <root_path>/models/<team>/<model_hash>
    """
    valid_train_is, _ = cv_is
    cv_hash = generic.get_cv_hash(valid_train_is)

    generic.logger.info("Validating on fold : %s" % cv_hash)

    valid_model_output, valid_time = test_trained_model(
        trained_model, X_train, cv_is)
    valid_f_name = generic.get_valid_f_name(full_model_path, cv_hash)
    valid_model_output.save(valid_f_name)
    write_execution_time(generic.get_valid_time_f_name(
        full_model_path, cv_hash), valid_time)


def test_on_fold(cv_is, full_model_path):
    """Tests a trained model on a validation fold represented by cv_is.
    Reloads the data so safe in each thread. Tries to unpickle the trained
    model, if can't, retrains. Called using run_on_folds().

    Parameters
    ----------
    cv_is : a pair of indices (train_train_is, valid_train_is)
    full_model_path : of the form <root_path>/models/<team>/<model_hash>
    """
    specific = config.config_object.specific

    X_train, y_train = specific.get_train_data()
    X_test, _ = specific.get_test_data()
    valid_train_is, _ = cv_is
    cv_hash = generic.get_cv_hash(valid_train_is)

    try:
        generic.logger.info("Loading from pickle on fold : %s" % cv_hash)
        with open(generic.get_model_f_name(
                full_model_path, cv_hash), 'r') as f:
            trained_model = pickle.load(f)
    except IOError:  # no pickled model, retrain
        generic.logger.info("No pickle, retraining on fold : %s" % cv_hash)
        trained_model = train_measure_and_pickle_on_fold(
            X_train, y_train, cv_is, full_model_path)

    test_trained_model_on_test(
        trained_model, X_test, cv_hash, full_model_path)


def train_valid_and_test_on_fold(cv_is, full_model_path):
    """Trains and validates a model on a validation fold represented by
    cv_is, then tests it. Reloads the data so safe in each thread.  Called
    using run_on_folds().

    Parameters
    ----------
    cv_is : a pair of indices (train_train_is, valid_train_is)
    full_model_path : of the form <root_path>/models/<team>/<model_hash>
    """
    specific = config.config_object.specific

    X_train, y_train = specific.get_train_data()
    X_test, _ = specific.get_test_data()
    valid_train_is, valid_test_is = cv_is
    cv_hash = generic.get_cv_hash(valid_train_is)

    trained_model = train_measure_and_pickle_on_fold(
        X_train, y_train, cv_is, full_model_path)

    test_trained_model_and_measure_on_valid(
        trained_model, X_train, cv_is, full_model_path)

    test_trained_model_on_test(
        trained_model, X_test, cv_hash, full_model_path)


def train_and_valid_on_fold(cv_is, full_model_path):
    """Trains and validates a model on a validation fold represented by
    cv_is. Reloads the data so safe in each thread.  Called using
    run_on_folds().

    Parameters
    ----------
    cv_is : a pair of indices (train_train_is, valid_train_is)
    full_model_path : of the form <root_path>/models/<team>/<model_hash>
    """
    specific = config.config_object.specific

    X_train, y_train = specific.get_train_data()

    trained_model = train_measure_and_pickle_on_fold(
        X_train, y_train, cv_is, full_model_path)

    test_trained_model_and_measure_on_valid(
        trained_model, X_train, cv_is, full_model_path)


def check_on_fold(cv_is, full_model_path):
    """Checks a model. Called using run_on_folds().

    Parameters
    ----------
    cv_is : a pair of indices (train_train_is, valid_train_is)
    full_model_path : of the form <root_path>/models/<team>/<model_hash>
    """
    specific = config.config_object.specific

    X_check, y_check = specific.get_check_data()
    module_path = generic.get_module_path(full_model_path)
    specific.check_model(module_path, X_check, y_check, cv_is)


def run_models(orig_models_df, infinitive, past_participle, gerund, error_state,
               method):
    """The master method that runs different pipelines (train+valid,
    train+valid+test, test).

    Parameters
    ----------
    orig_models_df : the table of the models that should be run
    infinitive, past_participle, gerund : three forms of the action naming
        to be run. Like train, trained, training. Besides message strings,
        past_participle is used for the final state of a successful run
        (trained, tested)
    error_state : the state we get in after an unsuccesful run. The error
        message is saved in <error_state>.txt, to be rendered on the web site
    method : the method to be run (train_and_valid_on_fold,
        train_valid_and_test_on_fold, test_on_fold)
    """
    specific = config.config_object.specific

    models_df = orig_models_df.sort("timestamp")

    if models_df.shape[0] == 0:
        generic.logger.info("No models to {}.".format(infinitive))
        return

    generic.logger.info("Reading data")
    X_train, y_train = specific.get_train_data()
    cv = specific.get_cv(y_train)

    for idx, model_df in models_df.iterrows():
        if model_df['state'] in ["ignore"]:
            continue

        full_model_path = generic.get_full_model_path(idx, model_df)

        generic.logger.info("{} : {}/{}".format(
            str.capitalize(gerund), model_df['team'], model_df['model']))

        try:
            run_on_folds(
                method, full_model_path, cv, team=model_df["team"], tag=model_df["model"])
            # failed_models.drop(idx, axis=0, inplace=True)
            orig_models_df.loc[idx, 'state'] = past_participle
        except Exception, e:
            orig_models_df.loc[idx, 'state'] = error_state
            if hasattr(e, "traceback"):
                msg = str(e.traceback)
            else:
                msg = repr(e)
            generic.logger.error("{} failed with exception: \n{}".format(
                str.capitalize(gerund), msg))

            # TODO: put the error in the database instead of a file
            # Keep the model folder clean.
            with open(generic.get_f_name(full_model_path, '.', error_state, "txt"), 'w') as f:
                error_msg = msg
                cut_exception_text = error_msg.rfind('--->')
                if cut_exception_text > 0:
                    error_msg = error_msg[cut_exception_text:]
                f.write("{}".format(error_msg))


def train_and_valid_models(orig_models_df):
    run_models(orig_models_df, "train", "trained", "training", "error",
               train_and_valid_on_fold)


def train_valid_and_test_submissions(orig_models_df):
    run_models(orig_models_df, "train/test", "tested", "training/testing", "error",
               train_valid_and_test_on_fold)


def test_submissions(orig_models_df):
    run_models(orig_models_df, "test", "tested", "testing", "test_error",
               test_on_fold)


def check_models(orig_models_df):
    run_models(orig_models_df, "check", "new", "checking", "error",
               check_on_fold)
