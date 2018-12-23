from collections import defaultdict
import os

import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine.url import URL

from ..exceptions import UnknownStateError
from ..model import Model
from ..model.submission import submission_states
from .query import select_submissions_by_state
from .query import select_submission_by_id
from .query import select_submission_by_name
from .query import select_event_by_name

STATES = submission_states.enums


def _setup_db(config):
    """Get the necessary handler to manipulate the database.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.

    Returns
    -------
    db : sqlalchemy.Engine
        The engine to connect to the database.
    Session : sqlalchemy.orm.Session
        Configured Session class which can later be used to communicate with
        the database.
    """
    # create the URL from the configuration
    db_url = URL(**config)
    db = create_engine(db_url)
    Session = sessionmaker(db)
    # Link the relational model to the database
    Model.metadata.create_all(db)

    return db, Session

def get_submissions(config, event_name, state='new'):
    """Get information about submissions from an event with a specific state
    optionally.

    The information for each dataset is the id, name of the submission, and
    the files associated with the submission.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    event_name : str
        The name of the RAMP event.
    state : None or str, default='new'
        The state of the requested submissions. If None, the state of the
        submissions will be ignored and all submissions for an event will be
        fetched.

    Returns
    -------
    submissions_info : list of tuple(int, str, list of str)
        List of submissions information. Each item is a tuple containing:

        * an integer containing the id of the submission;
        * a string with the name of the submission in the database;
        * a list of string representing the file associated with the
          submission.

    See also
    --------
    rampdb.tools.get_submission_by_id : Get a single submission using an id.
    rampdb.tools.get_submission_by_name : Get a single submission using names.
    """
    if state is not None and state not in STATES:
        raise UnknownStateError("Unrecognized state : '{}'".format(state))

    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)
        submissions = select_submissions_by_state(session, event_name, state)

        if not submissions:
            return []

        submission_id = [sub.id for sub in submissions]
        submission_files = [sub.files for sub in submissions]
        submission_basename = [sub.basename for sub in submissions]
        filenames = [[f.path for f in files] for files in submission_files]
    return list(zip(submission_id, submission_basename, filenames))


def get_submission_by_id(config, submission_id):
    """Get a submission given its id.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        The id of the submission to query

    Returns
    -------
    submission : :class:`rampdb.model.Submission`
        The queried submission.

    See also
    --------
    rampdb.tools.get_submissions : Get submissions information.
    rampdb.tools.get_submission_by_name : Get a single submission using names.
    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)
        submission = select_submission_by_id(session, submission_id)
        submission.event.name
        submission.team.name
    return submission


def get_submission_by_name(config, event_name, team_name, name):
    """Get a single submission filtering by event, team, and submission names.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    event_name : str
        The RAMP event.
    team_name : str
        The name of the team.
    name : str
        The name of the submission.

    Returns
    -------
    submission : :class:`rampdb.model.Submission`
        The queried submission.

    See also
    --------
    rampdb.tools.get_submissions : Get submissions information.
    rampdb.tools.get_submission_by_id : Get a single submission using an id.
    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)
        submission = select_submission_by_name(
            session,
            event_name,
            team_name,
            name)
        submission.event.name
        submission.team.name
    return submission


def set_submission_state(config, submission_id, state):
    """Set the set of a submission.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        id of the requested submission
    state : str
        The state of the submission. The possible states for a submission are:

        * 'new': submitted by user to the web interface;
        * 'sent_to_training': submission was send for training but not launch
          yet.
        * 'trained': training finished normally;
        * 'training_error': training finished abnormally;
        * 'validated': validation finished normally;
        * 'validating_error': validation finished abnormally;
        * 'tested': testing finished normally;
        * 'testing_error': testing finished abnormally;
        * 'training': training is running normally;
        * 'scored': submission scored.
    """
    if state not in STATES:
        raise UnknownStateError("Unrecognized state : '{}'".format(state))

    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)
        submission.set_state(state)

        session.commit()


def get_submission_state(config, submission_id):
    """Get the state of a submission given its id.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        id of the requested submission

    Returns
    -------
    submission_state : str
        The state associated with the submission.

    Notes
    -----
    The possible states for a submission are:

    * 'new': submitted by user to the web interface;
    * 'sent_to_training': submission was send for training but not launch
        yet.
    * 'trained': training finished normally;
    * 'training_error': training finished abnormally;
    * 'validated': validation finished normally;
    * 'validating_error': validation finished abnormally;
    * 'tested': testing finished normally;
    * 'testing_error': testing finished abnormally;
    * 'training': training is running normally;
    * 'scored': submission scored.
    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)
        submission = select_submission_by_id(session, submission_id)
    return submission.state


def set_predictions(config, submission_id, path_predictions):
    """Set the predictions in the database.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        The id of the submission.
    path_predictions : str
        The path where the results files are located.
    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)
        for fold_id, cv_fold in enumerate(submission.on_cv_folds):
            path_results = os.path.join(path_predictions,
                                        'fold_{}'.format(fold_id))
            cv_fold.full_train_y_pred = np.load(
                os.path.join(path_results, 'y_pred_train.npz'))['y_pred']
            cv_fold.test_y_pred = np.load(
                os.path.join(path_results, 'y_pred_test.npz'))['y_pred']
        session.commit()


def get_predictions(config, submission_id):
    """Get the predictions from the database of a submission.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        The id of the submission.

    Returns
    -------
    predictions : pd.DataFrame
        A pandas dataframe containing the predictions on each fold.
    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)
        results = defaultdict(list)
        for fold_id, cv_fold in enumerate(submission.on_cv_folds):
            results['fold'].append(fold_id)
            results['y_pred_train'].append(cv_fold.full_train_y_pred)
            results['y_pred_test'].append(cv_fold.test_y_pred)
        return pd.DataFrame(results).set_index('fold')


def set_time(config, submission_id, path_predictions):
    """Set the timing information in the database.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        The id of the submission.
    path_predictions : str
        The path where the results files are located.
    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)
        for fold_id, cv_fold in enumerate(submission.on_cv_folds):
            path_results = os.path.join(path_predictions,
                                        'fold_{}'.format(fold_id))
            results = {}
            for step in ('train', 'valid', 'test'):
                results[step + '_time'] = np.asscalar(
                    np.loadtxt(os.path.join(path_results, step + '_time'))
                )
            for key, value in results.items():
                setattr(cv_fold, key, value)
        session.commit()


def get_time(config, submission_id):
    """Get the computation time for each fold of a submission.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        The id of the submission.

    Returns
    -------
    computation_time : pd.DataFrame
        A pandas dataframe containing the computation time of each fold.
    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)
        results = defaultdict(list)
        for fold_id, cv_fold in enumerate(submission.on_cv_folds):
            results['fold'].append(fold_id)
            for step in ('train', 'valid', 'test'):
                results[step].append(getattr(cv_fold, '{}_time'.format(step)))
        return pd.DataFrame(results).set_index('fold')


def set_scores(config, submission_id, path_predictions):
    """Set the scores in the database.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        The id of the submission.
    path_predictions : str
        The path where the results files are located.
    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)
        for fold_id, cv_fold in enumerate(submission.on_cv_folds):
            path_results = os.path.join(path_predictions,
                                        'fold_{}'.format(fold_id))
            scores_update = pd.read_csv(
                os.path.join(path_results, 'scores.csv'), index_col=0
            )
            for score in cv_fold.scores:
                for step in scores_update.index:
                    value = scores_update.loc[step, score.name]
                    setattr(score, step + '_score', value)
        session.commit()


def get_scores(config, submission_id):
    """Get the scores for each fold of a submission.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        The id of the submission.

    Returns
    -------
    scores : pd.DataFrame
        A pandas dataframe containing the scores of each fold.
    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)
        results = defaultdict(list)
        index = []
        for fold_id, cv_fold in enumerate(submission.on_cv_folds):
            for step in ('train', 'valid', 'test'):
                index.append((fold_id, step))
                for score in cv_fold.scores:
                    results[score.name].append(getattr(score, step + '_score'))
        multi_index = pd.MultiIndex.from_tuples(index, names=['fold', 'step'])
        scores = pd.DataFrame(results, index=multi_index)
        return scores


def score_submission(config, submission_id):
    """Score a submission and change its state to 'scored'

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        submission id

    Raises
    ------
    ValueError :
        when the state of the submission is not 'tested'
        (only a submission with state 'tested' can be scored)
    """

    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)
        if submission.state != 'tested':
            raise ValueError('Submission state must be "tested"'
                             ' to score, not "{}"'.format(submission.state))

        # We are conservative:
        # only score if all stages (train, test, validation)
        # were completed. submission_on_cv_fold compute scores can be called
        # manually if needed for submission in various error states.
        for submission_on_cv_fold in submission.on_cv_folds:
            submission_on_cv_fold.session = session
            submission_on_cv_fold.compute_train_scores()
            submission_on_cv_fold.compute_valid_scores()
            submission_on_cv_fold.compute_test_scores()
            submission_on_cv_fold.state = 'scored'
        session.commit()
        # TODO: We are not managing the bagged score.
        # submission.compute_test_score_cv_bag(session)
        # submission.compute_valid_score_cv_bag(session)
        # Means and stds were constructed on demand by fetching fold times.
        # It was slow because submission_on_folds contain also possibly large
        # predictions. If postgres solves this issue (which can be tested on
        # the mean and std scores on the private leaderbord), the
        # corresponding columns (which are now redundant) can be deleted in
        # Submission and this computation can also be deleted.
        submission.train_time_cv_mean = np.mean(
            [ts.train_time for ts in submission.on_cv_folds])
        submission.valid_time_cv_mean = np.mean(
            [ts.valid_time for ts in submission.on_cv_folds])
        submission.test_time_cv_mean = np.mean(
            [ts.test_time for ts in submission.on_cv_folds])
        submission.train_time_cv_std = np.std(
            [ts.train_time for ts in submission.on_cv_folds])
        submission.valid_time_cv_std = np.std(
            [ts.valid_time for ts in submission.on_cv_folds])
        submission.test_time_cv_std = np.std(
            [ts.test_time for ts in submission.on_cv_folds])
        submission.state = 'scored'
        session.commit()


def set_submission_max_ram(config, submission_id, max_ram_mb):
    """Set the max amount RAM used by a submission during processing.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        The id of the submission.
    max_ram_mb : float
        The max amount of RAM in MB.
    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)
        submission.max_ram = max_ram_mb
        session.commit()


def get_submission_max_ram(config, submission_id):
    """Get the max amount RAM used by a submission during processing.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        The id of the submission.

    Returns
    -------
    max_ram_mb : float
        The max amount of RAM in MB.
    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)
        return submission.max_ram


def set_submission_error_msg(config, submission_id, error_msg):
    """Set the error message after that a submission failed to be processed.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        The id of the submission.
    error_msg : str
        The error message.
    """

    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)
        submission.error_msg = error_msg
        session.commit()


def get_submission_error_msg(config, submission_id):
    """Get the error message after that a submission failed to be processed.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        The id of the submission.

    Returns
    -------
    error_msg : str
        The error message.
    """

    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)
        return submission.error_msg


def get_event_nb_folds(config, event_name):
    """Get the number of fold for a given event.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    event_name : str
        The event name.

    Returns
    -------
    nb_folds : int
        The number of folds for a specific event.
    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)
        event = select_event_by_name(session, event_name)
        return len(event.cv_folds)
