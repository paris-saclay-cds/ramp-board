from __future__ import print_function, absolute_import

import os

import numpy as np

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
    """Submission lists

    Retrieve a list of submissions and their associated files
    depending on their current status

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
    """
    if state is not None and state not in STATES:
        raise UnknownStateError("Unrecognized state : '{}'".format(state))

    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)
        submissions = select_submissions_by_state(session, event_name, state)

        if not submissions:
            return []

        print(submissions)

        submission_id = [sub.id for sub in submissions]
        submission_files = [sub.files for sub in submissions]
        submission_basename = [sub.basename for sub in submissions]
        filenames = [[f.path for f in files] for files in submission_files]

    return list(zip(submission_id, submission_basename, filenames))


def get_submission_by_id(config, submission_id):
    """
    Get a `Submission` instance given a submission id

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.

    submission_id : int
        submission id

    Returns
    -------

    `Submission` instance
    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)
        submission = select_submission_by_id(session, submission_id)
        # force event name and team name to be cached
        submission.event.name
        submission.team.name
    return submission


def get_submission_by_name(config, event_name, team_name, name):
    """
    Get a submission by name

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    session :
        database connexion session
    event_name : str
        name of the RAMP event
    team_name : str
        name of the RAMP team
    name : str
        name of the submission

    Returns
    -------
    `Submission` instance

    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)
        submission = select_submission_by_name(
            session,
            event_name,
            team_name,
            name)
        # force event name and team name to be cached
        submission.event.name
        submission.team.name
    return submission


def set_submission_state(config, submission_id, state):
    """
    Modify the state of a submission in the RAMP database

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        id of the requested submission
    state : str
        new state of the submission

    Raises
    ------
    ValueError :
        when mandatory connexion parameters are missing from config
    UnknownStateError :
        when the requested state does not exist in the database

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
    """
    Modify the state of a submission in the RAMP database

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        id of the requested submission

    Raises
    ------
    ValueError :
        when mandatory connexion parameters are missing from config
    UnknownStateError :
        when the requested state does not exist in the database

    """
    db, Session = _setup_db(config)    # Connect to the dabase and perform action
    with db.connect() as conn:
        session = Session(bind=conn)
        submission = select_submission_by_id(session, submission_id)
    return submission.state


def set_predictions(config, submission_id, prediction_path, ext='npy'):
    """
    Insert predictions in the database after training/testing

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        id of the related submission
    prediction_path : str
        local path where predictions are saved.
        Should end with 'training_output'.
    ext : {'npy', 'npz', 'csv'}, optional
        extension of the saved prediction extension file (default is 'npy')

    Raises
    ------
    NotImplementedError :
        when the extension cannot be read properly

    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)

        for fold_id, cv_fold in enumerate(submission.on_cv_folds):
            cv_fold.full_train_y_pred = _load_submission(
                prediction_path, fold_id, 'train', ext)
            cv_fold.test_y_pred = _load_submission(
                prediction_path, fold_id, 'test', ext)
            cv_fold.train_time = _get_time(prediction_path, fold_id, 'train')
            cv_fold.valid_time = _get_time(prediction_path, fold_id, 'valid')
            cv_fold.test_time = _get_time(prediction_path, fold_id, 'test')
            cv_fold.state = 'tested'
            session.commit()

        submission.state = 'tested'
        session.commit()


def _load_submission(path, fold_id, typ, ext):
    """
    Prediction loader method

    Parameters
    ----------
    path : str
        local path where predictions are saved
    fold_id : int
        id of the current CV fold
    type : {'train', 'test'}
        type of prediction
    ext : {'npy', 'npz', 'csv'}
        extension of the saved prediction extension file

    Raises
    ------
    ValueError :
        when typ is neither 'train' nor 'test'
    NotImplementedError :
        when the extension cannot be read properly

    """
    pred_file = os.path.join(path,
                             'fold_{}'.format(fold_id),
                             'y_pred_{}.{}'.format(typ, ext))

    if typ not in ['train', 'test']:
        raise ValueError("Only 'train' or 'test' are expected for arg 'typ'")

    if ext.lower() in ['npy', 'npz']:
        return np.load(pred_file)['y_pred']
    elif ext.lower() == 'csv':
        # TODO: there is a bug here. This function does not exist.
        return np.loadfromtxt(pred_file)
    else:
        return NotImplementedError("No reader implemented for extension {ext}"
                                   .format(ext))


def _get_time(path, fold_id, typ):
    """
    get time duration in seconds of train or valid
    or test for a given fold.

    Parameters
    ----------
    path : str
        local path where predictions are saved
    fold_id : int
        id of the current CV fold
    typ : {'train', 'valid, 'test'}

    Raises
    ------
    ValueError :
        when typ is neither is not 'train' or 'valid' or test'
    """
    if typ not in ['train', 'valid', 'test']:
        raise ValueError(
            "Only 'train' or 'valid' or 'test' are expected for arg 'typ'")
    time_file = os.path.join(path, 'fold_{}'.format(fold_id), typ + '_time')
    return float(open(time_file).read())


def score_submission(config, submission_id):
    """
    Score a submission and change its state to 'scored'

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
            submission_on_cv_fold.compute_train_scores(session)
            submission_on_cv_fold.compute_valid_scores(session)
            submission_on_cv_fold.compute_test_scores(session)
            submission_on_cv_fold.state = 'scored'
        session.commit()
        submission.compute_test_score_cv_bag(session)
        submission.compute_valid_score_cv_bag(session)
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
    """
    Modify the max RAM mb usage of a submission

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        id of the requested submission
    max_ram_mb : float
        max ram usage in MB
    """
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)
        submission.max_ram = max_ram_mb
        session.commit()


def set_submission_error_msg(config, submission_id, error_msg):
    """
    Set submission message error

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.
    submission_id : int
        id of the requested submission
    error_msg : str
        message error
    """

    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)

        submission = select_submission_by_id(session, submission_id)
        submission.error_msg = error_msg
        session.commit()


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
