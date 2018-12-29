from collections import defaultdict
import datetime
import logging
import os
import shutil

import numpy as np
import pandas as pd

from ..exceptions import DuplicateSubmissionError
from ..exceptions import MissingExtensionError
from ..exceptions import MissingSubmissionFileError
from ..exceptions import TooEarlySubmissionError
from ..exceptions import UnknownStateError

from ..model.submission import submission_states
from ..model import Submission
from ..model import SubmissionFile
from ..model import SubmissionFileTypeExtension
from ..model import SubmissionOnCVFold

from ._query import select_event_by_name
from ._query import select_event_team_by_name
from ._query import select_extension_by_name
from ._query import select_submissions_by_state
from ._query import select_submission_by_id
from ._query import select_submission_by_name
from ._query import select_submission_file_type_by_name
from ._query import select_team_by_name

STATES = submission_states.enums
logger = logging.getLogger('DATABASE')


# Add functions: add information to the database
# TODO: move the queries in "_query"
# TODO: there is nothing regarding leaderboard update
def add_submission(session, event_name, team_name, submission_name,
                   submission_path, submission_deployment_path, is_sandbox):
    """Create a submission in the database and returns an handle.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event associated to the submission.
    team_name : str
        The team associated to the submission.
    submission_name : str
        The name to give to the current submission.
    submission_path : str
        The path of the files associated to the current submission.
    submission_deployment_path:
        Path to the deployment RAMP submissions directory.
    is_sandbox : bool
        If the submission is the sandbox submission.

    Returns
    -------
    submission : :class:`rampdb.model.Submission`
        The newly created submission.
    """
    event = select_event_by_name(session, event_name)
    team = select_team_by_name(session, team_name)
    event_team = select_event_team_by_name(session, event_name, team_name)
    submission = (session.query(Submission)
                         .filter(Submission.name == submission_name)
                         .filter(Submission.event_team == event_team)
                         .one_or_none())

    # create a new submission
    if submission is None:
        all_submissions = (session.query(Submission)
                                  .filter(Submission.event_team == event_team)
                                  .order_by(Submission.submission_timestamp)
                                  .all())
        last_submission = None if not all_submissions else all_submissions[-1]
        # check for non-admin user if they wait enough to make a new submission
        if (team.admin.access_level != 'admin' and last_submission is not None
                and last_submission.is_not_sandbox):
            time_to_last_submission = (datetime.datetime.utcnow() -
                                       last_submission.submission_timestamp)
            min_resubmit_time = datetime.timedelta(
                seconds=event.min_duration_between_submissions)
            awaiting_time = int((min_resubmit_time - time_to_last_submission)
                                .total_seconds())
            if awaiting_time > 0:
                raise TooEarlySubmissionError(
                    'You need to wait {} more seconds until next submission'
                    .format(awaiting_time))

        submission = Submission(
            name=submission_name, event_team=event_team,
            path_ramp_submissions=submission_deployment_path,
            is_sandbox=is_sandbox, session=session
        )
        for cv_fold in event.cv_folds:
            submission_on_cv_fold = SubmissionOnCVFold(submission=submission,
                                                       cv_fold=cv_fold)
            session.add(submission_on_cv_fold)
        session.add(submission)

    # the submission already exist
    else:
        # We allow resubmit for new or failing submissions
        if (submission.is_not_sandbox and
                (submission.state == 'new' or submission.is_error)):
            submission.set_state('new')
            submission.submission_timestamp = datetime.datetime.utcnow()
            for submission_on_cv_fold in submission.on_cv_folds:
                submission_on_cv_fold.reset()
        else:
            error_msg = ('Submission "{}" of team "{}" at event "{}" exists '
                         'already'
                         .format(submission_name, team_name, event_name))
            raise DuplicateSubmissionError(error_msg)

    files_type_extension = [os.path.splitext(filename)
                            for filename in os.listdir(submission_path)]
    # filter the files which contain an extension
    # remove the dot of the extension.
    files_type_extension = [(filename, extension[1:])
                            for filename, extension in files_type_extension
                            if extension != '']

    for workflow_element in event.problem.workflow.elements:
        try:
            desposited_types, deposited_extensions = zip(
                *[(filename, extension)
                  for filename, extension in files_type_extension
                  if filename == workflow_element.name]
            )
        except ValueError as e:
            session.rollback()
            if 'values to unpack' in str(e):
                # no file matching the workflow element
                raise MissingSubmissionFileError(
                    'No file corresponding to the workflow element "{}"'
                    .format(workflow_element)
                    )
            raise

        # check that files have the correct extension ...
        for extension_name in deposited_extensions:
            extension = select_extension_by_name(session, extension_name)
            if extension is not None:
                break
        # ... otherwise we raise an error
        else:
            session.rollback()
            raise MissingExtensionError(
                'All extensions "{}" are unknown for the submission "{}".'
                .format(", ".join(deposited_extensions), submission_name)
            )

        # check if it is a resubmission
        submission_file = (session.query(SubmissionFile)
                                  .filter(SubmissionFile.workflow_element ==
                                          workflow_element)
                                  .filter(SubmissionFile.submission ==
                                          submission)
                                  .one_or_none())
        # TODO: handle if resubmitted file changed extension
        if submission_file is None:
            submission_file_type = select_submission_file_type_by_name(
                session, workflow_element.file_type
            )
            type_extension = \
                (session.query(SubmissionFileTypeExtension)
                        .filter(SubmissionFileTypeExtension.type ==
                                submission_file_type)
                        .filter(SubmissionFileTypeExtension.extension ==
                                extension)
                        .one())
            submission_file = SubmissionFile(
                submission=submission, workflow_element=workflow_element,
                submission_file_type_extension=type_extension
            )
            session.add(submission_file)

    # for remembering it in the sandbox view
    event_team.last_submission_name = submission_name
    session.commit()

    # TODO: test missing there for those update
    # TODO: add those functions back
    # update_leaderboards(event_name)
    # update_user_leaderboards(event_name, team.name)
    return submission


# Getter functions: get information from the database
def get_submissions(session, event_name, state='new'):
    """Get information about submissions from an event with a specific state
    optionally.

    The information for each dataset is the id, name of the submission, and
    the files associated with the submission.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
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

    submissions = select_submissions_by_state(session, event_name, state)

    if not submissions:
        return []

    submission_id = [sub.id for sub in submissions]
    submission_files = [sub.files for sub in submissions]
    submission_basename = [sub.basename for sub in submissions]
    filenames = [[f.path for f in files] for files in submission_files]
    return list(zip(submission_id, submission_basename, filenames))


def get_submission_by_id(session, submission_id):
    """Get a submission given its id.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
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
    submission = select_submission_by_id(session, submission_id)
    submission.event.name
    submission.team.name
    return submission


def get_submission_by_name(session, event_name, team_name, name):
    """Get a single submission filtering by event, team, and submission names.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
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
    submission = select_submission_by_name(session, event_name, team_name,
                                           name)
    submission.event.name
    submission.team.name
    return submission


def get_submission_state(session, submission_id):
    """Get the state of a submission given its id.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
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
    submission = select_submission_by_id(session, submission_id)
    return submission.state


def get_predictions(session, submission_id):
    """Get the predictions from the database of a submission.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    submission_id : int
        The id of the submission.

    Returns
    -------
    predictions : pd.DataFrame
        A pandas dataframe containing the predictions on each fold.
    """
    submission = select_submission_by_id(session, submission_id)
    results = defaultdict(list)
    for fold_id, cv_fold in enumerate(submission.on_cv_folds):
        results['fold'].append(fold_id)
        results['y_pred_train'].append(cv_fold.full_train_y_pred)
        results['y_pred_test'].append(cv_fold.test_y_pred)
    return pd.DataFrame(results).set_index('fold')


def get_time(session, submission_id):
    """Get the computation time for each fold of a submission.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    submission_id : int
        The id of the submission.

    Returns
    -------
    computation_time : pd.DataFrame
        A pandas dataframe containing the computation time of each fold.
    """
    submission = select_submission_by_id(session, submission_id)
    results = defaultdict(list)
    for fold_id, cv_fold in enumerate(submission.on_cv_folds):
        results['fold'].append(fold_id)
        for step in ('train', 'valid', 'test'):
            results[step].append(getattr(cv_fold, '{}_time'.format(step)))
    return pd.DataFrame(results).set_index('fold')


def get_scores(session, submission_id):
    """Get the scores for each fold of a submission.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    submission_id : int
        The id of the submission.

    Returns
    -------
    scores : pd.DataFrame
        A pandas dataframe containing the scores of each fold.
    """
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


def get_submission_max_ram(session, submission_id):
    """Get the max amount RAM used by a submission during processing.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    submission_id : int
        The id of the submission.

    Returns
    -------
    max_ram_mb : float
        The max amount of RAM in MB.
    """
    submission = select_submission_by_id(session, submission_id)
    return submission.max_ram


def get_submission_error_msg(session, submission_id):
    """Get the error message after that a submission failed to be processed.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    submission_id : int
        The id of the submission.

    Returns
    -------
    error_msg : str
        The error message.
    """
    submission = select_submission_by_id(session, submission_id)
    return submission.error_msg


# TODO: maybe we should move this function
def get_event_nb_folds(session, event_name):
    """Get the number of fold for a given event.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event name.

    Returns
    -------
    nb_folds : int
        The number of folds for a specific event.
    """
    event = select_event_by_name(session, event_name)
    return len(event.cv_folds)


# Setter functions: set information in the database
def set_submission_state(session, submission_id, state):
    """Set the set of a submission.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
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

    submission = select_submission_by_id(session, submission_id)
    submission.set_state(state)
    session.commit()


def set_predictions(session, submission_id, path_predictions):
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
    submission = select_submission_by_id(session, submission_id)
    for fold_id, cv_fold in enumerate(submission.on_cv_folds):
        path_results = os.path.join(path_predictions,
                                    'fold_{}'.format(fold_id))
        cv_fold.full_train_y_pred = np.load(
            os.path.join(path_results, 'y_pred_train.npz'))['y_pred']
        cv_fold.test_y_pred = np.load(
            os.path.join(path_results, 'y_pred_test.npz'))['y_pred']
    session.commit()


def set_time(session, submission_id, path_predictions):
    """Set the timing information in the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    submission_id : int
        The id of the submission.
    path_predictions : str
        The path where the results files are located.
    """
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


def set_scores(session, submission_id, path_predictions):
    """Set the scores in the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    submission_id : int
        The id of the submission.
    path_predictions : str
        The path where the results files are located.
    """
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


def set_submission_max_ram(session, submission_id, max_ram_mb):
    """Set the max amount RAM used by a submission during processing.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    submission_id : int
        The id of the submission.
    max_ram_mb : float
        The max amount of RAM in MB.
    """
    submission = select_submission_by_id(session, submission_id)
    submission.max_ram = max_ram_mb
    session.commit()


def set_submission_error_msg(session, submission_id, error_msg):
    """Set the error message after that a submission failed to be processed.

    Parameters
    ----------
    config : dict
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    submission_id : int
        The id of the submission.
    error_msg : str
        The error message.
    """
    submission = select_submission_by_id(session, submission_id)
    submission.error_msg = error_msg
    session.commit()


# Computing functions: old style functions when it was using some functionality
#  from the database itself.
def score_submission(session, submission_id):
    """Score a submission and change its state to 'scored'

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    submission_id : int
        submission id

    Raises
    ------
    ValueError :
        when the state of the submission is not 'tested'
        (only a submission with state 'tested' can be scored)
    """
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


def submit_starting_kits(session, event_name, team_name, path_submission,
                         path_ramp_submissions, sandbox_name):
    """Submit all starting kits for a given event.

    Some kits contain several starting kits. This function allows to submit
    all these kits at once for a specific user.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The name of the event.
    team_name : str
        The name of the team.
    path_submission : str
        The path of the files associated to the current submission.
    path_ramp_submissions : str
        Path to the deployment RAMP submissions directory.
    sandbox_name : str
        The name of the sandbox submission.
    """
    event = select_event_by_name(session, event_name=event_name)
    submission_names = os.listdir(path_submission)
    # we temporary bypass the limit time between two submissions
    min_duration_between_submissions = event.min_duration_between_submissions
    event.min_duration_between_submissions = 0
    for submission_name in submission_names:
        from_submission_path = os.path.join(path_submission, submission_name)
        # one of the starting kit is usually used a sandbox and we need to
        # change the name to not have any duplicate
        submission_name = (submission_name
                           if submission_name != sandbox_name
                           else submission_name + '_test')
        submission = add_submission(session, event_name, team_name,
                                    submission_name, from_submission_path,
                                    path_ramp_submissions,
                                    False)
        # copy the files
        if os.path.exists(submission.path):
            shutil.rmtree(submission.path)
        os.makedirs(submission.path)
        for filename in submission.f_names:
            shutil.copy2(src=os.path.join(from_submission_path, filename),
                         dst=os.path.join(submission.path, filename))
        logger.info('Copying the submission files into the deployment folder')
        logger.info('Adding {}'.format(submission))
    # revert the minimum duration between two submissions
    event.min_duration_between_submissions = min_duration_between_submissions
    session.commit()
