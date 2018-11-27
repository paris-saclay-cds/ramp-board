import datetime
import imp
import logging
import os
import shutil
import time
import timeit

import numpy as np
import pandas as pd
# temporary fix for importing torch before sklearn
# import torch  # noqa
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.validation import assert_all_finite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

from . import app, db, ramp_config, ramp_kits_path
from .model import (CVFold, DetachedSubmissionOnCVFold,
                    DuplicateSubmissionError, Event, EventAdmin,
                    EventScoreType, EventTeam, Extension, Keyword,
                    MissingExtensionError, MissingSubmissionFileError,
                    NameClashError, Problem, ProblemKeyword, Submission,
                    SubmissionFile, SubmissionFileType,
                    SubmissionFileTypeExtension, SubmissionOnCVFold,
                    SubmissionSimilarity, Team, TooEarlySubmissionError, User,
                    UserInteraction, Workflow, WorkflowElement,
                    WorkflowElementType)
from .utils import (date_time_format, get_hashed_password, remove_non_ascii,
                    send_mail, table_format, encode_string)
from .utils import import_module_from_source

logger = logging.getLogger('databoard')
pd.set_option('display.max_colwidth', -1)  # cause to_html truncates the output


def get_team_members(team):
    # This works only if no team mergers. The commented code below
    # is general but slow.
    yield team.admin
    # if team.initiator is not None:
    #     # "yield from" in Python 3.3
    #     for member in get_team_members(team.initiator):
    #         yield member
    #     for member in get_team_members(team.acceptor):
    #         yield member
    # else:
    #     yield team.admin


# def get_user_teams(user):
#     # This works only if no team mergers. The commented code below
#     # is general but slow.
#     team = Team.query.filter_by(name=user.name).one()
#     yield team
#     # teams = Team.query.all()
#     # for team in teams:
#     #     if user in get_team_members(team):
#     #         yield team


def get_user_event_teams(event_name, user_name):
    # This works only if no team mergers. The commented code below
    # is general but slow.
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=user_name).one()
    event_team = EventTeam.query.filter_by(
        event=event, team=team).one_or_none()
    if event_team is not None:
        yield event_team
    # event = Event.query.filter_by(name=event_name).one()
    # user = User.query.filter_by(name=user_name).one()
    # event_teams = EventTeam.query.filter_by(event=event).all()
    # for event_team in event_teams:
    #     if user in get_team_members(event_team.team):
    #         yield event_team


# def get_n_user_teams(user):
#     return len(get_user_teams(user))


def get_active_user_event_team(event, user):
    # There should always be an active user team, if not, throw an exception
    # The current code works only if each user admins a single team.
    event_team = EventTeam.query.filter_by(
        event=event, team=user.admined_teams[0]).one_or_none()
    return event_team

    # This below works for the general case with teams with more than
    # on members but it is slow, eg in constructing user interactions
    # event_teams = EventTeam.query.filter_by(event=event).all()
    # for event_team in event_teams:
    #     if user in get_team_members(event_team.team) and
    #             event_team.is_active:
    #         return event_team


def combine_predictions_list(predictions_list, index_list=None):
    """Combine predictions in predictions_list[index_list].

    By taking the mean of their get_combineable_predictions views.

    E.g. for regression it is the actual
    predictions, and for classification it is the probability array (which
    should be calibrated if we want the best performance). Called both for
    combining one submission on cv folds (a single model that is trained on
    different folds) and several models on a single fold.
    Called by
    _get_bagging_score : which combines bags of the same model, trained on
        different folds, on the heldout test set
    _get_cv_bagging_score : which combines cv-bags of the same model, trained
        on different folds, on the training set
    get_next_best_single_fold : which does one step of the greedy forward
        selection (of different models) on a single fold
    _get_combined_predictions_single_fold : which does the full loop of greedy
        forward selection (of different models), until improvement, on a single
        fold
    _get_combined_test_predictions_single_fold : which computes the combination
        (constructed on the cv valid set) on the holdout test set, on a single
        fold
    _get_combined_test_predictions : which combines the foldwise combined
        and foldwise best test predictions into a single megacombination

    Parameters
    ----------
    predictions_list : list of instances of Predictions
        Each element of the list is an instance of Predictions of a given model
        on the same data points.
    index_list : None | list of integers
        The subset of predictions to be combined. If None, the full set is
        combined.

    Returns
    -------
    combined_predictions : instance of Predictions
        A predictions instance containing the combined (averaged) predictions.
    """
    Predictions = type(predictions_list[0])
    combined_predictions = Predictions.combine(predictions_list, index_list)
    return combined_predictions


def _get_score_cv_bags(event, score_type, predictions_list, ground_truths,
                       test_is_list=None):
    """
    Compute the bagged score of the predictions in predictions_list.

    Called by Submission.compute_valid_score_cv_bag and
    db_tools.compute_contributivity.

    Parameters
    ----------
    event : instance of Event
        Needed for the type of y_comb and
    predictions_list : list of instances of Predictions
    ground_truths : instance of Predictions
    test_is_list : list of integers
        Indices of points that should be bagged in each prediction. If None,
        the full prediction vectors will be bagged.
    Returns
    -------
    score_cv_bags : instance of Score ()
    """
    if test_is_list is None:  # we combine the full list
        test_is_list = [range(len(predictions.y_pred))
                        for predictions in predictions_list]

    y_comb = np.array(
        [event.Predictions(n_samples=len(ground_truths.y_pred))
         for _ in predictions_list])
    score_cv_bags = []
    for i, test_is in enumerate(test_is_list):
        y_comb[i].set_valid_in_train(predictions_list[i], test_is)
        combined_predictions = combine_predictions_list(y_comb[:i + 1])
        valid_indexes = combined_predictions.valid_indexes
        score_cv_bags.append(score_type.score_function(
            ground_truths, combined_predictions, valid_indexes))
        # XXX maybe use masked arrays rather than passing valid_indexes
    return combined_predictions, score_cv_bags


def get_next_best_single_fold(event, predictions_list, ground_truths,
                              best_index_list, min_improvement=0.0):
    """.

    Find the model that minimizes the score if added to
    predictions_list[best_index_list] using event.official_score_function.
    If there is no model improving the input
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
    predictions_list : list of instances of Predictions
        Each element of the list is an instance of Predictions of a model
        on the same (cross-validation valid) data points.
    ground_truths : instance of Predictions
        The ground truth.
    best_index_list : list of integers
        Indices of the current best model.

    Returns
    -------
    best_index_list : list of integers
        Indices of the models in the new combination. If the same as input,
        no models wer found improving the score.
    """
    best_predictions = combine_predictions_list(
        predictions_list, index_list=best_index_list)
    best_score = event.official_score_function(
        ground_truths, best_predictions)
    best_index = -1
    # Combination with replacement, what Caruana suggests. Basically, if a
    # model is added several times, it's upweighted, leading to
    # integer-weighted ensembles
    r = np.arange(len(predictions_list))
    # Randomization doesn't matter, only in case of exact equality.
    # np.random.shuffle(r)
    # print r
    for i in r:
        combined_predictions = combine_predictions_list(
            predictions_list, index_list=np.append(best_index_list, i))
        new_score = event.official_score_function(
            ground_truths, combined_predictions)
        is_lower_the_better = event.official_score_type.is_lower_the_better
        if (is_lower_the_better and new_score < best_score) or\
                (not is_lower_the_better and new_score > best_score):
            best_predictions = combined_predictions
            best_index = i
            best_score = new_score
    if best_index > -1:
        return np.append(best_index_list, best_index), best_score
    else:
        return best_index_list, best_score


def compute_valid_score_cv_bag(submission):
    """Cv-bag cv_fold.valid_predictions using combine_predictions_list.

    The predictions in predictions_list[i] belong to those indicated
    by self.on_cv_folds[i].test_is.
    """
    ground_truths_train = submission.event.problem.ground_truths_train()
    if submission.state == 'tested':
        predictions_list = [submission_on_cv_fold.valid_predictions for
                            submission_on_cv_fold in submission.on_cv_folds]
        test_is_list = [submission_on_cv_fold.cv_fold.test_is for
                        submission_on_cv_fold in submission.on_cv_folds]
        for score in submission.scores:
            _, score.valid_score_cv_bags = _get_score_cv_bags(
                submission.event, score.event_score_type, predictions_list,
                ground_truths_train, test_is_list)
            score.valid_score_cv_bag = float(score.valid_score_cv_bags[-1])
    else:
        for score in submission.scores:
            score.valid_score_cv_bag = float(score.event_score_type.worst)
            score.valid_score_cv_bags = None
    db.session.commit()


def compute_test_score_cv_bag(submission):
    """Bag cv_fold.test_predictions using combine_predictions_list.

    And stores the score of the bagged predictor in test_score_cv_bag. The
    scores of partial combinations are stored in test_score_cv_bags.
    This is for assessing the bagging learning curve, which is useful for
    setting the number of cv folds to its optimal value (in case the RAMP
    is competitive, say, to win a Kaggle challenge; although it's kinda
    stupid since in those RAMPs we don't have a test file, so the learning
    curves should be assessed in compute_valid_score_cv_bag on the
    (cross-)validation sets).
    """
    if submission.state == 'tested':
        # When we have submission id in Predictions, we should get the
        # team and submission from the db
        ground_truths = submission.event.problem.ground_truths_test()
        predictions_list = [submission_on_cv_fold.test_predictions for
                            submission_on_cv_fold in submission.on_cv_folds]
        combined_predictions_list = [
            combine_predictions_list(predictions_list[:i + 1]) for
            i in range(len(predictions_list))]
        for score in submission.scores:
            score.test_score_cv_bags = [
                score.score_function(
                    ground_truths, combined_predictions) for
                combined_predictions in combined_predictions_list]
            score.test_score_cv_bag = float(score.test_score_cv_bags[-1])
    else:
        for score in submission.scores:
            score.test_score_cv_bag = float(score.event_score_type.worst)
            score.test_score_cv_bags = None
    db.session.commit()


def send_password_mail(user_name, password):
    """Update <user_name>'s password to <password> and mail it to him/her.

    Parameters
    ----------
    user_name : user name
    password : new password
    """
    user = User.query.filter_by(name=user_name).one()
    user.hashed_password = get_hashed_password(password)
    db.session.commit()

    subject = 'RAMP login information'
    body = 'Here is your login information for the RAMP site:\n\n'
    body += 'username: {}\n'.format(encode_string(user.name))
    body += 'password: {}\n'.format(password)
    body += 'Please reset your password as soon as possible '
    body += 'through this link:\n'
    body += 'http://www.ramp.studio/reset_password'
    send_mail(user.email, subject, body)


def send_password_mails(password_f_name):
    """Update <name>'s password to <password>, read from <password_f_name>.

    Can be generated by `generate_passwords <generate_passwords>`.
    Parameters
    ----------
    password_f_name : a csv file with columns `name` and `password`
    """
    passwords = pd.read_csv(password_f_name)

    for _, u in passwords.iterrows():
        send_password_mail(remove_non_ascii(u['name']), u['password'])


def setup_files_extension_type():
    """Setup the files' extensions and types.

    This function registers the file extensions and types. This function
    should be called after creating the database.
    """
    extension_names = ['py', 'R', 'txt', 'csv']
    for name in extension_names:
        add_extension(name)

    submission_file_types = [
        ('code', True, 10 ** 5),
        ('text', True, 10 ** 5),
        ('data', False, 10 ** 8)
    ]
    for name, is_editable, max_size in submission_file_types:
        add_submission_file_type(name, is_editable, max_size)

    submission_file_type_extensions = [
        ('code', 'py'),
        ('code', 'R'),
        ('text', 'txt'),
        ('data', 'csv')
    ]
    for type_name, extension_name in submission_file_type_extensions:
        add_submission_file_type_extension(type_name, extension_name)


def add_extension(name):
    """Adding a new extension, e.g., 'py'."""
    extension = Extension.query.filter_by(name=name).one_or_none()
    if extension is None:
        extension = Extension(name=name)
        logger.info('Adding {}'.format(extension))
        db.session.add(extension)
        db.session.commit()


def add_submission_file_type(name, is_editable, max_size):
    """Add a new submission file type, e.g., ('code', True, 10 ** 5).

    Should be preceded by adding extensions.
    """
    submission_file_type = SubmissionFileType.query.filter_by(
        name=name).one_or_none()
    if submission_file_type is None:
        submission_file_type = SubmissionFileType(
            name=name, is_editable=is_editable, max_size=max_size)
        logger.info('Adding {}'.format(submission_file_type))
        db.session.add(submission_file_type)
        db.session.commit()


def add_submission_file_type_extension(type_name, extension_name):
    """Adding a new submission file type extension, e.g., ('code', 'py').

    Should be preceded by adding submission file types and extensions.
    """
    submission_file_type = SubmissionFileType.query.filter_by(
        name=type_name).one()
    extension = Extension.query.filter_by(name=extension_name).one()
    type_extension = SubmissionFileTypeExtension.query.filter_by(
        type=submission_file_type, extension=extension).one_or_none()
    if type_extension is None:
        type_extension = SubmissionFileTypeExtension(
            type=submission_file_type, extension=extension)
        logger.info('Adding {}'.format(type_extension))
        db.session.add(type_extension)
        db.session.commit()


def add_workflow(workflow_object):
    """Add a new workflow.

    Workflow class should exist in ``rampwf.workflows``. The name of the
    workflow will be the classname (e.g. Classifier). Element names are taken
    from ``workflow.element_names``. Element types are inferred from the
    extension. This is important because e.g. the max size and the editability
    will depend on the type.

    ``add_workflow`` is called by :func:`add_problem`, taking the workflow to
    add from the ``problem.py`` file of the starting kit.

    Parameters
    ----------
    workflow_object : ramp.workflows
        A ramp workflow instance.
    """
    workflow_name = workflow_object.__class__.__name__
    workflow = Workflow.query.filter_by(name=workflow_name).one_or_none()
    if workflow is None:
        db.session.add(Workflow(name=workflow_name))
        workflow = Workflow.query.filter_by(name=workflow_name).one()
    for element_name in workflow_object.element_names:
        tokens = element_name.split('.')
        element_filename = tokens[0]
        # inferring that file is code if there is no extension
        if len(tokens) > 2:
            raise ValueError('File name {} should contain at most one "."'
                             .format(element_name))
        element_file_extension_name = tokens[1] if len(tokens) == 2 else 'py'
        extension = Extension.query.filter_by(
            name=element_file_extension_name).one_or_none()
        if extension is None:
            raise ValueError('Unknown extension {}.'
                             .format(element_file_extension_name))
        type_extension = SubmissionFileTypeExtension.query.filter_by(
            extension=extension).one_or_none()
        if type_extension is None:
            raise ValueError('Unknown file type {}.'
                             .format(element_file_extension_name))

        workflow_element_type = WorkflowElementType.query.filter_by(
            name=element_filename).one_or_none()
        if workflow_element_type is None:
            workflow_element_type = WorkflowElementType(
                name=element_filename, type=type_extension.type)
            logger.info('Adding {}'.format(workflow_element_type))
            db.session.add(workflow_element_type)
            db.session.commit()
        workflow_element =\
            WorkflowElement.query.filter_by(
                workflow=workflow,
                workflow_element_type=workflow_element_type).one_or_none()
        if workflow_element is None:
            workflow_element = WorkflowElement(
                workflow=workflow,
                workflow_element_type=workflow_element_type)
            logger.info('Adding {}'.format(workflow_element))
            db.session.add(workflow_element)
    db.session.commit()


def add_problem(problem_name, force=False):
    """Adding a new RAMP problem."""
    problem = Problem.query.filter_by(name=problem_name).one_or_none()
    problem_kits_path = os.path.join(ramp_kits_path, problem_name)
    if problem is not None:
        if not force:
            raise ValueError('Attempting to delete problem and all linked '
                             'events. Use"force=True" if you wish to proceed.')
        delete_problem(problem_name)

    # XXX it's a bit ugly that we need to load the module here
    # perhaps if we can get rid of the workflow db table completely
    problem_module = import_module_from_source(
        os.path.join(problem_kits_path, 'problem.py'), 'problem')
    add_workflow(problem_module.workflow)
    problem = Problem(name=problem_name)
    logger.info('Adding {}'.format(problem))
    db.session.add(problem)
    db.session.commit()


# these could go into a delete callback in problem and event, I just don't know
# how to do that.
def delete_problem(problem_name):
    problem = Problem.query.filter_by(name=problem_name).one()
    for event in problem.events:
        delete_event(event.name)
    db.session.delete(problem)
    db.session.commit()


# the main reason having this is that I couldn't make a cascade delete in
# SubmissionSimilarity since it has two submission parents
def delete_event(event_name):
    event = Event.query.filter_by(name=event_name).one()
    submissions = get_submissions(event_name=event_name)
    delete_submission_similarity(submissions)
    db.session.delete(event)
    db.session.commit()


# the main reason having this is that I couldn't make a cascade delete in
# SubmissionSimilarity since it has two submission parents
def delete_submission_similarity(submissions):
    submission_similaritys = []
    for submission in submissions:
        submission_similaritys += SubmissionSimilarity.query.filter_by(
            target_submission=submission).all()
        submission_similaritys += SubmissionSimilarity.query.filter_by(
            source_submission=submission).all()
    for submission_similarity in submission_similaritys:
        db.session.delete(submission_similarity)
    db.session.commit()


def add_event(problem_name, event_name, event_title, is_public=False,
              force=False):
    """Adding a new RAMP event.

    Event file should be set up in
    databoard/specific/events/<event_name>. Should be preceded by adding
    a problem, then problem_name imported in the event file (problem_name
    is acting as a pointer for the join). Also adds CV folds.
    """
    event = Event.query.filter_by(name=event_name).one_or_none()
    if event is not None:
        if force:
            delete_event(event_name)
        else:
            logger.info(
                'Attempting to delete event, ' +
                'use "force=True" if you know what you are doing.')
            return
    event = Event(
        name=event_name, problem_name=problem_name, event_title=event_title)
    event.is_public = is_public
    event.is_send_submitted_mails = False
    event.is_send_trained_mails = False
    logger.info('Adding {}'.format(event))
    db.session.add(event)
    db.session.commit()

    X_train, y_train = event.problem.get_train_data()
    cv = event.problem.module.get_cv(X_train, y_train)
    for train_is, test_is in cv:
        cv_fold = CVFold(event=event, train_is=train_is, test_is=test_is)
        db.session.add(cv_fold)

    score_types = event.problem.module.score_types
    for score_type in score_types:
        event_score_type = EventScoreType(
            event=event, score_type_object=score_type)
        db.session.add(event_score_type)
    event.official_score_name = score_types[0].name
    db.session.commit()
    return event


def add_keyword(name, type, category=None, description=None, force=False):
    keyword = Keyword.query.filter_by(name=name).one_or_none()
    if keyword is not None:
        if force:
            keyword.type = type
            keyword.category = category
            keyword.description = description
        else:
            logger.info(
                'Attempting to update existing keyword, use ' +
                '"force=True" if you really want to update.')
            return
    else:
        keyword = Keyword(
            name=name, type=type, category=category, description=description)
        db.session.add(keyword)
    db.session.commit()


def add_problem_keyword(
        problem_name, keyword_name, description=None, force=False):
    problem = Problem.query.filter_by(name=problem_name).one()
    keyword = Keyword.query.filter_by(name=keyword_name).one()
    problem_keyword = ProblemKeyword.query.filter_by(
        problem=problem, keyword=keyword).one_or_none()
    if problem_keyword is not None:
        if force:
            problem_keyword.description = description
        else:
            logger.info(
                'Attempting to update existing problem-keyword, use ' +
                '"force=True" if you really want to update.')
            return
    else:
        problem_keyword = ProblemKeyword(
            problem=problem, keyword=keyword, description=description)
        db.session.add(problem_keyword)
    db.session.commit()


def add_submission_similarity(type, user, source_submission,
                              target_submission, similarity, timestamp):
    submission_similarity = SubmissionSimilarity(
        type=type, user=user, source_submission=source_submission,
        target_submission=target_submission, similarity=similarity,
        timestamp=timestamp)
    db.session.add(submission_similarity)
    db.session.commit()


def create_user(name, password, lastname, firstname, email,
                access_level='user', hidden_notes='', linkedin_url='',
                twitter_url='', facebook_url='', google_url='', github_url='',
                website_url='', bio='', is_want_news=True):
    hashed_password = get_hashed_password(password)
    user = User(name=name, hashed_password=hashed_password,
                lastname=lastname, firstname=firstname, email=email,
                access_level=access_level, hidden_notes=hidden_notes,
                linkedin_url=linkedin_url, twitter_url=twitter_url,
                facebook_url=facebook_url, google_url=google_url,
                github_url=github_url, website_url=website_url, bio=bio,
                is_want_news=is_want_news)

    # Creating default team with the same name as the user
    # user is admin of her own team
    team = Team(name=name, admin=user)
    db.session.add(team)
    db.session.add(user)
    try:
        db.session.commit()
    except IntegrityError as e:
        db.session.rollback()
        message = ''
        try:
            User.query.filter_by(name=name).one()
            message += 'username is already in use'
        except NoResultFound:
            # We only check for team names if username is not in db
            try:
                Team.query.filter_by(name=name).one()
                message += 'username is already in use as a team name'
            except NoResultFound:
                pass
        try:
            User.query.filter_by(email=email).one()
            if len(message) > 0:
                message += ' and '
            message += 'email is already in use'
        except NoResultFound:
            pass
        if len(message) > 0:
            raise NameClashError(message)
        else:
            raise e
    logger.info('Creating {}'.format(user))
    logger.info('Creating {}'.format(team))
    return user


def update_user(user, form):
    logger.info('Updating {}'.format(user))
    if encode_string(user.lastname) != encode_string(form.lastname.data):
        logger.info('Updating lastname from {} to {}'.format(
            encode_string(user.lastname), encode_string(form.lastname.data)))
    if encode_string(user.firstname) != encode_string(form.firstname.data):
        logger.info('Updating firstname from {} to {}'.format(
            encode_string(user.firstname),
            encode_string(form.firstname.data)))
    if user.email != form.email.data:
        logger.info('Updating email from {} to {}'.format(
            user.email, form.email.data))
    if user.is_want_news != form.is_want_news.data:
        logger.info('Updating is_want_news from {} to {}'.format(
            user.is_want_news, form.is_want_news.data))
    user.lastname = encode_string(form.lastname.data)
    user.firstname = encode_string(form.firstname.data)
    user.email = form.email.data
    user.linkedin_url = encode_string(form.linkedin_url.data)
    user.twitter_url = encode_string(form.twitter_url.data)
    user.facebook_url = encode_string(form.facebook_url.data)
    user.google_url = encode_string(form.google_url.data)
    user.github_url = encode_string(form.github_url.data)
    user.website_url = encode_string(form.website_url.data)
    user.bio = encode_string(form.bio.data)
    user.is_want_news = form.is_want_news.data
    try:
        db.session.commit()
    except IntegrityError as e:
        db.session.rollback()
        message = ''
        try:
            User.query.filter_by(email=user.email).one()
            message += 'email is already in use'
        except NoResultFound:
            pass
        if len(message) > 0:
            logger.error(message)
            raise NameClashError(message)
        else:
            logger.error(repr(e))
            raise e


def get_sandbox(event, user):
    event_team = get_active_user_event_team(event, user)

    submission = Submission.query.filter_by(
        event_team=event_team, is_not_sandbox=False).one_or_none()
    return submission


def ask_sign_up_team(event_name, team_name):
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=team_name).one()
    event_team = EventTeam.query.filter_by(
        event=event, team=team).one_or_none()
    if event_team is None:
        event_team = EventTeam(event=event, team=team)
        db.session.add(event_team)
        db.session.commit()


def sign_up_team(event_name, team_name):
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=team_name).one()
    event_team = EventTeam.query.filter_by(
        event=event, team=team).one_or_none()
    if event_team is None:
        ask_sign_up_team(event_name, team_name)
        event_team = EventTeam.query.filter_by(
            event=event, team=team).one_or_none()
    # submitting the starting kit for team
    from_submission_path = os.path.join(
        ramp_kits_path, event.problem.name, ramp_config['submissions_dir'],
        ramp_config['sandbox_dir'])
    make_submission_and_copy_files(
        event_name, team_name, ramp_config['sandbox_dir'],
        from_submission_path)
    for user in get_team_members(team):
        send_mail(
            to=user.email,
            subject='signed up for {} as team {}'.format(
                event_name, team_name),
            body='')
    event_team.approved = True
    db.session.commit()


def submit_starting_kit(event_name, team_name):
    """Submit all starting kits in ramp_kits_path/ramp_name/submissions."""
    event = Event.query.filter_by(name=event_name).one()
    submission_path = os.path.join(
        ramp_kits_path, event.problem.name, ramp_config['submissions_dir'])
    submission_names = os.listdir(submission_path)
    min_duration_between_submissions = event.min_duration_between_submissions
    event.min_duration_between_submissions = 0
    for submission_name in submission_names:
        from_submission_path = os.path.join(submission_path, submission_name)
        if submission_name == ramp_config['sandbox_dir']:
            submission_name = ramp_config['sandbox_dir'] + '_test'
        make_submission_and_copy_files(
            event_name, team_name, submission_name, from_submission_path)
    event.min_duration_between_submissions = min_duration_between_submissions
    db.session.commit()


def approve_user(user_name):
    user = User.query.filter_by(name=user_name).one()
    if user.access_level == 'asked':
        user.access_level = 'user'
    user.is_authenticated = True
    db.session.commit()
    send_mail(user.email, 'RAMP sign-up approved', '')


def make_event_admin(event_name, admin_name):
    event = Event.query.filter_by(name=event_name).one()
    admin = User.query.filter_by(name=admin_name).one()
    event_admin = EventAdmin.query.filter_by(
        event=event, admin=admin).one_or_none()
    if event_admin is None:
        event_admin = EventAdmin(event=event, admin=admin)
        db.session.commit()


def set_state(event_name, team_name, submission_name, state):
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=team_name).one()
    event_team = EventTeam.query.filter_by(event=event, team=team).one()
    submission = Submission.query.filter_by(
        name=submission_name, event_team=event_team).one()
    submission.set_state(state)
    db.session.commit()


def exclude_from_ensemble(event_name, team_name, submission_name):
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=team_name).one()
    event_team = EventTeam.query.filter_by(event=event, team=team).one()
    submission = Submission.query.filter_by(
        name=submission_name, event_team=event_team).one()
    submission.is_to_ensemble = False
    submission.contributivity = 0
    submission.historical_contributivity = 0
    for submission_on_cv_fold in submission.on_cv_folds:
        submission_on_cv_fold.contributivity = 0
        submission_on_cv_fold.best = False
    db.session.commit()


def delete_submission(event_name, team_name, submission_name):
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=team_name).one()
    event_team = EventTeam.query.filter_by(event=event, team=team).one()
    submission = Submission.query.filter_by(
        name=submission_name, event_team=event_team).one()
    shutil.rmtree(submission.path)

    cv_folds = CVFold.query.filter_by(event=event).all()
    for cv_fold in cv_folds:
        submission_on_cv_fold = SubmissionOnCVFold.query.filter_by(
            submission=submission, cv_fold=cv_fold).one()
        cv_fold.submissions.remove(submission_on_cv_fold)

    delete_submission_similarity([submission])
    db.session.delete(submission)
    db.session.commit()
    compute_contributivity(event_name)
    compute_historical_contributivity(event_name)
    update_user_leaderboards(event_name, team_name)
    set_n_submissions(event_name)


def make_submission_on_cv_folds(cv_folds, submission):
    for cv_fold in cv_folds:
        submission_on_cv_fold = SubmissionOnCVFold(
            submission=submission, cv_fold=cv_fold)
        db.session.add(submission_on_cv_fold)


def make_submission(event_name, team_name, submission_name, submission_path):
    # TODO: to call unit tests on submitted files. Those that are found
    # in the table that describes the workflow. For the rest just check
    # maybe size
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=team_name).one()
    event_team = EventTeam.query.filter_by(event=event, team=team).one()
    submission = Submission.query.filter_by(
        name=submission_name, event_team=event_team).one_or_none()
    if submission is None:
        # Checking if submission is too early
        all_submissions = Submission.query.filter_by(
            event_team=event_team).order_by(
                Submission.submission_timestamp).all()
        if len(all_submissions) == 0:
            last_submission = None
        else:
            last_submission = all_submissions[-1]
        if last_submission is not None and last_submission.is_not_sandbox:
            now = datetime.datetime.utcnow()
            last = last_submission.submission_timestamp
            min_diff = event.min_duration_between_submissions
            diff = (now - last).total_seconds()
            if team.admin.access_level != 'admin' and diff < min_diff:
                raise TooEarlySubmissionError(
                    'You need to wait {} more seconds until next submission'
                    .format(int(min_diff - diff)))
        submission = Submission(name=submission_name, event_team=event_team)
        make_submission_on_cv_folds(event.cv_folds, submission)
        db.session.add(submission)
    else:
        # We allow resubmit for new or failing submissions
        if submission.is_not_sandbox and\
                (submission.state == 'new' or submission.is_error):
            submission.set_state('new')
            submission.submission_timestamp = datetime.datetime.utcnow()
            for submission_on_cv_fold in submission.on_cv_folds:
                submission_on_cv_fold.reset()
        else:
            error_msg = 'Submission "{}" '.format(submission_name)
            error_msg = 'of team "{}" '.format(team_name)
            error_msg += 'at event "{}" exists already'.format(event_name)
            raise DuplicateSubmissionError(error_msg)

    # All file names with at least a . in them
    deposited_f_name_list = os.listdir(submission_path)
    deposited_f_name_list = [
        f_name for f_name in deposited_f_name_list
        if f_name.find('.') >= 0]
    # TODO: more error checking
    deposited_types = [f_name.split('.')[0]
                       for f_name in deposited_f_name_list]
    deposited_extensions = [f_name.split('.')[1]
                            for f_name in deposited_f_name_list]
    for workflow_element in event.problem.workflow.elements:
        # We find all files with matching names to workflow_element.name.
        # If none found, raise error.
        # Then look for one that has a legal extension. If none found,
        # raise error. If there are several ones, for now we use the first
        # matching file.

        i_names = [i for i in range(len(deposited_types))
                   if deposited_types[i] == workflow_element.name]
        if len(i_names) == 0:
            db.session.rollback()
            raise MissingSubmissionFileError('{}/{}/{}/{}: {}'.format(
                event_name, team_name, submission_name, workflow_element.name,
                submission_path))

        for i_name in i_names:
            extension_name = deposited_extensions[i_name]
            extension = Extension.query.filter_by(
                name=extension_name).one_or_none()
            if extension is not None:
                break
        else:
            db.session.rollback()
            extensions = [deposited_extensions[i_name] for i_name in i_names]
            extensions = extensions.join(",")
            for i_name in i_names:
                extensions.append(deposited_extensions[i_name])
            raise MissingExtensionError('{}/{}/{}/{}/{}: {}'.format(
                event_name, team_name, submission_name, workflow_element.name,
                extensions, submission_path))

        # maybe it's a resubmit
        submission_file = SubmissionFile.query.filter_by(
            workflow_element=workflow_element,
            submission=submission).one_or_none()
        # TODO: handle if resubmitted file changed extension
        if submission_file is None:
            submission_file_type = SubmissionFileType.query.filter_by(
                name=workflow_element.file_type).one()
            type_extension = SubmissionFileTypeExtension.query.filter_by(
                type=submission_file_type, extension=extension).one()
            submission_file = SubmissionFile(
                submission=submission, workflow_element=workflow_element,
                submission_file_type_extension=type_extension)
            db.session.add(submission_file)

    # for remembering it in the sandbox view
    event_team.last_submission_name = submission_name
    db.session.commit()

    update_leaderboards(event_name)
    update_user_leaderboards(event_name, team.name)

    # We should copy files here
    return submission


def make_submission_and_copy_files(event_name, team_name, new_submission_name,
                                   from_submission_path):
    """Make submission and copy files to submission.path.

    Called from sign_up_team(), merge_teams(), fetch.add_models(),
    view.sandbox().
    """
    submission = make_submission(
        event_name, team_name, new_submission_name, from_submission_path)
    # clean up the model directory in case it's a resubmission
    if os.path.exists(submission.path):
        shutil.rmtree(submission.path)
    os.mkdir(submission.path)
    open(os.path.join(submission.path, '__init__.py'), 'a').close()

    # copy the submission files into the model directory, should all this
    # probably go to Submission
    for f_name in submission.f_names:
        src = os.path.join(from_submission_path, f_name)
        dst = os.path.join(submission.path, f_name)
        shutil.copy2(src, dst)  # copying also metadata
        logger.info('Copying {} to {}'.format(src, dst))

    logger.info('Adding {}'.format(submission))
    return submission


def send_trained_mails(submission):
    event = submission.event_team.event
    if event.is_send_trained_mails:
        users = get_team_members(submission.event_team.team)
        subject = '{} in RAMP {} was trained at {}'.format(
            submission.name, event.name,
            date_time_format(submission.training_timestamp))
        body = ''
        if submission.is_error:
            error_msg = submission.error_msg.replace(
                '{}'.format(submission.path), '')
            body += 'error: {}\n'.format(encode_string(error_msg))
        else:
            for score in submission.scores:
                body += '{} = {}\n'.format(
                    score.score_name,
                    round(score.valid_score_cv_bag, score.precision))
        for user in users:
            send_mail(user.email, subject, body)


def send_submission_mails(user, submission, event_team):
    #  later can be joined to the ramp admins
    event = event_team.event
    team = event_team.team
    recipient_list = app.config.get('RAMP_ADMIN_MAILS')
    event_admins = EventAdmin.query.filter_by(event=event)
    recipient_list += [event_admin.admin.email for event_admin in event_admins]

    subject = 'fab train_test:e="{}",t="{}",s="{}"'.format(
        event.name,
        encode_string(team.name),
        encode_string(submission.name))
    body = 'user = {}\nevent = {}\nsubmission dir = {}\n'.format(
        user,
        event.name,
        submission.path)
    for recipient in recipient_list:
        send_mail(recipient, subject, body)


def send_ask_for_event_mails(user, event, n_students):
    #  later can be joined to the ramp admins
    recipient_list = app.config.get('RAMP_ADMIN_MAILS')

    subject = '{} is asking for an event on {}'.format(
        encode_string(user.name), encode_string(event.problem.name))
    body = 'user name = {} {}\n'.format(
        encode_string(user.firstname), encode_string(user.lastname))
    body += 'user email = {}\n'.format(user.email)
    body += 'event name = {}\n'.format(event.name)
    body += 'event title = {}\n'.format(encode_string(event.title))
    body += 'event start = {}\n'.format(event.opening_timestamp)
    body += 'event end = {}\n'.format(event.closing_timestamp)
    body += 'approximate number of students = {}\n'.format(n_students)
    body += 'minimum duration between submissions = {}s\n'.format(
        event.min_duration_between_submissions)
    # this is too important to miss in case mailing bugs
    logger.info(subject)
    logger.info(body)
    for recipient in recipient_list:
        send_mail(recipient, subject, body)


def _user_mail_body(user):
    body = ''
    body += 'user = {}\n'.format(encode_string(user.name))
    body += 'name = {} {}\n'.format(
        encode_string(user.firstname),
        encode_string(user.lastname))
    body += 'email = {}\n'.format(user.email)
    body += 'linkedin = {}\n'.format(user.linkedin_url)
    body += 'twitter = {}\n'.format(user.twitter_url)
    body += 'facebook = {}\n'.format(user.facebook_url)
    body += 'github = {}\n'.format(user.github_url)
    if user.hidden_notes is not None:
        body += 'notes = {}\n'.format(encode_string(user.hidden_notes))
    if user.bio is not None:
        body += 'bio = {}\n\n'.format(encode_string(user.bio))
    body += '\n'
    return body


def send_sign_up_request_mail(event, user):
    team = Team.query.filter_by(name=user.name).one()
    recipient_list = app.config.get('RAMP_ADMIN_MAILS')
    event_admins = EventAdmin.query.filter_by(event=event)
    recipient_list += [event_admin.admin.email for event_admin in event_admins]

    subject = 'fab sign_up_team:e="{}",t="{}"'.format(event.name, team.name)
    body = 'event = {}\n'.format(event.name)
    body += _user_mail_body(user)
    url_approve = 'https://www.ramp.studio/events/{}/sign_up/{}'.format(
        event.name, encode_string(user.name))
    body += 'Click on this link to approve this user for this event: {}\n'.\
        format(url_approve)

    for recipient in recipient_list:
        send_mail(recipient, subject, body)


def send_register_request_mail(user):
    recipient_list = app.config.get('RAMP_ADMIN_MAILS')
    subject = 'fab approve_user:u="{}"'.format(user.name)
    body = _user_mail_body(user)
    url_approve = 'http://www.ramp.studio/sign_up/{}'.format(
        encode_string(user.name))
    body += 'Click on the link to approve the registration '
    body += 'of this user: {}\n'.format(url_approve)

    for recipient in recipient_list:
        send_mail(recipient, subject, body)


def get_earliest_new_submission(event_name=None):
    if event_name is None:
        new_submissions = Submission.query.filter_by(
            state='new').filter(Submission.is_not_sandbox).order_by(
            Submission.submission_timestamp).all()
    # a fast fix: prefixing event name with 'not_' will exclude the event
    elif event_name[:4] == 'not_':
        event_name = event_name[4:]
        new_submissions = db.session.query(
            Submission, Event, EventTeam).filter(
            Event.name != event_name).filter(
            Event.id == EventTeam.event_id).filter(
            EventTeam.id == Submission.event_team_id).filter(
            Submission.state == 'new').filter(
            Submission.is_not_sandbox).order_by(
            Submission.submission_timestamp).all()
        if new_submissions:
            new_submissions = list(zip(*new_submissions)[0])
        else:
            new_submissions = []
    else:
        new_submissions = db.session.query(
            Submission, Event, EventTeam).filter(
            Event.name == event_name).filter(
            Event.id == EventTeam.event_id).filter(
            EventTeam.id == Submission.event_team_id).filter(
            Submission.state == 'new').filter(
            Submission.is_not_sandbox).order_by(
            Submission.submission_timestamp).all()
        if new_submissions:
            new_submissions = list(zip(*new_submissions)[0])
        else:
            new_submissions = []
    # Give ten seconds to upload submission files. Can be eliminated
    # once submission files go into database.
    new_submissions = [
        s for s in new_submissions
        if datetime.datetime.utcnow() - s.submission_timestamp >
        datetime.timedelta(0, 10)]

    if len(new_submissions) == 0:
        return None
    else:
        return new_submissions[0]


def set_n_submissions(event_name=None):
    if event_name is None:
        events = Event.query.all()
        for event in events:
            event.set_n_submissions()
    else:
        event = Event.query.filter_by(name=event_name).one()
        event.set_n_submissions()
    db.session.commit()


def set_contributivity(submission, is_commit=True):
    submission.set_contributivity()
    if is_commit:
        db.session.commit()


def backend_train_test_loop(event_name=None, timeout=20,
                            is_compute_contributivity=True,
                            is_parallelize=None):
    if is_parallelize is not None:
        app.config.update({'RAMP_PARALLELIZE': is_parallelize})
    event_names = set()
    while(True):
        earliest_new_submission = get_earliest_new_submission(event_name)
        logger.info('Automatic training {} at {}'.format(
            earliest_new_submission, datetime.datetime.utcnow()))
        if earliest_new_submission is not None:
            train_test_submission(earliest_new_submission)
            score_submission(earliest_new_submission)
            event_names.add(earliest_new_submission.event.name)
            update_leaderboards(earliest_new_submission.event.name)
            update_all_user_leaderboards(earliest_new_submission.event.name)
        else:
            # We only compute contributivity if nobody is waiting
            if is_compute_contributivity:
                for event_name in event_names:
                    compute_contributivity(event_name)
                    compute_historical_contributivity(event_name)
                    set_n_submissions(event_name)
            event_names = set()
        time.sleep(timeout)


def train_test_submissions(submissions=None, force_retrain_test=False,
                           is_parallelize=None):
    """Train and test submission.

    If submissions is None, trains and tests all submissions.
    """
    if is_parallelize is not None:
        app.config.update({'RAMP_PARALLELIZE': is_parallelize})
    if submissions is None:
        submissions = Submission.query.filter(
            Submission.is_not_sandbox).order_by(Submission.id).all()
    for submission in submissions:
        train_test_submission(submission, force_retrain_test)
        score_submission(submission)


# For parallel call
def train_test_submission(submission, force_retrain_test=False):
    """Train and test submission.

    We do it here so it's dockerizable.
    """
    detached_submission_on_cv_folds = [
        DetachedSubmissionOnCVFold(submission_on_cv_fold)
        for submission_on_cv_fold in submission.on_cv_folds]

    if force_retrain_test:
        logger.info('Forced retraining/testing {}'.format(submission))

    X_train, y_train = submission.event.problem.get_train_data()
    X_test, y_test = submission.event.problem.get_test_data()

    submission.state = 'training'
    db.session.commit()

    # Parallel, dict
    if app.config.get('RAMP_PARALLELIZE'):
        # We are using 'threading' so train_test_submission_on_cv_fold
        # updates the detached submission_on_cv_fold objects. If it doesn't
        # work, we can go back to multiprocessing and
        logger.info('Number of processes = {}'.format(
            submission.event.n_jobs))
        detached_submission_on_cv_folds = Parallel(
            n_jobs=submission.event.n_jobs, verbose=5)(
            delayed(train_test_submission_on_cv_fold)(
                submission_on_cv_fold, X_train, y_train, X_test, y_test,
                force_retrain_test)
            for submission_on_cv_fold in detached_submission_on_cv_folds)
        for detached_submission_on_cv_fold, submission_on_cv_fold in\
                zip(detached_submission_on_cv_folds, submission.on_cv_folds):
            try:
                submission_on_cv_fold.update(detached_submission_on_cv_fold)
            except Exception as e:
                detached_submission_on_cv_fold.state = 'training_error'
                log_msg, detached_submission_on_cv_fold.error_msg =\
                    _make_error_message(e)
                logger.error(
                    'Training {} failed with exception: \n{}'.format(
                        detached_submission_on_cv_fold, log_msg))
                submission_on_cv_fold.update(detached_submission_on_cv_fold)
            db.session.commit()
    else:
        # detached_submission_on_cv_folds = []
        for detached_submission_on_cv_fold, submission_on_cv_fold in\
                zip(detached_submission_on_cv_folds, submission.on_cv_folds):
            train_test_submission_on_cv_fold(
                detached_submission_on_cv_fold,
                X_train, y_train, X_test, y_test,
                force_retrain_test)
            submission_on_cv_fold.update(detached_submission_on_cv_fold)
            db.session.commit()
    submission.training_timestamp = datetime.datetime.utcnow()
    submission.set_state_after_training()
    db.session.commit()


def score_submission(submission):
    # We are conservative: only score if all stages (train, test, validation)
    # were completed. submission_on_cv_fold compute scores can be called
    # manually if needed for submission in various error states.
    if submission.state == 'tested':
        logger.info('Scoring  {}'.format(submission))
        for submission_on_cv_fold in submission.on_cv_folds:
            submission_on_cv_fold.compute_train_scores()
            submission_on_cv_fold.compute_valid_scores()
            submission_on_cv_fold.compute_test_scores()
            submission_on_cv_fold.state = 'scored'
        db.session.commit()
        compute_test_score_cv_bag(submission)
        compute_valid_score_cv_bag(submission)
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
        db.session.commit()
        for score in submission.scores:
            logger.info('valid_score {} = {}'.format(
                score.score_name, score.valid_score_cv_bag))
            logger.info('test_score {} = {}'.format(
                score.score_name, score.test_score_cv_bag))
        submission.state = 'scored'
        db.session.commit()
    send_trained_mails(submission)


def _make_error_message(e):
    """Make an error message in train/test.

    log_msg is the full error what we print into logger.error. error_msg
    is what we save and display to the user. Ideally error_msg is the part
    of the code that is related to the user submission.
    """
    if hasattr(e, 'traceback'):
        log_msg = str(e.traceback)
    else:
        log_msg = repr(e)
    error_msg = log_msg
    # TODO: It the user calls something in his classifier, that part of the
    # stack is lost. We should make this more intelligent.
    cut_exception_text = error_msg.rfind('--->')
    if cut_exception_text > 0:
        error_msg = error_msg[cut_exception_text:]
    return log_msg, error_msg


def train_test_submission_on_cv_fold(detached_submission_on_cv_fold,
                                     X_train, y_train,
                                     X_test, y_test, force_retrain_test=False):
    train_submission_on_cv_fold(
        detached_submission_on_cv_fold, X_train, y_train,
        force_retrain=force_retrain_test)
    if 'error' not in detached_submission_on_cv_fold.state:
        test_submission_on_cv_fold(
            detached_submission_on_cv_fold, X_test, y_test,
            force_retest=force_retrain_test)
    # to avoid pickling error when joblib tries to return the model
    detached_submission_on_cv_fold.trained_submission = None
    # When called in a single thread, we don't need the return value,
    # submission_on_cv_fold is modified in place. When called in parallel
    # multiprocessing mode, however, copies are made when the function is
    # called, so we have to explicitly return the modified object (so it is
    # ercopied into the original object)
    return detached_submission_on_cv_fold


def train_submission_on_cv_fold(detached_submission_on_cv_fold, X, y,
                                force_retrain=False):
    if detached_submission_on_cv_fold.state not in ['new', 'checked']\
            and not force_retrain:
        if 'error' in detached_submission_on_cv_fold.state:
            logger.error('Trying to train failed {}'.format(
                detached_submission_on_cv_fold))
        else:
            logger.info('Already trained {}'.format(
                detached_submission_on_cv_fold))
        return

    # so to make it importable, TODO: should go to make_submission
    # open(os.path.join(self.submission.path, '__init__.py'), 'a').close()

    train_is = detached_submission_on_cv_fold.train_is

    logger.info('Training {}'.format(detached_submission_on_cv_fold))
    start = timeit.default_timer()
    try:
        detached_submission_on_cv_fold.state = 'training'
        detached_submission_on_cv_fold.trained_submission =\
            detached_submission_on_cv_fold.workflow.train_submission(
                detached_submission_on_cv_fold.path, X, y, train_is)
        detached_submission_on_cv_fold.state = 'trained'
    except Exception as e:
        detached_submission_on_cv_fold.state = 'training_error'
        log_msg, detached_submission_on_cv_fold.error_msg =\
            _make_error_message(e)
        logger.error(
            'Training {} failed with exception: \n{}'.format(
                detached_submission_on_cv_fold, log_msg))
        return
    end = timeit.default_timer()
    detached_submission_on_cv_fold.train_time = end - start

    logger.info('Validating {}'.format(detached_submission_on_cv_fold))
    start = timeit.default_timer()
    try:
        # Computing predictions on full training set
        y_pred = detached_submission_on_cv_fold.workflow.test_submission(
            detached_submission_on_cv_fold.trained_submission, X)
        assert_all_finite(y_pred)
        if len(y_pred) == len(y):
            detached_submission_on_cv_fold.full_train_y_pred = y_pred
            detached_submission_on_cv_fold.state = 'validated'
        else:
            detached_submission_on_cv_fold.error_msg =\
                'Wrong output dimension in ' +\
                'predict: {} instead of {}'.format(len(y_pred), len(y))
            detached_submission_on_cv_fold.state = 'validating_error'
            logger.error(
                'Validating {} failed with exception: \n{}'.format(
                    detached_submission_on_cv_fold.error_msg))
            return
    except Exception as e:
        detached_submission_on_cv_fold.state = 'validating_error'
        log_msg, detached_submission_on_cv_fold.error_msg =\
            _make_error_message(e)
        logger.error(
            'Validating {} failed with exception: \n{}'.format(
                detached_submission_on_cv_fold, log_msg))
        return
    end = timeit.default_timer()
    detached_submission_on_cv_fold.valid_time = end - start


def test_submission_on_cv_fold(detached_submission_on_cv_fold, X, y,
                               force_retest=False):
    if detached_submission_on_cv_fold.state not in\
            ['new', 'checked', 'trained', 'validated'] and not force_retest:
        if 'error' in detached_submission_on_cv_fold.state:
            logger.error('Trying to test failed {}'.format(
                detached_submission_on_cv_fold))
        else:
            logger.info('Already tested {}'.format(
                detached_submission_on_cv_fold))
        return

    logger.info('Testing {}'.format(detached_submission_on_cv_fold))
    start = timeit.default_timer()
    try:
        y_pred = detached_submission_on_cv_fold.workflow.test_submission(
            detached_submission_on_cv_fold.trained_submission, X)
        assert_all_finite(y_pred)
        if len(y_pred) == len(y):
            detached_submission_on_cv_fold.test_y_pred = y_pred
            detached_submission_on_cv_fold.state = 'tested'
        else:
            detached_submission_on_cv_fold.error_msg =\
                'Wrong output dimension in ' +\
                'predict: {} instead of {}'.format(len(y_pred), len(y))
            detached_submission_on_cv_fold.state = 'testing_error'
            logger.error(
                'Testing {} failed with exception: \n{}'.format(
                    detached_submission_on_cv_fold.error_msg))
    except Exception as e:
        detached_submission_on_cv_fold.state = 'testing_error'
        log_msg, detached_submission_on_cv_fold.error_msg =\
            _make_error_message(e)
        logger.error(
            'Testing {} failed with exception: \n{}'.format(
                detached_submission_on_cv_fold, log_msg))
        return
    end = timeit.default_timer()
    detached_submission_on_cv_fold.test_time = end - start


def compute_contributivity(event_name, start_time_stamp=None,
                           end_time_stamp=None, force_ensemble=False,
                           is_save_y_pred=False):
    compute_contributivity_no_commit(
        event_name, start_time_stamp, end_time_stamp, force_ensemble,
        is_save_y_pred)
    db.session.commit()


def compute_contributivity_no_commit(
        event_name, start_time_stamp=None, end_time_stamp=None,
        force_ensemble=False, is_save_y_pred=False):
    """Compute contributivity leaderboard scores.

    Parameters
    ----------
    event_name : string
    force_ensemble : boolean
        To force include deleted models.
    """
    logger.info('Combining models')
    # The following should go into config, we'll get there when we have a
    # lot of models.
    # One of Caruana's trick: bag the models
    # selected_index_lists = np.array([random.sample(
    #    range(len(models_df)), int(0.8*models_df.shape[0]))
    #    for _ in range(n_bags)])
    # Or you can select a subset
    # selected_index_lists = np.array([[24, 26, 28, 31]])
    # Or just take everybody
    # Now all of this should be handled by submission.is_to_ensemble parameter
    # It would also make more sense to bag differently in eah fold, see the
    # comment in cv_fold.get_combined_predictions

    event = Event.query.filter_by(name=event_name).one()
    submissions = get_submissions(event_name=event_name)

    ground_truths_train = event.problem.ground_truths_train()
    ground_truths_test = event.problem.ground_truths_test()

    combined_predictions_list = []
    best_predictions_list = []
    combined_test_predictions_list = []
    best_test_predictions_list = []
    test_is_list = []
    for cv_fold in CVFold.query.filter_by(event=event).all():
        logger.info('{}'.format(cv_fold))
        ground_truths_valid = event.problem.ground_truths_valid(
            cv_fold.test_is)
        combined_predictions, best_predictions,\
            combined_test_predictions, best_test_predictions =\
            _compute_contributivity_on_fold(
                cv_fold, ground_truths_valid,
                start_time_stamp, end_time_stamp, force_ensemble)
        # TODO: if we do asynchron CVs, this has to be revisited
        if combined_predictions is None:
            logger.info('No submissions to combine')
            return
        combined_predictions_list.append(combined_predictions)
        best_predictions_list.append(best_predictions)
        combined_test_predictions_list.append(combined_test_predictions)
        best_test_predictions_list.append(best_test_predictions)
        test_is_list.append(cv_fold.test_is)
    for submission in submissions:
        set_contributivity(submission, is_commit=False)
    # if there are no predictions to combine, it crashed
    combined_predictions_list = [c for c in combined_predictions_list
                                 if c is not None]
    if len(combined_predictions_list) > 0:
        combined_predictions, scores = _get_score_cv_bags(
            event, event.official_score_type, combined_predictions_list,
            ground_truths_train, test_is_list=test_is_list)
        if is_save_y_pred:
            np.savetxt(
                'y_train_pred.csv', combined_predictions.y_pred, delimiter=',')
        logger.info('Combined combined valid score = {}'.format(scores))
        event.combined_combined_valid_score = float(scores[-1])
    else:
        event.combined_combined_valid_score = None

    best_predictions_list = [c for c in best_predictions_list
                             if c is not None]
    if len(best_predictions_list) > 0:
        _, scores = _get_score_cv_bags(
            event, event.official_score_type, best_predictions_list,
            ground_truths_train, test_is_list=test_is_list)
        logger.info('Combined foldwise best valid score = {}'.format(scores))
        event.combined_foldwise_valid_score = float(scores[-1])
    else:
        event.combined_foldwise_valid_score = None

    combined_test_predictions_list = [c for c in combined_test_predictions_list
                                      if c is not None]
    if len(combined_test_predictions_list) > 0:
        combined_predictions, scores = _get_score_cv_bags(
            event, event.official_score_type, combined_test_predictions_list,
            ground_truths_test)
        if is_save_y_pred:
            np.savetxt(
                'y_test_pred.csv', combined_predictions.y_pred, delimiter=',')
        logger.info('Combined combined test score = {}'.format(scores))
        event.combined_combined_test_score = float(scores[-1])
    else:
        event.combined_combined_test_score = None

    best_test_predictions_list = [c for c in best_test_predictions_list
                                  if c is not None]
    if len(best_test_predictions_list) > 0:
        _, scores = _get_score_cv_bags(
            event, event.official_score_type, best_test_predictions_list,
            ground_truths_test)
        logger.info('Combined foldwise best valid score = {}'.format(scores))
        event.combined_foldwise_test_score = float(scores[-1])
    else:
        event.combined_foldwise_test_score = None

    return event.combined_combined_valid_score,\
        event.combined_foldwise_valid_score,\
        event.combined_combined_test_score,\
        event.combined_foldwise_test_score


def _compute_contributivity_on_fold(cv_fold, ground_truths_valid,
                                    start_time_stamp=None, end_time_stamp=None,
                                    force_ensemble=False, min_improvement=0.0):
    """Construct the best model combination on a single fold.

    Using greedy forward selection with replacement. See
    http://www.cs.cornell.edu/~caruana/ctp/ct.papers/
    caruana.icml04.icdm06long.pdf.
    Then sets foldwise contributivity.

    Parameters
    ----------
    force_ensemble : boolean
        To force include deleted models
    """
    # The submissions must have is_to_ensemble set to True. It is for
    # fogetting models. Users can also delete models in which case
    # we make is_valid false. We then only use these models if
    # force_ensemble is True.
    # We can further bag here which should be handled in config (or
    # ramp table.) Or we could bag in get_next_best_single_fold

    # this is the bottleneck
    selected_submissions_on_fold = [
        submission_on_fold for submission_on_fold in cv_fold.submissions
        if (submission_on_fold.submission.is_valid or force_ensemble) and
        submission_on_fold.submission.is_to_ensemble and
        submission_on_fold.submission.is_in_competition and
        submission_on_fold.state == 'scored' and
        submission_on_fold.submission.is_not_sandbox
    ]
    # reset
    for submission_on_fold in selected_submissions_on_fold:
        submission_on_fold.best = False
        submission_on_fold.contributivity = 0.0
    # select submissions in time interval
    if start_time_stamp is not None:
        selected_submissions_on_fold = [
            submission_on_fold for submission_on_fold
            in selected_submissions_on_fold
            if submission_on_fold.submission.submission_timestamp >=
            start_time_stamp
        ]
    if end_time_stamp is not None:
        selected_submissions_on_fold = [
            submission_on_fold for submission_on_fold
            in selected_submissions_on_fold
            if submission_on_fold.submission.submission_timestamp <=
            end_time_stamp
        ]

    if len(selected_submissions_on_fold) == 0:
        return None, None, None, None
    # TODO: maybe this can be simplified. Don't need to get down
    # to prediction level.
    predictions_list = [
        submission_on_fold.valid_predictions
        for submission_on_fold in selected_submissions_on_fold]
    valid_scores = [
        submission_on_fold.official_score.valid_score
        for submission_on_fold in selected_submissions_on_fold]
    if cv_fold.event.official_score_type.is_lower_the_better:
        best_prediction_index = np.argmin(valid_scores)
    else:
        best_prediction_index = np.argmax(valid_scores)
    best_index_list = np.array([best_prediction_index])
    improvement = True
    while improvement and len(best_index_list) < cv_fold.event.max_n_ensemble:
        old_best_index_list = best_index_list
        best_index_list, score = get_next_best_single_fold(
            cv_fold.event, predictions_list, ground_truths_valid,
            best_index_list, min_improvement)
        improvement = len(best_index_list) != len(old_best_index_list)
        logger.info('\t{}: {}'.format(old_best_index_list, score))
    # set
    selected_submissions_on_fold[best_index_list[0]].best = True
    # we share a unit of 1. among the contributive submissions
    unit_contributivity = 1. / len(best_index_list)
    for i in best_index_list:
        selected_submissions_on_fold[i].contributivity +=\
            unit_contributivity
    combined_predictions = combine_predictions_list(
        predictions_list, index_list=best_index_list)
    best_predictions = predictions_list[best_index_list[0]]

    test_predictions_list = [
        submission_on_fold.test_predictions
        for submission_on_fold in selected_submissions_on_fold
    ]
    if any(test_predictions_list) is None:
        logger.error("Can't compute combined test score," +
                     " some submissions are untested.")
        combined_test_predictions = None
        best_test_predictions = None
    else:
        combined_test_predictions = combine_predictions_list(
            test_predictions_list, index_list=best_index_list)
        best_test_predictions = test_predictions_list[best_index_list[0]]

    return combined_predictions, best_predictions,\
        combined_test_predictions, best_test_predictions


def compute_historical_contributivity_no_commit(event_name):
    submissions = get_submissions(event_name=event_name)
    submissions.sort(key=lambda x: x.submission_timestamp, reverse=True)
    for submission in submissions:
        submission.historical_contributivity = 0.0
    for submission in submissions:
        submission.historical_contributivity += submission.contributivity
        submission_similaritys = SubmissionSimilarity.query.filter_by(
            type='target_credit', target_submission=submission).all()
        if submission_similaritys:
            # if a target team enters several credits to a source submission
            # we only take the latest
            submission_similaritys.sort(
                key=lambda x: x.timestamp, reverse=True)
            processed_submissions = []
            historical_contributivity = submission.historical_contributivity
            for submission_similarity in submission_similaritys:
                source_submission = submission_similarity.source_submission
                if source_submission not in processed_submissions:
                    partial_credit = historical_contributivity *\
                        submission_similarity.similarity
                    source_submission.historical_contributivity +=\
                        partial_credit
                    submission.historical_contributivity -= partial_credit
                    processed_submissions.append(source_submission)


def compute_historical_contributivity(event_name):
    compute_historical_contributivity_no_commit(event_name)
    db.session.commit()
    update_leaderboards(event_name)
    update_all_user_leaderboards(event_name)


def is_user_signed_up(event_name, user_name):
    for event_team in get_user_event_teams(event_name, user_name):
        if event_team.is_active and event_team.approved:
            return True
    return False


def is_user_asked_sign_up(event_name, user_name):
    for event_team in get_user_event_teams(event_name, user_name):
        if event_team.is_active and not event_team.approved:
            return True
    return False


def is_admin(event, user):
    if user.access_level == 'admin':
        return True
    event_admin = EventAdmin.query.filter_by(
        event=event, admin=user).one_or_none()
    if event_admin is None:
        return False
    else:
        return True


def is_public_event(event, user):
    if event is None:
        return False
    if user.access_level == 'asked':
        return False
    if event.is_public or is_admin(event, user):
        return True
    return False


def is_open_leaderboard(event, current_user):
    """
    True if current_user can look at submission of event.

    If submission is None, it is assumed to be the sandbox.
    """
    if not current_user.is_authenticated or not current_user.is_active:
        return False
    if is_admin(event, current_user):
        return True
    if not is_user_signed_up(event.name, current_user.name):
        return False
    if event.is_public_open:
        return True
    return False


def is_open_code(event, current_user, submission=None):
    """
    True if current_user can look at submission of event.

    If submission is None, it is assumed to be the sandbox.
    """
    if not current_user.is_authenticated or not current_user.is_active:
        return False
    if is_admin(event, current_user):
        return True
    if not is_user_signed_up(event.name, current_user.name):
        return False
    if submission is None:
        # It's probably stupid since we could just return True here, but this
        # access right thing will have to be cleaned up anyways
        submission = get_sandbox(event, current_user)
    if current_user in get_team_members(submission.event_team.team):
        return True
    if event.is_public_open:
        return True
    return False


def get_leaderboards(event_name, user_name=None):
    """Create leaderboards.

    Returns
    -------
    leaderboard_html_with_links : html string
    leaderboard_html_with_no_links : html string
    """
    submissions = get_submissions(event_name=event_name, user_name=user_name)
    submissions = [submission for submission in submissions
                   if submission.is_public_leaderboard and submission.is_valid]
    event = Event.query.filter_by(name=event_name).one()

    score_names = [score_type.name for score_type in event.score_types]
    scoress = np.array([
        [round(score.valid_score_cv_bag, score.precision)
         for score in submission.ordered_scores(score_names)]
        for submission in submissions
    ]).T
    leaderboard_df = pd.DataFrame()
    leaderboard_df['team'] = [
        submission.event_team.team.name for submission in submissions]
    leaderboard_df['submission'] = [
        submission.name_with_link for submission in submissions]
    leaderboard_df['submission no link'] = [
        submission.name[:20] for submission in submissions]
    leaderboard_df['contributivity'] = [
        int(round(100 * submission.contributivity))
        for submission in submissions]
    leaderboard_df['historical contributivity'] = [
        int(round(100 * submission.historical_contributivity))
        for submission in submissions]
    for score_name in score_names:  # to make sure the column is created
        leaderboard_df[score_name] = 0
    for score_name, scores in zip(score_names, scoress):
        leaderboard_df[score_name] = scores
    leaderboard_df['train time [s]'] = [
        int(round(submission.train_time_cv_mean))
        for submission in submissions]
    leaderboard_df['test time [s]'] = [
        int(round(submission.valid_time_cv_mean))
        for submission in submissions]
    leaderboard_df['submitted at (UTC)'] = [
        date_time_format(submission.submission_timestamp)
        for submission in submissions]
    sort_column = event.official_score_name
    leaderboard_df = leaderboard_df.sort_values(
        sort_column, ascending=event.official_score_type.is_lower_the_better)

    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        # classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    leaderboard_html_with_links = leaderboard_df.drop(
        'submission no link', axis=1).to_html(**html_params)
    leaderboard_df = leaderboard_df.drop('submission', axis=1)
    leaderboard_df = leaderboard_df.rename(
        columns={'submission no link': 'submission'})
    leaderboard_html_no_links = leaderboard_df.to_html(**html_params)

    return (
        table_format(leaderboard_html_with_links),
        table_format(leaderboard_html_no_links)
    )


def get_private_leaderboards(event_name, user_name=None):
    """Create private leaderboards.

    Returns
    -------
    leaderboard_html_with_links : html string
    leaderboard_html_with_no_links : html string
    """
    submissions = get_submissions(event_name=event_name, user_name=user_name)
    submissions = [submission for submission in submissions
                   if submission.is_private_leaderboard]
    event = Event.query.filter_by(name=event_name).one()

    score_names = [score_type.name for score_type in event.score_types]
    scoresss = np.array([
        [[round(score.valid_score_cv_bag, score.precision),
          round(score.valid_score_cv_mean, score.precision),
          round(score.valid_score_cv_std, score.precision + 1),
          round(score.test_score_cv_bag, score.precision),
          round(score.test_score_cv_mean, score.precision),
          round(score.test_score_cv_std, score.precision + 1)]
         for score in submission.ordered_scores(score_names)]
        for submission in submissions
    ])
    if len(submissions) > 0:
        scoresss = np.swapaxes(scoresss, 0, 1)
    leaderboard_df = pd.DataFrame()
    leaderboard_df['team'] = [
        submission.event_team.team.name for submission in submissions]
    leaderboard_df['submission'] = [
        submission.name_with_link for submission in submissions]
    for score_name in score_names:  # to make sure the column is created
        leaderboard_df[score_name + ' pub bag'] = 0
        leaderboard_df[score_name + ' pub mean'] = 0
        leaderboard_df[score_name + ' pub std'] = 0
        leaderboard_df[score_name + ' pr bag'] = 0
        leaderboard_df[score_name + ' pr mean'] = 0
        leaderboard_df[score_name + ' pr std'] = 0
    for score_name, scoress in zip(score_names, scoresss):
        leaderboard_df[score_name + ' pub bag'] = scoress[:, 0]
        leaderboard_df[score_name + ' pub mean'] = scoress[:, 1]
        leaderboard_df[score_name + ' pub std'] = scoress[:, 2]
        leaderboard_df[score_name + ' pr bag'] = scoress[:, 3]
        leaderboard_df[score_name + ' pr mean'] = scoress[:, 4]
        leaderboard_df[score_name + ' pr std'] = scoress[:, 5]
    leaderboard_df['contributivity'] = [
        int(round(100 * submission.contributivity))
        for submission in submissions]
    leaderboard_df['historical contributivity'] = [
        int(round(100 * submission.historical_contributivity))
        for submission in submissions]
    leaderboard_df['train time [s]'] = [
        int(round(submission.train_time_cv_mean))
        for submission in submissions]
    leaderboard_df['trt std'] = [
        int(round(submission.train_time_cv_std))
        for submission in submissions]
    leaderboard_df['test time [s]'] = [
        int(round(submission.valid_time_cv_mean))
        for submission in submissions]
    leaderboard_df['tet std'] = [
        int(round(submission.valid_time_cv_std))
        for submission in submissions]
    leaderboard_df['max RAM [MB]'] = [
        int(round(submission.max_ram)) if type(submission.max_ram) == float
        else 0
        for submission in submissions]
    leaderboard_df['submitted at (UTC)'] = [
        date_time_format(submission.submission_timestamp)
        for submission in submissions]
    sort_column = event.official_score_name + ' pr bag'
    leaderboard_df = leaderboard_df.sort_values(
        sort_column, ascending=event.official_score_type.is_lower_the_better)
    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        # classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    leaderboard_html = leaderboard_df.to_html(**html_params)

    # logger.info(u'private leaderboard construction takes {}ms'.format(
    #     int(1000 * (time.time() - start))))

    return table_format(leaderboard_html)


def get_competition_leaderboards(event_name):
    """Create leaderboards.

    Returns
    -------
    leaderboard_html_with_links : html string
    leaderboard_html_with_no_links : html string
    """
    submissions = get_submissions(event_name=event_name)
    submissions = [
        submission for submission in submissions
        if submission.is_public_leaderboard and submission.is_valid and
        submission.is_in_competition]
    event = Event.query.filter_by(name=event_name).one()
    score_type = event.official_score_type
    score_name = event.official_score_name

    # construct full leaderboard
    leaderboard_df = pd.DataFrame()
    leaderboard_df['team'] = [
        submission.event_team.team.name for submission in submissions]
    leaderboard_df['submission'] = [
        submission.name[:20] for submission in submissions]
    leaderboard_df['public ' + score_name] = [
        round(
            submission.official_score.valid_score_cv_bag, score_type.precision)
        for submission in submissions]
    leaderboard_df['private ' + score_name] = [
        round(
            submission.official_score.test_score_cv_bag, score_type.precision)
        for submission in submissions]
    leaderboard_df['train time [s]'] = [
        int(round(submission.train_time_cv_mean))
        for submission in submissions]
    leaderboard_df['test time [s]'] = [
        int(round(submission.valid_time_cv_mean))
        for submission in submissions]
    leaderboard_df['submitted at (UTC)'] = [
        date_time_format(submission.submission_timestamp)
        for submission in submissions]

    # select best submission for each team
    if score_type.is_lower_the_better:
        best_df = leaderboard_df.groupby('team').min()
    else:
        best_df = leaderboard_df.groupby('team').max()
    best_df = best_df[['public ' + score_name]].reset_index()
    best_df['best'] = True

    # merge to get a best indicator column then select best
    leaderboard_df = pd.merge(
        leaderboard_df, best_df, how='left',
        left_on=['team', 'public ' + score_name],
        right_on=['team', 'public ' + score_name])
    leaderboard_df = leaderboard_df.fillna(False)
    leaderboard_df = leaderboard_df[leaderboard_df['best']]
    leaderboard_df = leaderboard_df.drop(columns='best')

    # dealing with ties: we need the lowest timestamp
    best_df = leaderboard_df.groupby('team').min()
    best_df = best_df[['submitted at (UTC)']].reset_index()
    best_df['best'] = True
    leaderboard_df = pd.merge(
        leaderboard_df, best_df, how='left',
        left_on=['team', 'submitted at (UTC)'],
        right_on=['team', 'submitted at (UTC)'])
    leaderboard_df = leaderboard_df.fillna(False)
    leaderboard_df = leaderboard_df[leaderboard_df['best']]
    leaderboard_df = leaderboard_df.drop(columns='best')

    # sort by public score then by submission timestamp, compute rank
    leaderboard_df = leaderboard_df.sort_values(
        by=['public ' + score_name, 'submitted at (UTC)'],
        ascending=[score_type.is_lower_the_better, True])
    leaderboard_df['public rank'] = np.arange(len(leaderboard_df)) + 1

    # sort by private score then by submission timestamp, compute rank
    leaderboard_df = leaderboard_df.sort_values(
        by=['private ' + score_name, 'submitted at (UTC)'],
        ascending=[score_type.is_lower_the_better, True])
    leaderboard_df['private rank'] = np.arange(len(leaderboard_df)) + 1

    leaderboard_df['move'] =\
        leaderboard_df['public rank'] - leaderboard_df['private rank']
    leaderboard_df['move'] = [
        '{0:+d}'.format(m) if m != 0 else '-' for m in leaderboard_df['move']]

    public_leaderboard_df = leaderboard_df[[
        'public rank', 'team', 'submission', 'public ' + score_name,
        'train time [s]', 'test time [s]', 'submitted at (UTC)']]
    public_leaderboard_df = public_leaderboard_df.rename(columns={
        'public ' + score_name: score_name,
        'public rank': 'rank'
    })
    public_leaderboard_df = public_leaderboard_df.sort_values(by='rank')

    private_leaderboard_df = leaderboard_df[[
        'private rank', 'move', 'team', 'submission', 'private ' + score_name,
        'train time [s]', 'test time [s]', 'submitted at (UTC)']]
    private_leaderboard_df = private_leaderboard_df.rename(columns={
        'private ' + score_name: score_name,
        'private rank': 'rank'
    })
    private_leaderboard_df = private_leaderboard_df.sort_values(by='rank')

    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        # classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    public_leaderboard_html = public_leaderboard_df.to_html(**html_params)
    private_leaderboard_html = private_leaderboard_df.to_html(**html_params)

    return (
        table_format(public_leaderboard_html),
        table_format(private_leaderboard_html)
    )


def update_leaderboards(event_name):
    private_leaderboard_html = get_private_leaderboards(
        event_name)
    leaderboards = get_leaderboards(event_name)
    failed_leaderboard_html = get_failed_leaderboard(event_name)
    new_leaderboard_html = get_new_leaderboard(event_name)
    competition_leaderboards_html = get_competition_leaderboards(event_name)

    event = Event.query.filter_by(name=event_name).one()
    event.private_leaderboard_html = private_leaderboard_html
    event.public_leaderboard_html_with_links = leaderboards[0]
    event.public_leaderboard_html_no_links = leaderboards[1]
    event.failed_leaderboard_html = failed_leaderboard_html
    event.new_leaderboard_html = new_leaderboard_html
    event.public_competition_leaderboard_html =\
        competition_leaderboards_html[0]
    event.private_competition_leaderboard_html = \
        competition_leaderboards_html[1]
    db.session.commit()


def update_user_leaderboards(event_name, user_name):
    logger.info('Leaderboard is updated for user {} in event {}.'.format(
        user_name, event_name))
    leaderboards = get_leaderboards(event_name, user_name)
    failed_leaderboard_html = get_failed_leaderboard(event_name, user_name)
    new_leaderboard_html = get_new_leaderboard(event_name, user_name)
    for event_team in get_user_event_teams(event_name, user_name):
        event_team.leaderboard_html = leaderboards[0]
        event_team.failed_leaderboard_html = failed_leaderboard_html
        event_team.new_leaderboard_html = new_leaderboard_html
    db.session.commit()


def update_all_user_leaderboards(event_name):
    event = Event.query.filter_by(name=event_name).one()
    event_teams = EventTeam.query.filter_by(event=event).all()
    for event_team in event_teams:
        user_name = event_team.team.name
        leaderboards = get_leaderboards(event_name, user_name)
        failed_leaderboard_html = get_failed_leaderboard(
            event_name, user_name)
        new_leaderboard_html = get_new_leaderboard(event_name, user_name)
        event_team.leaderboard_html = leaderboards[0]
        event_team.failed_leaderboard_html = failed_leaderboard_html
        event_team.new_leaderboard_html = new_leaderboard_html
    db.session.commit()


def get_failed_leaderboard(event_name, team_name=None, user_name=None):
    """Create failed leaderboards.

    Returns
    -------
    leaderboard_html : html string
    """
    submissions = get_submissions(
        event_name=event_name, team_name=team_name, user_name=user_name)
    submissions = [submission for submission in submissions
                   if submission.is_error]

    columns = ['team',
               'submission',
               'submitted at (UTC)',
               'error']
    leaderboard_dict_list = [
        {column: value for column, value in zip(
            columns, [submission.event_team.team.name,
                      submission.name_with_link,
                      date_time_format(submission.submission_timestamp),
                      submission.state_with_link])}
        for submission in submissions
    ]
    leaderboard_df = pd.DataFrame(leaderboard_dict_list, columns=columns)
    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        # classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    leaderboard_html = leaderboard_df.to_html(**html_params)

    # logger.info(u'failed leaderboard construction takes {}ms'.format(
    #     int(1000 * (time.time() - start))))

    return table_format(leaderboard_html)


def get_new_leaderboard(event_name, team_name=None, user_name=None):
    """Create new leaderboards.

    Returns
    -------
    leaderboard_html : html string
    """
    submissions = get_submissions(
        event_name=event_name, team_name=team_name, user_name=user_name)
    submissions = [submission for submission in submissions
                   if submission.state in
                   ['new', 'training', 'sent_to_training'] and
                   submission.is_not_sandbox]

    columns = ['team',
               'submission',
               'submitted at (UTC)']
    leaderboard_dict_list = [
        {column: value for column, value in zip(
            columns, [submission.event_team.team.name,
                      submission.name_with_link,
                      date_time_format(
                          submission.submission_timestamp)])}
        for submission in submissions
    ]
    leaderboard_df = pd.DataFrame(leaderboard_dict_list, columns=columns)
    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        # classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )

    # logger.info(u'new leaderboard construction takes {}ms'.format(
    #     int(1000 * (time.time() - start))))

    leaderboard_html = leaderboard_df.to_html(**html_params)
    return table_format(leaderboard_html)


def get_submissions(event_name=None, team_name=None, user_name=None,
                    submission_name=None):
    if event_name is None:  # All submissions
        submissions = Submission.query.all()
    else:
        if team_name is None:
            if user_name is None:  # All submissions in a given event
                submissions_ = db.session.query(
                    Submission, Event, EventTeam).filter(
                    Event.name == event_name).filter(
                    Event.id == EventTeam.event_id).filter(
                    EventTeam.id == Submission.event_team_id).all()
                if submissions_:
                    submissions = [s for (s, e, et) in submissions_]
                else:
                    submissions = []
            else:
                # All submissions for a given event by all the teams of a user
                submissions = []
                for event_team in get_user_event_teams(event_name, user_name):
                    submissions += db.session.query(
                        Submission).filter(
                        Event.name == event_name).filter(
                        Event.id == event_team.event_id).filter(
                        event_team.id == Submission.event_team_id).all()
        else:
            if submission_name is None:
                # All submissions in a given event and team
                submissions_ = db.session.query(
                    Submission, Event, Team, EventTeam).filter(
                    Event.name == event_name).filter(
                    Team.name == team_name).filter(
                    Event.id == EventTeam.event_id).filter(
                    Team.id == EventTeam.team_id).filter(
                    EventTeam.id == Submission.event_team_id).all()
            else:  # Given submission
                submissions_ = db.session.query(
                    Submission, Event, Team, EventTeam).filter(
                    Submission.name == submission_name).filter(
                    Event.name == event_name).filter(
                    Team.name == team_name).filter(
                    Event.id == EventTeam.event_id).filter(
                    Team.id == EventTeam.team_id).filter(
                    EventTeam.id == Submission.event_team_id).all()
            if submissions_:
                submissions = [s for (s, e, t, et) in submissions_]
            else:
                submissions = []
    return submissions


def get_submissions_of_state(state):
    return Submission.query.filter(Submission.state == state).all()


def set_error(team_name, submission_name, error, error_msg):
    team = Team.query.filter_by(name=team_name).one()
    submission = Submission.query.filter_by(
        team=team, name=submission_name).one()
    submission.set_error(error, error_msg)
    db.session.commit()


# def get_top_score_of_user(user, closing_timestamp):
#     """Find the best test score of user before closing timestamp.

#     Returns
#     -------
#     best_test_score : The bagged test score of the submission with the best
#     bagged valid score, from among all the submissions of the user, before
#     closing_timestamp.
#     """
#     # team = get_active_user_team(user)
#     # XXX this will not work if we allow team mergers
#     team = Team.query.filter_by(name=user.name).one()
#     submissions = Submission.query.filter_by(team=team).filter(
#         Submission.is_private_leaderboard).all()
#     best_valid_score = config.config_object.specific.score.worst
#     best_test_score = config.config_object.specific.score.worst
#     for submission in submissions:
#         if submission.valid_score_cv_bag > best_valid_score:
#             best_valid_score = submission.valid_score_cv_bag
#             best_test_score = submission.test_score_cv_bag
#     return best_test_score


# def get_top_score_per_user(closing_timestamp=None):
#     if closing_timestamp is None:
#         closing_timestamp = datetime.datetime.utcnow()
#     users = db.session.query(User).all()
#     columns = ['name',
#                'score']
#     top_score_per_user_dict = [
#         {column: value for column, value in zip(
#             columns, [
#                 user.name, get_top_score_of_user(user, closing_timestamp)])}
#         for user in users
#     ]
#     top_score_per_user_dict_df = pd.DataFrame(
#         top_score_per_user_dict, columns=columns)
#     top_score_per_user_dict_df = top_score_per_user_dict_df.sort_values(
#         'name')
#     return top_score_per_user_dict_df


def add_user_interaction(**kwargs):
    user_interaction = UserInteraction(**kwargs)
    db.session.add(user_interaction)
    db.session.commit()


def get_user_interactions_df():
    """Create user interaction table.

    Returns
    -------
    user_interactions_df : pd.DataFrame
    """
    user_interactions = UserInteraction.query.all()

    columns = ['timestamp (UTC)',
               'IP',
               'interaction',
               'user',
               'event',
               'team',
               'submission_id',
               'submission',
               'file',
               'code similarity',
               'diff']

    def user_name(user_interaction):
        user = user_interaction.user
        if user is None:
            return ''
        else:
            return user.name

    def event_name(user_interaction):
        event_team = user_interaction.event_team
        if event_team is None:
            return ''
        else:
            return event_team.event.name

    def team_name(user_interaction):
        event_team = user_interaction.event_team
        if event_team is None:
            return ''
        else:
            return event_team.team.name

    def submission_id(user_interaction):
        submission = user_interaction.submission
        if submission is None:
            return -1
        else:
            return submission.id

    def submission_name(user_interaction):
        submission = user_interaction.submission
        if submission is None:
            return ''
        else:
            return submission.name_with_link

    def submission_file_name(user_interaction):
        submission_file = user_interaction.submission_file
        if submission_file is None:
            return ''
        else:
            return submission_file.name_with_link

    def submission_similarity(user_interaction):
        similarity = user_interaction.submission_file_similarity
        if similarity is None:
            return ''
        else:
            return str(round(similarity, 2))

    def submission_diff_with_link(user_interaction):
        diff_link = user_interaction.submission_file_diff_link
        if diff_link is None:
            return ''
        else:
            return '<a href="' + diff_link + '">diff</a>'

    user_interactions_dict_list = [
        {column: value for column, value in zip(
            columns, [date_time_format(user_interaction.timestamp),
                      user_interaction.ip,
                      user_interaction.interaction,
                      user_name(user_interaction),
                      event_name(user_interaction),
                      team_name(user_interaction),
                      submission_id(user_interaction),
                      submission_name(user_interaction),
                      submission_file_name(user_interaction),
                      submission_similarity(user_interaction),
                      submission_diff_with_link(user_interaction)
                      ])}
        for user_interaction in user_interactions
    ]
    user_interactions_df = pd.DataFrame(
        user_interactions_dict_list, columns=columns)
    user_interactions_df = user_interactions_df.sort_values(
        'timestamp (UTC)', ascending=False)
    return user_interactions_df


def get_user_interactions():
    """Create user interaction table.

    Returns
    -------
    user_interactions_html : html string
    """
    user_interactions_df = get_user_interactions_df()
    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        # classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    user_interactions_html = user_interactions_df.to_html(**html_params)
    return table_format(user_interactions_html)


def get_source_submissions(submission):
    submissions = Submission.query.filter_by(
        event_team=submission.event_team).all()
    users = get_team_members(submission.team)
    for user in users:
        user_interactions = UserInteraction.query.filter_by(
            user=user, interaction='looking at submission').all()
        submissions += [user_interaction.submission for
                        user_interaction in user_interactions if
                        user_interaction.event == submission.event]
    submissions = list(set(submissions))
    submissions = [s for s in submissions if
                   s.submission_timestamp < submission.submission_timestamp]
    submissions.sort(key=lambda x: x.submission_timestamp, reverse=True)
    return submissions
