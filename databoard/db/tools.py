import os
import bcrypt
import pandas as pd

from databoard.db.model import db, User, Team, Submission, SubmissionFile,\
    CVFold, SubmissionOnCVFold
from databoard.db.model import NameClashError, MergeTeamError,\
    DuplicateSubmissionError, max_members_per_team
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

import databoard.config as config
import databoard.generic as generic
import databoard.train_test as train_test


def date_time_format(date_time):
    return date_time.strftime('%a %Y-%m-%d %H:%M:%S')


######################### Users ###########################

def get_user_teams(user):
    teams = db.session.query(Team).all()
    for team in teams:
        if user in get_team_members(team):
            yield team


def get_n_user_teams(user):
    return len(get_user_teams(user))


def get_hashed_password(plain_text_password):
    """Hash a password for the first time
    (Using bcrypt, the salt is saved into the hash itself)"""
    return bcrypt.hashpw(plain_text_password, bcrypt.gensalt())


def check_password(plain_text_password, hashed_password):
    """Check hased password. Using bcrypt, the salt is saved into the
    hash itself"""
    return bcrypt.checkpw(plain_text_password, hashed_password)


def create_user(name, password, lastname, firstname, email):
    hashed_password = get_hashed_password(password)
    user = User(name=name, hashed_password=hashed_password,
                lastname=lastname, firstname=firstname, email=email)
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
            db.session.query(User).filter_by(name=name).one()
            message += 'username is already in use'
        except NoResultFound:
            # We only check for team names if username is not in db
            try:
                db.session.query(Team).filter_by(name=name).one()
                db.message += 'username is already in use as a team name'
            except NoResultFound:
                pass
        try:
            db.session.query(User).filter_by(email=email).one()
            if len(message) > 0:
                message += ' and '
            message += 'email is already in use'
        except NoResultFound:
            pass
        if len(message) > 0:
            raise NameClashError(message)
        else:
            raise e
    return user


def validate_user(user):
    # from 'asked' to 'user'
    user.access_level = 'user'


def print_users():
    print('***************** List of users ****************')
    for user in db.session.query(User).order_by(User.id_):
        print('{} belongs to teams:'.format(user))
        for team in get_user_teams(user):
            print('\t{}'.format(team))


######################### Teams ###########################


def get_team_members(team):
    if team.initiator is not None:
        # "yield from" in Python 3.3
        for member in get_team_members(team.initiator):
            yield member
        for member in get_team_members(team.acceptor):
            yield member
    else:
        yield team.admin


def get_n_team_members(team):
    return len(list(get_team_members(team)))


def merge_teams(name, initiator_name, acceptor_name):
    initiator = db.session.query(Team).filter_by(name=initiator_name).one()
    acceptor = db.session.query(Team).filter_by(name=acceptor_name).one()
    if not initiator.is_active:
        raise MergeTeamError('Merge initiator is not active')
    if not acceptor.is_active:
        raise MergeTeamError('Merge acceptor is not active')
    # Meaning they are already in another team for the ramp

    # Testing if team size is <= max_members_per_team
    n_members_initiator = get_n_team_members(initiator)
    n_members_acceptor = get_n_team_members(acceptor)
    n_members_new = n_members_initiator + n_members_acceptor
    if n_members_new > max_members_per_team:
        raise MergeTeamError(
            'Too big team: new team would be of size {}, the max is {}'.format(
                n_members_new, max_members_per_team))

    members_initiator = get_team_members(initiator)
    members_acceptor = get_team_members(acceptor)

    # Testing if team (same members) exists under a different name. If the
    # name is the same, we break. If the loop goes through, we add new team.
    members_set = set(members_initiator).union(set(members_acceptor))
    for team in db.session.query(Team):
        if members_set == set(get_team_members(team)):
            if name == team.name:
                break  # ok, but don't add new team, just set them to inactive
            raise MergeTeamError(
                'Team exists with the same members, team name = {}'.format(
                    team.name))
    else:
        team = Team(name=name, admin=initiator.admin,
                    initiator=initiator, acceptor=acceptor)
        db.session.add(team)
    initiator.is_active = False
    acceptor.is_active = False
    try:
        db.session.commit()
    except IntegrityError as e:
        db.session.rollback()
        try:
            db.session.query(Team).filter_by(name=name).one()
            raise NameClashError('team name is already in use')
        except NoResultFound:
            raise e
    return team


def print_active_teams():
    print('***************** List of active teams ****************')
    for team in db.session.query(Team).filter(Team.is_active):
        print('{} members:'.format(team))
        for member in get_team_members(team):
            print('\t{}'.format(member))


######################### Submissions ###########################


def make_submission(team_name, name, f_name_list):
    # TODO: to call unit tests on submitted files. Those that are found
    # in the table that describes the workflow. For the rest just check
    # maybe size 
    team = db.session.query(Team).filter_by(name=team_name).one()
    submission = db.session.query(Submission).filter_by(
        name=name, team=team).one_or_none()
    cv_folds = db.session.query(CVFold).all()
    if submission is None:
        submission = Submission(name=name, team=team)
        # Adding submission files
        for f_name in f_name_list:
            submission_file = SubmissionFile(
                name=f_name, submission=submission)
            db.session.add(submission_file)
        db.session.add(submission)
        # Adding (empty) submission on cv folds
        for cv_fold in cv_folds:
            submission_on_cv_fold = SubmissionOnCVFold(
                submission=submission, cv_fold=cv_fold)
            db.session.add(submission_on_cv_fold)
        db.session.commit()
    else:
        # We allow resubmit for new or failing submissions
        if submission.state == 'new' or 'error' in submission.state:
            submission.state = 'new'
            # Updating existing files or adding new if not in list
            for f_name in f_name_list:
                submission_file = db.session.query(SubmissionFile).filter_by(
                    name=f_name, submission=submission).one_or_none()
                if submission_file is None:
                    submission_file = SubmissionFile(
                        name=f_name, submission=submission)
                    db.session.add(submission_file)
                # else copy the file should be there
                # right now we don't delete files that were not resubmitted,
                # allowing partial resubmission

            # Updating submission on cv folds
            submission_on_cv_folds = db.session.query(
                SubmissionOnCVFold).filter(
                    SubmissionOnCVFold.submission == submission).all()
            for submission_on_cv_fold in submission_on_cv_folds:
                # couldn't figure out how to reset to default values
                db.session.delete(submission_on_cv_fold)
                db.session.add(submission_on_cv_fold)
            db.session.commit()
#            for cv_fold in cv_folds:
#                submission_on_cv_fold = SubmissionOnCVFold(
#                    submission=submission, cv_fold=cv_fold)
#                db.session.add(submission_on_cv_fold)
#            db.session.commit()
        else:
            raise DuplicateSubmissionError(
                'Submission "{}" of team "{}" exists already'.format(
                    name, team_name))

    # We should copy files here
    return submission


def get_public_leaderboard():
    """
    Returns
    -------
    leaderboard_html : html string
    """

    # We can't query on non-hybrid properties like Submission.name_with_link,
    # so first we make the join then we extract the class members,
    # including @property members (that can't be compiled into
    # SQL queries)
    submissions_teams = db.session.query(Submission, Team).filter(
        Team.id_ == Submission.team_id).filter(
        Submission.is_public_leaderboard).all()
    columns = ['team',
               'submission',
               'valid score',
               'contributivity',
               'train time',
               'test time',
               'submitted at (UTC)']
    leaderboard_dict_list = [
        {column: value for column, value in zip(
            columns, [team.name,
                      submission.name_with_link,
                      round(submission.valid_score_cv_mean, 2),
                      submission.contributivity,
                      int(submission.train_time_cv_mean + 0.5),
                      int(submission.valid_time_cv_mean + 0.5),
                      date_time_format(submission.submission_timestamp)])}
        for submission, team in submissions_teams
    ]
    leaderboard_df = pd.DataFrame(leaderboard_dict_list, columns=columns)
    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )
    leaderboard_html = leaderboard_df.to_html(**html_params)
    return leaderboard_html


def print_submissions():
    print('***************** List of submissions ****************')
    submissions = db.session.query(Submission).order_by(Submission.id_).all()
    for submission in submissions:
        print submission
        submission_on_cv_folds = db.session.query(SubmissionOnCVFold).filter(
            SubmissionOnCVFold.submission == submission).all()
        for submission_on_cv_fold in submission_on_cv_folds:
            print '\t' + str(submission_on_cv_fold)


def print_cv_folds():
    print('***************** CV folds ****************')
    cv_folds = db.session.query(CVFold).all()
    for i, cv_fold in enumerate(cv_folds):
        print i, cv_fold


######################### CVFold ###########################


def add_cv_folds(cv):
    for train_is, test_is in cv:
        cv_fold = CVFold(train_is=train_is, test_is=test_is)
        db.session.add(cv_fold)
    db.session.commit()
