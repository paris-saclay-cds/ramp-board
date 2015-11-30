# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

import os
import git
import glob
import shutil
import logging
import hashlib
import pandas as pd

from flask_mail import Mail
from flask_mail import Message

from databoard import app
from databoard.model_shelve import shelve_database, columns
from databoard.config import notification_recipients, submissions_path,\
    repos_path, deposited_submissions_path
import databoard.config as config
from databoard.model import NameClashError
import databoard.db_tools as db_tools

logger = logging.getLogger('databoard')


def get_model_hash(team_name, submission_name, **kwargs):
    sha_hasher = hashlib.sha1()
    sha_hasher.update(team_name)
    sha_hasher.update(submission_name)
    model_hash = 'm{}'.format(sha_hasher.hexdigest())
    return model_hash


def send_mail_notif(submissions):
    specific = config.config_object.specific

    with app.app_context():
        mail = Mail(app)

        logger.info('Sending notification email to: {}'.format(
            ', '.join(notification_recipients)))
        msg = Message('New submissions in the ' + specific.hackaton_title + ' hackaton',
                      reply_to='djalel.benbouzid@gmail.com')

        msg.recipients = notification_recipients

        body_message = '<b>Dataset</b>: {}<br/>'.format(
            specific.hackaton_title)
        body_message += '<b>Server</b>: {}<br/>'.format(
            config.config.deploy_server)
        body_message += '<b>Port</b>: {}<br/>'.format(
            config.config.server_port)
        body_message += '<b>Path</b>: {}<br/>'.format(
            config.config.get_destination_path())  # XXX buggy
        body_message += '<b>Num_CPUs</b>: {}<br/>'.format(
            config.config.num_cpus)

        body_message += 'New submissions: <br/><ul>'
        for team, tag in submissions:
            body_message += '<li><b>{}</b>: {}</li>'.format(team, tag)
        body_message += '</ul>'
        msg.html = body_message

        mail.send(msg)


def copy_git_tree(tree, dest_folder):
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    for file_elem in tree.blobs:
        with open(os.path.join(dest_folder, file_elem.name), 'w') as f:
            shutil.copyfileobj(file_elem.data_stream, f)
    for tree_elem in tree.trees:
        copy_git_tree(tree_elem, os.path.join(dest_folder, tree_elem.name))


def fetch_models():
    repo_paths = sorted(glob.glob(os.path.join(repos_path, '*')))

    if not os.path.exists(submissions_path):
        logger.warning(
            "Models folder didn't exist. An empty folder was created.")
        os.mkdir(submissions_path)
    open(
        os.path.join(submissions_path, '__init__.py'), 'a').close()

    new_submissions = set()  # a set of submission hashes

    # create the database if it doesn't exist
    with shelve_database() as db:
        if 'models' not in db:
            db['models'] = pd.DataFrame(columns=columns)
        models = db['models']
        old_submissions = set(models.index)
        old_failed_submissions = set(models[models['state'] == 'error'].index)

    for repo_path in repo_paths:

        logger.debug('Repo name: {}'.format(repo_path))

        if not os.path.isdir(repo_path):
            continue

        team_name = os.path.basename(repo_path)
        try:
            repo = git.Repo(repo_path)
        except Exception as e:
            logger.error('Problem when reading a repo: \n{}'.format(e))
            continue

        for t in repo.tags:
            submission_name = t.name
            model_hash = get_model_hash(team_name, submission_name)
            # We delete tags of failed submissions, so they
            # can be refetched
            if model_hash in old_failed_submissions:
                logger.debug('Deleting local tag: {}'.format(submission_name))
                repo.delete_tag(submission_name)

        try:
            # just in case the working tree is dirty
            repo.head.reset(index=True, working_tree=True)
        except Exception as e:
            logger.error('Unable to reset to HEAD: \n{}'.format(e))

        try:
            repo.remotes.origin.pull()
        except Exception as e:
            logger.error(
                'Unable to pull from repo. Possibly no connexion: \n{}'.format(e))

        repo_path = os.path.join(config.submissions_path, team_name)
        if not os.path.exists(repo_path):
            os.mkdir(repo_path)
        open(os.path.join(repo_path, '__init__.py'), 'a').close()

        if len(repo.tags) == 0:
            logger.debug('No tag found for %s' % team_name)

        for t in repo.tags:

            # FIXME: this huge try-except is a nightmare
            try:
                submission_name = t.name
                # submission_name = submission_name.replace(' ', '_')
                logger.debug('Tag name: {}'.format(submission_name))

                # will serve as dataframe index
                model_hash = get_model_hash(team_name, submission_name)
                logger.debug('Tag alias: {}'.format(model_hash))
                model_path = os.path.join(repo_path, model_hash)

                new_commit_time = t.commit.committed_date

                with shelve_database() as db:

                    # skip if the model is trained, otherwise, replace the
                    # entry with a new one
                    if model_hash in db['models'].index:
                        if db['models'].loc[model_hash, 'state'] in \
                                ['tested', 'trained', 'ignore']:
                            continue
                        elif db['models'].loc[model_hash, 'state'] == 'error':

                            # if the failed model timestamp has changed
                            if db['models'].loc[model_hash, 'timestamp'] < new_commit_time:
                                db['models'].drop(model_hash, inplace=True)
                                old_failed_submissions.remove(model_hash)
                            else:
                                new_submissions.add(model_hash)
                                continue
                        else:
                            # default case
                            db['models'].drop(model_hash, inplace=True)

                    # recursively copy the model files
                    try:
                        copy_git_tree(t.object.tree, model_path)
                    except:
                        continue
                    open(os.path.join(model_path, '__init__.py'), 'a').close()

                    new_submissions.add(model_hash)
                    # relative_path = os.path.join(team_name, model_hash)

                    # listing the model files
                    file_listing = [f for f in os.listdir(
                        model_path) if os.path.isfile(os.path.join(model_path, f))]

                    # filtering useless files
                    file_listing = filter(
                        lambda f: not f.startswith('__'), file_listing)
                    file_listing = filter(
                        lambda f: not f.endswith('.pyc'), file_listing)
                    # file_listing = filter(lambda f: not f.endswith('.csv'), file_listing)
                    file_listing = filter(
                        lambda f: not f.endswith('error.txt'), file_listing)
                    file_listing = '|'.join(file_listing)

                    # prepre a dataframe for the concatnation
                    new_entry = pd.DataFrame({
                        'team': team_name,
                        'model': submission_name,
                        'timestamp': new_commit_time,
                        'state': "new",
                        'listing': file_listing,
                    }, index=[model_hash])

                    # set a list into a cell
                    # new_entry.set_value(model_hash, 'listing', file_listing)
                    db['models'] = db['models'].append(new_entry)

            except Exception as e:
                raise
                logger.error("%s" % e)

    # remove the failed submissions that have been deleted
    removed_failed_submissions = old_failed_submissions - new_submissions
    with shelve_database() as db:
        db['models'].drop(removed_failed_submissions, inplace=True)

    # read-only
    # with shelve_database('r') as db:
    with shelve_database() as db:
        df = db['models']
        logger.debug(df)
        really_new_submissions = df.loc[
            new_submissions - old_submissions][['team', 'model']].values

    if len(really_new_submissions):
        try:
            send_mail_notif(really_new_submissions)
        except:
            logger.error('Unable to send email notifications for new models.')
    else:
        logger.debug('No new submission.')


def add_models():
    deposited_submission_paths = sorted(
        glob.glob(os.path.join(deposited_submissions_path, '*/*')))

    open(
        os.path.join(submissions_path, '__init__.py'), 'a').close()

    for deposited_submission_path in deposited_submission_paths:
        deposited_team_path, submission_name = os.path.split(
            deposited_submission_path)
        _, team_name = os.path.split(deposited_team_path)
        try:
            db_tools.create_user(
                name=team_name, password='bla', lastname=team_name,
                firstname=team_name, email=team_name + '@team_name.com')
        except NameClashError:  # test user already in db, no problem
            pass
        db_tools.make_submission_and_copy_all_files(
            team_name, submission_name, deposited_submission_path)
