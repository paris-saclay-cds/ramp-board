# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

import os
import sys
import git
import glob
import shutil
import logging
import hashlib 
import numpy as np
import pandas as pd

from flask_mail import Mail
from flask_mail import Message

from databoard import app
from .model import shelve_database, columns, ModelState
from .generic import changedir
import specific
from .config_databoard import repos_path, root_path, tag_len_limit, notification_recipients, server_name, models_path


logger = logging.getLogger('databoard')

def get_tag_uid(team_name, tag_name, **kwargs):
    sha_hasher = hashlib.sha1()
    sha_hasher.update(team_name)
    sha_hasher.update(tag_name)
    tag_name_alias = 'm{}'.format(sha_hasher.hexdigest())
    return tag_name_alias

def send_mail_notif(submissions):
    with app.app_context():
        mail = Mail(app)

        logger.info('Sending notification email to: {}'.format(', '.join(notification_recipients)))
        msg = Message('New submissions in the ' + specific.hackaton_title + ' hackaton', 
            reply_to='djalel.benbouzid@gmail.com')

        msg.recipients = notification_recipients

        body_message = '<b>Dataset</b>: {}<br/>'.format(specific.hackaton_title)
        body_message += '<b>Server</b>: {}<br/>'.format(specific.hserver_name)
        body_message += '<b>Folder</b>: {}<br/>'.format(os.path.abspath(root_path))

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
    base_path = repos_path
    repo_paths = sorted(glob.glob(os.path.join(base_path, '*')))

    if not os.path.exists(models_path):
        logger.warning("Models folder didn't exist. An empty folder was created.")
        os.mkdir(models_path)
    open(os.path.join(models_path, '__init__.py'), 'a').close()

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
            tag_name = t.name
            tag_name_alias = get_tag_uid(team_name, tag_name)
            # We delete tags of failed submissions, so they 
            # can be refetched
            if tag_name_alias in old_failed_submissions:
                logger.debug('Deleting local tag: {}'.format(tag_name))
                repo.delete_tag(tag_name)

        try:
            # just in case the working tree is dirty
            repo.head.reset(index=True, working_tree=True)
        except Exception as e:
            logger.error('Unable to reset to HEAD: \n{}'.format(e))

        try:
            repo.remotes.origin.pull()
        except Exception as e:
            logger.error('Unable to pull from repo. Possibly no connexion: \n{}'.format(e))

        repo_path = os.path.join(models_path, team_name)
        if not os.path.exists(repo_path):
            os.mkdir(repo_path)
        open(os.path.join(repo_path, '__init__.py'), 'a').close()

        if len(repo.tags) == 0:
            logger.debug('No tag found for %s' % team_name)

        for t in repo.tags:
            
            # FIXME: this huge try-except is a nightmare
            try:
                tag_name = t.name
                logger.debug('Tag name: {}'.format(tag_name))

                # will serve as dataframe index
                tag_name_alias = get_tag_uid(team_name, tag_name)
                logger.debug('Tag alias: {}'.format(tag_name_alias))
                model_path = os.path.join(repo_path, tag_name_alias)

                new_commit_time = t.commit.committed_date

                with shelve_database() as db:

                    # skip if the model is trained, otherwise, replace the entry with a new one
                    if tag_name_alias in db['models'].index:
                        if db['models'].loc[tag_name_alias, 'state'] in \
                            ['tested', 'trained', 'ignore']:
                            continue
                        elif db['models'].loc[tag_name_alias, 'state'] == 'error':

                            # if the failed model timestamp has changed
                            if db['models'].loc[tag_name_alias, 'timestamp'] < new_commit_time:
                                db['models'].drop(tag_name_alias, inplace=True)
                                old_failed_submissions.remove(tag_name_alias)
                            else:
                                new_submissions.add(tag_name_alias)
                                continue
                        else:
                            # default case
                            db['models'].drop(tag_name_alias, inplace=True)

                    new_submissions.add(tag_name_alias)

                    # recursively copy the model files
                    copy_git_tree(t.object.tree, model_path)
                    open(os.path.join(model_path, '__init__.py'), 'a').close()

                    relative_path = os.path.join(team_name, tag_name_alias)

                    # listing the model files
                    file_listing = [f for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))]

                    # filtering useless files
                    file_listing = filter(lambda f: not f.startswith('__'), file_listing)
                    file_listing = filter(lambda f: not f.endswith('.pyc'), file_listing)
                    file_listing = filter(lambda f: not f.endswith('.csv'), file_listing)
                    file_listing = filter(lambda f: not f.endswith('error.txt'), file_listing)
                    file_listing = '|'.join(file_listing)

                    # prepre a dataframe for the concatnation 
                    new_entry = pd.DataFrame({
                        'team': team_name, 
                        'model': tag_name, 
                        'timestamp': new_commit_time, 
                        'state': "new",
                        'listing': file_listing,
                    }, index=[tag_name_alias])

                    # set a list into a cell
                    # new_entry.set_value(tag_name_alias, 'listing', file_listing)
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
        really_new_submissions = df.loc[new_submissions - old_submissions][['team', 'model']].values

    if len(really_new_submissions):
        try:
            send_mail_notif(really_new_submissions)
        except:
            logger.error('Unable to send email notifications for new models.')
    else:
        logger.debug('No new submission.')
