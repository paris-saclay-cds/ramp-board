import sys
if '' not in sys.path:  # Balazs bug
    sys.path.insert(0, '')

# for pickling theano
# sys.setrecursionlimit(50000)

import os
import logging
import numpy as np
import pandas as pd
from distutils.util import strtobool
# import fabric.contrib.project as project
# from fabric.api import *
# from fabric.contrib.files import exists
# from fabric.state import connections

# DON'T IMPORT ANYTHING FROM DATABOARD HERE !

# Open ports in Stratuslab
# 22, 80, 389, 443, 636, 2135, 2170, 2171, 2172, 2811, 3147, 5001, 5010, 5015,
# 8080, 8081, 8095, 8188, 8443, 8444, 9002, 10339, 10636, 15000, 15001, 15002,
# 15003, 15004, 20000-25000.

# the user to use for the remote commands
# env.user = config.get_ramp_field('deploy_user')
# env.use_ssh_config = True

# the servers where the commands are executed
# env.hosts = [config.get_ramp_field('train_user') + '@' + config.get_ramp_field('train_server'),
#             config.get_ramp_field('web_user') + '@' + config.get_ramp_field('web_server'),]

# production = env.hosts[0]
logger = logging.getLogger('databoard')



def clear_cache():
    from sklearn.externals.joblib import Memory
    from databoard.config import cachedir

    logger.info('Flushing the joblib cache.')
    mem = Memory(cachedir=cachedir)
    mem.clear()


def clear_db():
    from databoard.model_shelve import columns
    from databoard.model_shelve import shelve_database

    logger.info('Clearing the database.')
    with shelve_database('c') as db:
        db.clear()
        db['models'] = pd.DataFrame(columns=columns)
        db['leaderboard1'] = pd.DataFrame(columns=['score'])
        db['leaderboard2'] = pd.DataFrame(columns=['contributivity'])


def clear_registrants():
    import shutil
    from databoard.config import repos_path, submissions_path
    # Prepare the teams repo submodules
    # logger.info('Init team repos git')
    # repo = Repo.init(config.repos_path)  # does nothing if already exists

    # Remove the git repos of the teams
    logger.info('Clearing the teams repositories.')
    shutil.rmtree(repos_path, ignore_errors=True)
    os.mkdir(repos_path)

    logger.info('Clearing the models directory.')
    shutil.rmtree(submissions_path, ignore_errors=True)
    os.mkdir(submissions_path)
    open(os.path.join(submissions_path, '__init__.py'), 'a').close()


def clear_pred_files():
    import glob
    from databoard.config import submissions_path
    fnames = glob.glob(
        os.path.join(submissions_path, '*', '*', '*', '*.csv'))

    for fname in fnames:
        if os.path.exists(fname):
            logger.info("Removing {}".format(fname))
            os.remove(fname)


def prepare_data():
    import databoard.config as config
    import databoard.db_tools as db_tools
    specific = config.config_object.specific

    # Preparing the data set, typically public train/private held-out test cut
    logger.info('Preparing the dataset.')
    specific.prepare_data()
    logger.info('Adding CV folds.')
    _, y_train = specific.get_train_data()
    cv = specific.get_cv(y_train)
    db_tools.add_cv_folds(cv)


def test_setup():
    from databoard.remove_test_db import recreate_test_db

    recreate_test_db()
    prepare_data()
    clear_registrants()
    # Flush joblib cache
    clear_cache()
    # todo: set up database


def print_db(table='models', state=None):
    from databoard.model_shelve import shelve_database

    with shelve_database('c') as db:
        if table not in db:
            print('Select one of the following tables:')
            print '\n'.join('\t- {}'.format(t) for t in db)
            return
        df = db[table]
    pd.set_option('display.max_rows', len(df))
    if not state:
        print df
    else:
        print df[df.state == state]


def fetch():
    from databoard.fetch import fetch_models
    fetch_models()


def add_models():
    from databoard.fetch import add_models
    add_models()


def repeat_fetch(delay='60'):
    import time
    while True:
        fetch()
        delay = int(os.getenv('FETCH_DELAY', delay))
        time.sleep(delay)


def check(state=False, tag=None, team=None):
    from databoard.train_test import check_models
    from databoard.model_shelve import shelve_database

    with shelve_database() as db:
        models = db['models']

    if tag is not None:
        models = models[models.model == tag]
        state = 'all'  # force train all the selected models
        if len(models) == 0:
            print('No existing model with the tag: {}'.format(tag))
            return

    if team is not None:
        models = models[models.team == team]
        state = 'all'  # force train all the selected models
        if len(models) == 0:
            print('No existing model with the team: {}'.format(tag))
            return

    if not state:
        state = 'new'

    if state != 'all':
        models = models[models.state == state]

    check_models(models)

    idx = models.index

    with shelve_database() as db:
        db['models'].loc[idx, :] = models


def train(state=False, tag=None, team=None):
    from databoard.train_test import train_and_valid_models
    from databoard.model_shelve import shelve_database

    with shelve_database() as db:
        models = db['models']

    if tag is not None:
        models = models[models.model == tag]
        state = 'all'  # force train all the selected models
        if len(models) == 0:
            print('No existing model with the tag: {}'.format(tag))
            return

    if team is not None:
        models = models[models.team == team]
        state = 'all'  # force train all the selected models
        if len(models) == 0:
            print('No existing model with the team: {}'.format(tag))
            return

    if not state:
        state = 'new'

    if state != 'all':
        models = models[models.state == state]

    train_and_valid_models(models)

    idx = models.index

    with shelve_database() as db:
        db['models'].loc[idx, :] = models


def test(state=False, tag=None, team=None):
    from databoard.train_test import test_submissions
    from databoard.model_shelve import shelve_database

    with shelve_database() as db:
        models = db['models']

    if tag is not None:
        models = models[models.model == tag]
        state = 'all'  # force train all the selected models
        if len(models) == 0:
            print('No existing model with the tag: {}'.format(tag))
            return

    if team is not None:
        models = models[models.team == team]
        state = 'all'  # force train all the selected models
        if len(models) == 0:
            print('No existing model with the team: {}'.format(tag))
            return

    if not state:
        state = 'trained'

    if state != 'all':
        models = models[models.state == state]

    test_submissions(models)

    idx = models.index

    with shelve_database() as db:
        db['models'].loc[idx, :] = models


def train_test(t=None, s=None, force='False'):
    force = strtobool(force)
    team_name = t
    submission_name = s

    from databoard.db_tools import train_test_submissions, get_team_submissions

    if team_name is None:
        train_test_submissions(force_retrain_test=force)  # All
    else:
        if submission_name is None:
            train_test_submissions(
                get_team_submissions(team_name), force_retrain_test=force)
        else:
            train_test_submissions(
                get_team_submissions(
                    team_name, submission_name), force_retrain_test=force)
    compute_contributivity()


def compute_contributivity():
    from databoard.db_tools import compute_contributivity
    compute_contributivity()


def print_submissions(t=None, s=None):
    team_name = t
    submission_name = s

    from databoard.db_tools import print_submissions, get_team_submissions
    if team_name is None:
        print_submissions()  # All
    else:
        if submission_name is None:
            print_submissions(get_team_submissions(team_name))
        else:
            print_submissions(get_team_submissions(team_name, submission_name))


def set_state(t, s, state):
    team_name = t
    submission_name = s

    from databoard.db_tools import set_state
    set_state(team_name, submission_name, state)


def delete_submission(t, s):
    team_name = t
    submission_name = s

    from databoard.db_tools import delete_submission
    delete_submission(team_name, submission_name)


def kill(team, tag):
    import glob
    import signal
    from databoard.fetch import get_tag_uid
    from databoard.config import config_object

    answer = 'y'
    while answer != 'y':
        answer = raw_input('Sure? (y/n): ')

    pid_filenames = os.path.join(
        config_object.submissions_path, team, get_tag_uid(team, tag), 'pid_*')
    print pid_filenames
    for f in glob.glob(pid_filenames):
        with open(f) as pid_file:
            pid = pid_file.read()
            os.kill(int(pid), signal.SIGKILL)


def serve(port=None, test='False'):
    test = strtobool(test)
    from databoard import app
    from databoard.config import config_object
    # loads url/function bindings through @app.route decorators
    import databoard.views  # noqa

    if test:
        pass
    if port is None:
        server_port = int(config_object.server_port)
    else:
        server_port = int(port)
    app.run(
        debug=False,
        port=server_port,
        use_reloader=False,
        host='0.0.0.0')


# TODO: fill up the following functions so to easily deploy
# databoard on the server

# FIXME: dtach not working
# @hosts(production)
# def rserve(sockname="db_server"):
#    if not exists("/usr/bin/dtach"):
#        sudo("apt-get install dtach")
#
#    with cd(config.dest_path):
#        # run('export SERV_PORT={}'.format(server_port))
#        # run('fab serve')
#        # run('dtach -n `mktemp -u /tmp/{}.XXXX` export SERV_PORT={};fab serve'.format(sockname, server_port))
#        return run('dtach -n `mktemp -u /tmp/{}.XXXX` fab serve:port={}'.format(sockname, server_port))

# from importlib import import_module


ramp_deployment = [
    'fabfile.py',
    'ramp_index.txt',
]

software = [
    'databoard/__init__.py',
    'databoard/base_prediction.py',
    'databoard/config.py',
    'databoard/db_tools.py',
    'databoard/fetch.py',
    'databoard/forms.py',
    'databoard/generic.py',
    'databoard/leaderboard.py',
    'databoard/machine_parallelism.py',
    'databoard/model.py',
    'databoard/multiclass_prediction.py',
    'databoard/regression_prediction.py',
    'databoard/remove_test_db.py',
    'databoard/scores.py',
    'databoard/train_test.py',
    'databoard/views.py',
    'databoard/ramps/__init__.py',
    'databoard/ramps/air_passengers/__init__.py',
    'databoard/ramps/air_passengers/specific.py',
    'databoard/ramps/boston_housing/__init__.py',
    'databoard/ramps/boston_housing/specific.py',
    'databoard/ramps/el_nino_bagged_cv_future/__init__.py',
    'databoard/ramps/el_nino_bagged_cv_future/specific.py',
    'databoard/ramps/el_nino_block_cv/__init__.py',
    'databoard/ramps/el_nino_block_cv/specific.py',
    'databoard/ramps/iris/__init__.py',
    'databoard/ramps/iris/specific.py',
    'databoard/ramps/kaggle_otto/__init__.py',
    'databoard/ramps/kaggle_otto/specific.py',
    'databoard/ramps/mortality_prediction/__init__.py',
    'databoard/ramps/mortality_prediction/specific.py',
    'databoard/ramps/pollenating_insects/__init__.py',
    'databoard/ramps/pollenating_insects/specific.py',
    'databoard/ramps/variable_stars/__init__.py',
    'databoard/ramps/variable_stars/specific.py',
    'databoard/tests/test_model.py',
    'databoard/tests/test_multiclass_predictions.py',
    'databoard/tests/test_regression_predictions.py',
    'databoard/tests/test_scores.py',
    'databoard/static',
    'databoard/templates',
]


def _save_ramp_index(ramp_index):
    # we save ramp_index in the main dir so the deplyment can query itself
    # for example, in serve (to get the port number) and sepcific (to get
    # the number of CPUs). generic.get_ramp_index() reads it in
    with open('ramp_index.txt', 'w') as f:
        f.write(ramp_index)


def publish(ramp_index, test='False'):
    test = strtobool(test)
    _save_ramp_index(ramp_index)

    # don't import before saving the ramp_index.txt file
    from databoard.config import config_object, deposited_submissions_d_name,\
        sandbox_d_name
    ramp_name = config_object.ramp_name
    # TODO: check if ramp_name is the same as in
    #      'ramps/' + ramp_name + '/specific.py'

    # publishing ramp_deployment
    local = config_object.train_server == 'localhost'
    command = "rsync -pthrRvz -c "
    if not local:
        command += "--rsh=\'ssh -i " + os.path.expanduser("~")
        command += "/.ssh/datacamp/id_rsa -p 22\' "
    for file in ramp_deployment:
        command += file + " "
    if not config_object.is_same_web_and_train_servers():
        command += config_object.get_deployment_target('web')
    else:
        command += config_object.get_deployment_target('train')
    print command
    os.system(command)

    command = "rsync -pthrvz -c "
    if not local:
        command += "--rsh=\'ssh -i " + os.path.expanduser("~")
        command += "/.ssh/datacamp/id_rsa -p 22\' "
    command += " ramps/" + ramp_name + '/' + sandbox_d_name + ' '
    if not config_object.is_same_web_and_train_servers():
        command += config_object.get_deployment_target('web')
    else:
        command += config_object.get_deployment_target('train')
    print command
    os.system(command)

    # publishing software
    if not local:
        command = "rsync -pthrRvz -c "
        command += "--rsh=\'ssh -i " + os.path.expanduser("~")
        command += "/.ssh/datacamp/id_rsa -p 22\' "
        for file in software:
            command += file + " "
        if not config_object.is_same_web_and_train_servers():
            command += config_object.get_software_target('web')
        else:
            command += config_object.get_software_target('train')
        print command
        os.system(command)

    if test:
        command = "rsync -pthrvz -c "
        if not local:
            command += "--rsh=\'ssh -i " + os.path.expanduser("~")
            command += "/.ssh/datacamp/id_rsa -p 22\' "
        command += " ramps/" + ramp_name + '/' +\
            deposited_submissions_d_name + ' '
        if not config_object.is_same_web_and_train_servers():
            command += config_object.get_deployment_target('web')
        else:
            command += config_object.get_deployment_target('train')
        print command
        os.system(command)

"""
    # rsyncing specific
    command = "rsync -pthrvz -c "
    if not local:
        command += "--rsh=\'ssh -i " + os.path.expanduser("~")
        command += "/.ssh/datacamp/id_rsa -p 22\' "
    command += " ramps/" + ramp_name + "/specific.py "
    if not config_object.is_same_web_and_train_servers():
        command += config_object.get_deployment_target('web')
        command += "/databoard/"
    else:
        command += config_object.get_deployment_target('train')
        command += "/databoard/"
    print command
    os.system(command)
"""



# (re)publish data set from 'ramps/' + ramp_name + '/data'
# fab prepare_data should be run at the destination
def publish_data(ramp_index):
    from databoard.config import config_object
    local = config_object.train_server == 'localhost'
    ramp_name = config_object.ramp_name
    command = "rsync --delete -pthrvz -c "
    if not local:
        command += "--rsh=\'ssh -i " + os.path.expanduser("~")
        command += "/.ssh/datacamp/id_rsa -p 22\' "
    command += 'ramps/' + ramp_name + '/data '
    if not config_object.is_same_web_and_train_servers():
        command += config_object.get_deployment_target('web') + "/"
    else:
        command += config_object.get_deployment_target('train') + "/"
    print command
    os.system(command)


def _clear_destination_path_aux(destination_path, destination_root):
    if os.path.exists(destination_path):
        os.system('rm -rf ' + destination_path + '/*')
    else:
        if not os.path.exists(destination_root):
            os.mkdir(destination_root)
        os.mkdir(destination_path)


def clear_destination_path(ramp_index):
    _save_ramp_index(ramp_index)  # todo before importing databoard

    from databoard.config import config_object
    if not config_object.is_same_web_and_train_servers():
        # destination_path eg
        # '/tmp/databoard_local/databoard_mortality_prediction_8080'
        destination_path = config_object.web_destination_path
        destination_root = config_object.web_root
        _clear_destination_path_aux(destination_path, destination_root)
    destination_path = config_object.train_destination_path
    destination_root = config_object.train_root
    _clear_destination_path_aux(destination_path, destination_root)


# For now, test is two-phased: 1) publish_test<ramp_index> first, which is a
# publish and publish_data that clears the destination first completely, and 2)
# go to the destination, and test_ramp there. It's because I don't know
# how to make import path (for specific and user submission) run time. If
# everything is deployed from a database, we can have single test commands that
# publish and test locally and remotely using fabric magic.
def publish_test(ramp_index):
    clear_destination_path(ramp_index)
    publish(ramp_index, test='True')
    publish_data(ramp_index)


def test_ramp():
    import databoard.config as config
    from databoard.tests.test_model import test_create_user, test_merge_teams
    min_duration_between_submissions = config.min_duration_between_submissions
    config.min_duration_between_submissions = 0
    test_setup()
    test_create_user()  # kegl, agramfort, akazakci, mcherti, pwd = 'bla'
    test_merge_teams()  # kemfort, mkezakci
    add_models()  # will create user and team 'test'
    train_test()
    config.min_duration_between_submissions = min_duration_between_submissions
    # leaderboard(test='True')


def create_user(name, password, lastname, firstname, email, 
                access_level='user', hidden_notes=''):
    from databoard.config import sandbox_d_name
    from databoard.db_tools import create_user
    create_user(
        name, password, lastname, firstname, email, access_level, hidden_notes)
    train_test(t=name, s=sandbox_d_name)


def add_users_from_file(users_to_add_f_name):
    from databoard.db_tools import add_users_from_file
    from databoard.remove_test_db import recreate_test_db
    from databoard.config import sandbox_d_name

    test_setup()
    users_to_add = add_users_from_file(users_to_add_f_name)

    for _, u in users_to_add.iterrows():
        print u
        create_user(u['name'], u['password'], u['lastname'], u['firstname'],
                    u['email'], u['access_level'], u['hidden_notes'])

def send_password_mails(users_to_add_f_name):
    from databoard.db_tools import send_password_mails
    send_password_mails(users_to_add_f_name)
