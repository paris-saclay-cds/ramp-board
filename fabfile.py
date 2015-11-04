# for pickling theano
import sys
if '' not in sys.path:  # Balazs bug
    sys.path.insert(0, '')

#sys.setrecursionlimit(50000)

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


def create_test_db():
    import databoard.db.create_test_db


def clear_cache():
    from sklearn.externals.joblib import Memory
    from databoard.config import cachedir

    logger.info('Flushing the joblib cache.')
    mem = Memory(cachedir=cachedir)
    mem.clear()


def clear_db():
    from databoard.model import columns
    from databoard.model import shelve_database

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
    from databoard.specific import prepare_data

    # Preparing the data set, typically public train/private held-out test cut
    logger.info('Preparing the dataset.')
    prepare_data()


def setup():
    prepare_data()
    clear_db()
    clear_registrants()
    # Flush joblib cache
    clear_cache()
    # todo: set up database


def print_db(table='models', state=None):
    from databoard.model import shelve_database

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


def leaderboard(which='all', test='False'):
    test = strtobool(test)
    from databoard.model import shelve_database
    from databoard.leaderboard import (
        leaderboard_classical,
        leaderboard_combination,
        leaderboard_execution_times,
    )

    with shelve_database() as db:
        submissions = db['models']
        trained_models = submissions[
            np.any([submissions['state'] == "test_error",
                    submissions['state'] == "trained",
                    submissions['state'] == "tested"], axis=0)]
        tested_models = submissions[submissions['state'] == "tested"]
        pd.set_option('display.max_rows', len(trained_models))
        print trained_models

    if which in ('all', 'classical'):
        l1 = leaderboard_classical(trained_models)
        # The following assignments only work because
        # leaderboard_classical & co are idempotent.
        # FIXME (potentially)
        with shelve_database() as db:
            db['leaderboard1'] = l1
            if test:
                l_test = leaderboard_classical(tested_models, subdir="test")
                db['leaderboard_classical_test'] = l_test

    if which in ('all', 'combined'):
        l2 = leaderboard_combination(trained_models, test)
        # FIXME: same as above
        with shelve_database() as db:
            db['leaderboard2'] = l2

    if which in ('all', 'times'):
        l_times = leaderboard_execution_times(trained_models)
        # FIXME: same as above
        with shelve_database() as db:
            db['leaderboard_execution_times'] = l_times


def check(state=False, tag=None, team=None):
    from databoard.train_test import check_models
    from databoard.model import shelve_database

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
    from databoard.model import shelve_database

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
    from databoard.train_test import test_models
    from databoard.model import shelve_database

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

    test_models(models)

    idx = models.index

    with shelve_database() as db:
        db['models'].loc[idx, :] = models


def train_test(state=False, tag=None, team=None):
    from databoard.train_test import train_valid_and_test_models
    from databoard.model import shelve_database

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

    train_valid_and_test_models(models)

    idx = models.index

    with shelve_database() as db:
        db['models'].loc[idx, :] = models


def change_state(from_state, to_state):
    from databoard.model import shelve_database

    with shelve_database() as db:
        models = db['models']
    models = models[models['state'] == from_state]

    idx = models.index
    with shelve_database() as db:
        db['models'].loc[idx, 'state'] = to_state


def set_state(team, tag, state):
    from databoard.model import shelve_database

    with shelve_database() as db:
        models = db['models']
    models = models[np.logical_and(models['model'] == tag,
                                   models['team'] == team)]

    if len(models) > 1:
        print "ambiguous selection"
        return
    if len(models) == 0:
        print "no model found"
        return
    idx = models.index
    with shelve_database() as db:
        db['models'].loc[idx, 'state'] = state


def kill(team, tag):
    import glob
    import signal
    from databoard.fetch import get_tag_uid
    from databoard.config import config

    answer = 'y'
    while answer != 'y':
        answer = raw_input('Sure? (y/n): ')

    pid_filenames = os.path.join(
        config.submissions_path, team, get_tag_uid(team, tag), 'pid_*')
    print pid_filenames
    for f in glob.glob(pid_filenames):
        with open(f) as pid_file:
            pid = pid_file.read()
            os.kill(int(pid), signal.SIGKILL)


def serve(port=None):
    from databoard import app
    from databoard.config import config
    # loads url/function bindings through @app.route decorators
    import databoard.views  # noqa

    if port is None:
        server_port = int(config.server_port)
    else:
        server_port = int(port)
    app.run(
        debug=True,
        port=server_port,
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

software = [
    'fabfile.py',
    'ramp_index.txt',
    # 'databoard/config_databoard.py',
    'databoard/config.py',
    'databoard/__init__.py',
    'databoard/fetch.py',
    'databoard/generic.py',
    'databoard/leaderboard.py',
    'databoard/machine_parallelism.py',
    'databoard/model.py',  # db model, TODO should be renamed
    'databoard/base_prediction.py',
    'databoard/multiclass_prediction.py',
    'databoard/regression_prediction.py',
    'databoard/scores.py',
    'databoard/train_test.py',
    'databoard/views.py',
    'databoard/static',
    'databoard/templates',
    'databoard/db',
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
    from databoard.config import config
    ramp_name = config.ramp_name
    # TODO: check if ramp_name is the same as in
    #      'ramps/' + ramp_name + '/specific.py'

    local = config.train_server == 'localhost'
    command = "rsync -pthrRvz -c "
    if not local:
        command += "--rsh=\'ssh -i " + os.path.expanduser("~")
        command += "/.ssh/datacamp/id_rsa -p 22\' "
    for file in software:
        command += file + " "

    if not config.is_same_web_and_train_servers():
        command += config.get_deployment_target('web')
    else:
        command += config.get_deployment_target('train')
    print command
    os.system(command)

    # rsyncing specific
    command = "rsync -pthrvz -c "
    if not local:
        command += "--rsh=\'ssh -i " + os.path.expanduser("~")
        command += "/.ssh/datacamp/id_rsa -p 22\' "
    command += " ramps/" + ramp_name + "/specific.py "
    if not config.is_same_web_and_train_servers():
        command += config.get_deployment_target('web')
        command += "/databoard/"
    else:
        command += config.get_deployment_target('train')
        command += "/databoard/"
    print command
    os.system(command)

    if test:
        command = "rsync -pthrvz -c "
        if not local:
            command += "--rsh=\'ssh -i " + os.path.expanduser("~")
            command += "/.ssh/datacamp/id_rsa -p 22\' "
        command += " ramps/" + ramp_name + "/deposited_submissions "
        if not config.is_same_web_and_train_servers():
            command += config.get_deployment_target('web')
        else:
            command += config.get_deployment_target('train')
        print command
        os.system(command)


# (re)publish data set from 'ramps/' + ramp_name + '/data'
# fab prepare_data should be run at the destination
def publish_data(ramp_index):
    from databoard.config import config
    local = config.train_server == 'localhost'
    ramp_name = config.ramp_name
    command = "rsync --delete -pthrvz -c "
    if not local:
        command += "--rsh=\'ssh -i " + os.path.expanduser("~")
        command += "/.ssh/datacamp/id_rsa -p 22\' "
    command += 'ramps/' + ramp_name + '/data '
    if not config.is_same_web_and_train_servers():
        command += config.get_deployment_target('web') + "/"
    else:
        command += config.get_deployment_target('train') + "/"
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

    from databoard.config import config
    if not config.is_same_web_and_train_servers():
        # destination_path eg
        # '/tmp/databoard_local/databoard_mortality_prediction_8080'
        destination_path = config.web_destination_path
        destination_root = config.web_root
        _clear_destination_path_aux(destination_path, destination_root)
    destination_path = config.train_destination_path
    destination_root = config.train_root
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
    create_test_db()
    setup()
    add_models()
    train_test()
    leaderboard(test='True')
