import os
import logging
import pandas as pd
import fabric.contrib.project as project
from fabric.api import *
from fabric.contrib.files import exists
from databoard.model import shelve_database, ModelState
from databoard.config_databoard import (
    root_path, 
    cachedir,
    repos_path,
    ground_truth_path,
    # output_path,
    models_path,
    local_deployment,
)

# Open ports in Stratuslab
# 22, 80, 389, 443, 636, 2135, 2170, 2171, 2172, 2811, 3147, 5001, 5010, 5015, 
# 8080, 8081, 8095, 8188, 8443, 8444, 9002, 10339, 10636, 15000, 15001, 15002, 
# 15003, 15004, 20000-25000.

env.user = 'root'  # the user to use for the remote commands
env.use_ssh_config = True

# the servers where the commands are executed
env.hosts = ['onevm-54.lal.in2p3.fr']
#env.hosts = ['onevm-188.lal.in2p3.fr']
production = env.hosts[0]
#dest_path = '/mnt/datacamp/databoard_03_9002_test'
#server_port = '9002'
dest_path = '/mnt/datacamp/databoard_03_8080_test'
#server_port = '80'
#dest_path = '/mnt/datacamp/databoard_03_80_deployment'
server_port = '8080'

logger = logging.getLogger('databoard')


def all():
    fetch()
    train()
    leaderboard()

def clear_cache():
    from sklearn.externals.joblib import Memory
    mem = Memory(cachedir=cachedir)
    mem.clear()

def clear_db():
    from databoard.model import columns
    
    with shelve_database('c') as db:
        db.clear()
        db['models'] = pd.DataFrame(columns=columns)
        db['leaderboard1'] = pd.DataFrame(columns=['score'])
        db['leaderboard2'] = pd.DataFrame(columns=['contributivity'])

def clear_registrants():
    import shutil
    # Prepare the teams repo submodules
    # logger.info('Init team repos git')
    # repo = Repo.init(repos_path)  # does nothing if already exists
    shutil.rmtree(repos_path, ignore_errors=True)
    os.mkdir(repos_path)

def clear_pred_files():
    import glob
    fnames = []

    # TODO: some of the following will be removed after switching to a database
    fnames += glob.glob(os.path.join(models_path, '*', '*', 'pred_*'))
    fnames += glob.glob(os.path.join(models_path, '*', '*', 'score.csv'))
    fnames += glob.glob(os.path.join(models_path, '*', '*', 'error.txt'))

    for fname in fnames:
        if os.path.exists(fname):
            os.remove(fname)

def clear_groundtruth():
    import shutil    
    shutil.rmtree(ground_truth_path, ignore_errors=True)
    os.mkdir(ground_truth_path)

def init_config():
    pass
    # TODO

def print_db(table='models', state=None):
    with shelve_database('c') as db:
        if table not in db:
            print('Select one of the following tables:')
            print '\n'.join('\t- {}'.format(t) for t in db)
            return
        df = db[table]
    if not state:
        print df
    else:
        print df[df.state == state]


def setup(wipeall=False):
    from databoard.generic import setup_ground_truth
    from databoard.specific import prepare_data
    
    # Preparing the data set, typically public train/private held-out test cut
    logger.info('Prepare the dataset.')
    prepare_data()

    logger.info('Remove the ground truth files.')
    clear_groundtruth()

    # Set up the ground truth predictions for the CV folds
    logger.info('Setup the groundtruth.')
    setup_ground_truth()
    
    logger.info('Clear the database.')
    clear_db()

    if not os.path.exists(models_path):
        os.mkdir(models_path)
    open(os.path.join(models_path, '__init__.py'), 'a').close()

    logger.info('Config init.')
    init_config()

    if wipeall:
        # Remove the git repos of the teams
        logger.info('Clear the teams repositories.')
        clear_registrants()

        # Flush joblib cache
        logger.info('Flush the joblib cache.')
        clear_cache()

def clean_pyc():
    local('find . -name "*.pyc" | xargs rm -f')


def fetch():
    from databoard.fetch import fetch_models
    fetch_models()

def repeat_fetch(delay='60'):
    import time
    while True:
        fetch()
        delay = int(os.getenv('FETCH_DELAY', delay))
        time.sleep(delay)

def leaderboard(which='all'):
    from databoard.generic import (
        leaderboard_classical, 
        leaderboard_combination, 
    )

    groundtruth_path = os.path.join(root_path, 'ground_truth')

    # submissions_path = os.path.join(root_path, 'output/trained_submissions.csv')
    with shelve_database() as db:
        submissions = db['models']
        trained_models = submissions[submissions.state == "trained"]
        # trained_models = pd.read_csv(submissions_path)

    if which in ('all', '1'):
        l1 = leaderboard_classical(groundtruth_path, trained_models)
        # The following assignments only work because leaderboard_classical & co
        # are idempotent.
        # FIXME (potentially)
        with shelve_database() as db:
            db['leaderboard1'] = l1

    if which in ('all', '2'):
        l2 = leaderboard_combination(groundtruth_path, trained_models)
        # FIXME: same as above
        with shelve_database() as db:
            db['leaderboard2'] = l2

    # l3 = private_leaderboard_classical(trained_models)


def train(state=False, tag=None):
    from databoard.generic import train_and_valid_models

    with shelve_database() as db:
        models = db['models']

    if tag is not None:
        models = models[models.model.str.contains(tag)]
        state = 'all'  # force train all the selected models
        if len(models) == 0:
            print('No existing model containing the tag: {}'.format(tag))
            return

    if not state:
        state = 'new'
    
    if state != 'all': 
        models = models[models.state == state]

    train_and_valid_models(models)

    idx = models.index

    with shelve_database() as db:
        db['models'].loc[idx, :] = models

def test(state=False, tag=None):
    from databoard.generic import test_models

    with shelve_database() as db:
        models = db['models']

    if tag is not None:
        models = models[models.model.str.contains(tag)]
        state = 'all'  # force test all the selected models
        if len(models) == 0:
            print('No existing model containing the tag: {}'.format(tag))
            return

    if not state:
        state = 'trained'
    
    if state != 'all': 
        models = models[models.state == state]

    test_models(models)

    idx = models.index

    with shelve_database() as db:
        db['models'].loc[idx, :] = models

import numpy as np

def set_state(team, tag, state):
    with shelve_database() as db:
        models = db['models']
    models = models[np.logical_and(models['model'] == tag, 
        models['team'] == team)]

    if len(models) > 1:
        print "ambiguous selection"
        print selected_models
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

    answer = 'y'
    while answer != 'y':
        answer = raw_input('Sure? (y/n): ')

    pid_filenames = os.path.join(models_path, team, get_tag_uid(team, tag), 'pid_*')
    print pid_filenames
    for f in glob.glob(pid_filenames):
        with open(f) as pid_file:
            pid = pid_file.read()
            os.kill(int(pid), signal.SIGKILL)    
            

def serve(port=8080):
    from databoard import app
    import databoard.views

    debug_mode = os.getenv('DEBUGLB', local_deployment)
    try: 
        debug_mode = bool(int(debug_mode))
    except ValueError:
        debug_mode = True  # a non empty string means debug
    app.run(
        debug=bool(debug_mode), 
        port=int(os.getenv('SERV_PORT', port)), 
        host='0.0.0.0')


# TODO: fill up the following functions so to easily deploy
# databoard on the server

# FIXME: dtach not working
@hosts(production)
def rserve(sockname="db_server"):
    if not exists("/usr/bin/dtach"):
        sudo("apt-get install dtach")

    with cd(dest_path):
        # run('export SERV_PORT={}'.format(server_port))
        # run('fab serve')
        # run('dtach -n `mktemp -u /tmp/{}.XXXX` export SERV_PORT={};fab serve'.format(sockname, server_port))
        return run('dtach -n `mktemp -u /tmp/{}.XXXX` fab serve:port={}'.format(sockname, server_port))

@hosts(production)
def publish():
    local('')
    project.rsync_project(
        remote_dir=dest_path,
        exclude=[ '.DS_Store', 
                  'TeamsRepos', 
                  'teams_repos', 
                  'models', 
                  'output',
                  'joblib',
                  'shelve*',
                  '*.ipynb*',
                  '*.log',
                  '.git*',
                  '.gitignore',
                  '*.bak',
                  '*.pyc'],
        local_dir='.',
        delete=False,
        extra_opts='-c',
    )
