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
env.hosts = ['onevm-222.lal.in2p3.fr']
production = env.hosts[0]
dest_path = '/mnt/datacamp/databoard_02_8080_kegl'
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
        db['leaderboard1'] = pd.DataFrame(columns=['scores'])
        db['leaderboard2'] = pd.DataFrame(columns=['scores'])

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

def leaderboard():
    from databoard.generic import (
        leaderboard_classical, 
        leaderboard_combination, 
        private_leaderboard_classical,
    )

    groundtruth_path = os.path.join(root_path, 'ground_truth')

    # submissions_path = os.path.join(root_path, 'output/trained_submissions.csv')
    with shelve_database() as db:
        submissions = db['models']
        trained_models = submissions[submissions.state == "trained"]
        # trained_models = pd.read_csv(submissions_path)

    l1 = leaderboard_classical(groundtruth_path, trained_models)
    l2 = leaderboard_combination(groundtruth_path, trained_models)
    # l3 = private_leaderboard_classical(trained_models)

    # The following assignments only work because leaderboard_classical & co
    # are idempotent.
    # FIXME (potentially)
    with shelve_database() as db:
        db['leaderboard1'] = l1
        db['leaderboard2'] = l2

    # l1.to_csv("output/leaderboard1.csv", index=False)
    # l2.to_csv("output/leaderboard2.csv", index=False)
    # l3.to_csv("output/leaderboard3.csv", index=False)


def train():
    from databoard.generic import train_models

    with shelve_database() as db:
        models = db['models']

    # models = pd.read_csv("output/submissions.csv")
    # trained_models, failed_models = train_models(models)
    train_models(models)

    with shelve_database() as db:
        db['models'] = models
        logger.debug(models[models['state'] == "trained"])
        logger.debug(models[models['state'] == "error"])

    # trained_models.to_csv("output/trained_submissions.csv", index=False)
    # failed_models.to_csv("output/failed_submissions.csv", index=False)


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
                  '*.pyc'],
        local_dir='.',
        delete=False,
        extra_opts='-c',
    )
