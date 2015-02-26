import os
import logging
import pandas as pd
import fabric.contrib.project as project
from fabric.api import *
from databoard.config_databoard import (
    root_path, 
    cachedir,
    repos_path,
    ground_truth_path,
    output_path,
    models_path,
    local_deployment,
)


env.user = 'root'  # the user to use for the remote commands

# the servers where the commands are executed
env.hosts = ['onevm-222.lal.in2p3.fr']
production = env.hosts[0]
dest_path = '/mnt/datacamp/databoard'

logger = logging.getLogger('databoard')


def setup():
    from git import Repo
    from databoard.generic import setup_ground_truth
    from databoard.specific import prepare_data
    from sklearn.externals.joblib import Memory
    
    clean()

    open(os.path.join(models_path, '__init__.py'), 'a').close()
    # Prepare the teams repo submodules
    repo = Repo.init(repos_path)  # does nothing if already exists
    # Preparing the data set, typically public train/private held-out test cut
    prepare_data()
    # Set up the ground truth predictions for the CV folds
    setup_ground_truth()
    # Flush joblib cache
    mem = Memory(cachedir=cachedir)
    mem.clear()

    init_config()


def init_config():
    pass
    # TOOD:


def clean():
    import glob
    import shutil
    
    shutil.rmtree(ground_truth_path, ignore_errors=True)
    os.mkdir(ground_truth_path)

    shutil.rmtree(output_path, ignore_errors=True)
    os.mkdir(output_path)

    if not os.path.exists(models_path):
        os.mkdir(models_path)


    fnames = []
    # if os.path.exists(ground_truth_path):
    #     fnames = glob.glob(os.path.join(ground_truth_path, 'pred_*'))
    # else:
    #     os.mkdir(ground_truth_path)
    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)
    # if not os.path.exists(models_path):
    #     os.mkdir(models_path)

    # TODO: some of the following will be removed after switching to a database
    fnames += glob.glob(os.path.join(models_path, '*', '*', 'pred_*'))
    fnames += glob.glob(os.path.join(models_path, '*', '*', 'score.csv'))
    fnames += glob.glob(os.path.join(models_path, '*', '*', 'error.txt'))
    for fname in fnames:
        if os.path.exists(fname):
            os.remove(fname)

    # old_fnames = glob.glob(os.path.join(output_path, '*.csv'))
    # for fname in old_fnames:
    #     if os.path.exists(fname):
    #         os.remove(fname)


def fetch():
    from databoard.fetch import fetch_models
    fetch_models()


def leaderboard():
    from databoard.generic import (
        leaderboard_classical, 
        leaderboard_combination, 
        private_leaderboard_classical,
    )

    groundtruth_path = os.path.join(root_path, 'ground_truth')

    submissions_path = os.path.join(root_path, 'output/trained_submissions.csv')
    trained_models = pd.read_csv(submissions_path)

    l1 = leaderboard_classical(groundtruth_path, trained_models)
    l2 = leaderboard_combination(groundtruth_path, trained_models)
    l3 = private_leaderboard_classical(trained_models)

    l1.to_csv("output/leaderboard1.csv", index=False)
    l2.to_csv("output/leaderboard2.csv", index=False)
    l3.to_csv("output/leaderboard3.csv", index=False)


def train():
    from databoard.generic import train_models

    models = pd.read_csv("output/submissions.csv")
    trained_models, failed_models = train_models(models)
    logger.debug(trained_models)
    logger.debug(failed_models)
    trained_models.to_csv("output/trained_submissions.csv", index=False)
    failed_models.to_csv("output/failed_submissions.csv", index=False)


def serve():
    from databoard import app
    import databoard.views

    debug_mode = os.getenv('DEBUGLB', local_deployment)
    try: 
        debug_mode = bool(int(debug_mode))
    except ValueError:
        debug_mode = True  # a non empty string means debug
    app.run(
        debug=bool(debug_mode), 
        port=os.getenv('SERV_PORT', 8080), 
        host='0.0.0.0')



def rserve():
    with cd(dest_path):
        run('python server.py')



def clean_pyc():
    local('find . -name "*.pyc" | xargs rm -f')


def remote_pull():
    with cd(dest_path):
        run('git pull')
        

@hosts(production)
def publish():
    local('')
    project.rsync_project(
        remote_dir=dest_path,
        exclude=".DS_Store",
        local_dir='.',
        delete=False,
        extra_opts='-c',
    )
