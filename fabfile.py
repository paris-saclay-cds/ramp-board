import fabric.contrib.project as project
from fabric.api import *

# the user to use for the remote commands
env.user = 'root'

# the servers where the commands are executed
env.hosts = ['onevm-222.lal.in2p3.fr']
production = env.hosts[0]
dest_path = '/mnt/datacamp/databoard'

def setup(type='tabularasa'):
    local('python scripts/setup_databoard.py')

def fetch():
    local('python scripts/fetch_models.py')

def train():
    local('python scripts/train_models.py')

def serve():
    local('python server.py')

def leaderboard():
    local('python scripts/make_leaderboards.py')

def clean():
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
