import os
import pandas as pd
from git import Repo

# root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = '.'

tag_len_limit = 40

# paths
repos_path = os.path.join(root_path, 'teams_repos')
ground_truth_path = os.path.join(root_path, 'ground_truth')
models_path = os.path.join(root_path, 'models')
submissions_path = os.path.join(root_path, 'teams_submissions')
data_path = os.path.join(root_path, 'data')
raw_data_path = os.path.join(data_path, 'raw')
public_data_path = os.path.join(data_path, 'public')
private_data_path = os.path.join(data_path, 'private')

cachedir = '.'

is_parallelize = True  # make it False if parallel training is not working
# make it True to use parallelism across machines
is_parallelize_across_machines = False
# maximum number of seconds per model training for parallelize across machines
timeout_parallelize_across_machines = 10800
# often doesn't work and takes a lot of disk space
is_pickle_trained_model = False

# Open ports in Stratuslab
# 22, 80, 389, 443, 636, 2135, 2170, 2171, 2172, 2811, 3147, 5001, 5010, 5015,
# 8080, 8081, 8095, 8188, 8443, 8444, 9002, 10339, 10636, 15000, 15001, 15002,
# 15003, 15004, 20000-25000.

# amadeus
#server_port = '8443'
#dest_path = '/mnt/datacamp/databoard_06_8443_test'

# pollenating insects
#server_port = '8444'
#dest_path = '/mnt/datacamp/databoard_07_8444_test'

# el nino
#server_port = '8188'
#dest_path = '/mnt/datacamp/databoard_05_8188_test'

# kaggle otto with skf_test_size = 0.5
#server_port = '8081'
#dest_path = '/mnt/datacamp/databoard_04_8081_test'

# kaggle otto with skf_test_size = 0.2
#server_port = '8095'
#dest_path = '/mnt/datacamp/databoard_04_8095_test'

# variable star
#server_port = '8080'
#dest_path = '/mnt/datacamp/databoard_03_8080_test'

#debug_server = 'http://' + "localhost:{}".format(server_port)
#train_server = 'http://' + socket.gethostname() + ".lal.in2p3.fr:{}".format(server_port)
#server_name = debug_server if local_deployment else train_server

vd_server = 'onevm-85.lal.in2p3.fr'
reims_server = 'romeo1.univ-reims.fr'
vd_root = '/mnt/datacamp'
local_root = '/tmp/databoard_local'  # for local publishing / testing

class RampConfig(object):
    def __init__(ramp_name, # for naming the library where the data and specific.py is
                 train_server,  # the server for training
                 train_user,  # the username on the train_server
                 train_root,  # the root dir of databoard on the train_server
                 num_cpus,  # number of cpus on the train_server
                 web_server,  # the server for the web site (and possibly leaderboard)
                 web_user,  # the username on the web_server
                 web_root,  # the root dir of databoard on the web_server
                 server_port,  # the server port on the web_server
                 cv_test_size,
                 random_state):
        self.ramp_name = ramp_name
        self.train_server = train_server
        self.train_user = train_user
        self.train_root = train_root
        self.num_cpus = num_cpus
        self.web_server = web_server
        self.web_user = web_user
        self.web_root = web_root
        self.server_port = server_port
        self.cv_test_size = cv_test_size
        self.random_state = random_state

    def get_destination_path(self):
        # XXX
        destination_root = get_ramp_field(root, ramp_index)
        ramp_name = get_ramp_field('ramp_name', ramp_index)
        server_port = get_ramp_field('server_port', ramp_index)
        destination_path = os.path.join(
            destination_root, "databoard_" + ramp_name + "_" + server_port)
        return destination_path

    @property
    def train_destination_path(self):
        return # XXX get_destination_path('train_root', ramp_index)

    @property
    def web_destination_path():
        return # XXX get_destination_path('web_root', ramp_index)


    def is_same_web_and_train_servers(ramp_index):
        # XXX don't know what to do with this...
        return (get_ramp_field('web_server', ramp_index) ==
                get_ramp_field('train_server', ramp_index)
                and get_ramp_field('web_user', ramp_index) ==
                get_ramp_field('train_user', ramp_index)
                and get_ramp_field('web_root', ramp_index) ==
                get_ramp_field('train_root', ramp_index))


ramps = dict()

ramps['iris_test'] = RampConfig(ramp_name='iris',
                               train_server='localhost',
                               train_user='',
                               train_root=local_root,
                               num_cpus=2,
                               web_server='localhost',
                               web_user='',
                               web_root=local_root,
                               server_port='8080',
                               cv_test_size=0.2,
                               random_state=57)

