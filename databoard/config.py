import os
from importlib import import_module

root_path = '.'

# paths
db_path = os.path.join(os.getcwd(), 'db')
db_f_name = os.path.join(db_path, 'test.db')
repos_path = os.path.join(root_path, 'git_submissions')
ground_truth_path = os.path.join(root_path, 'ground_truth')
submissions_d_name = 'submissions'
submissions_path = os.path.join(root_path, submissions_d_name)
deposited_submissions_path = os.path.join(root_path, 'deposited_submissions')
data_path = os.path.join(root_path, 'data')
raw_data_path = os.path.join(data_path, 'raw')
public_data_path = os.path.join(data_path, 'public')
private_data_path = os.path.join(data_path, 'private')

cachedir = '.'

is_parallelize = False  # make it False if parallel training is not working
# make it True to use parallelism across machines
is_parallelize_across_machines = False
# maximum number of seconds per model training for parallelize across machines
timeout_parallelize_across_machines = 10800
# often doesn't work and takes a lot of disk space
is_pickle_trained_submission = False

notification_recipients = []
notification_recipients.append("balazs.kegl@gmail.com")
notification_recipients.append("alexandre.gramfort@gmail.com")

# Open ports in Stratuslab
# 22, 80, 389, 443, 636, 2135, 2170, 2171, 2172, 2811, 3147, 5001, 5010, 5015,
# 8080, 8081, 8095, 8188, 8443, 8444, 9002, 10339, 10636, 15000, 15001, 15002,
# 15003, 15004, 20000-25000.

# amadeus
# server_port = '8443'
# dest_path = '/mnt/datacamp/databoard_06_8443_test'

# pollenating insects
# server_port = '8444'
# dest_path = '/mnt/datacamp/databoard_07_8444_test'

# el nino
# server_port = '8188'
# dest_path = '/mnt/datacamp/databoard_05_8188_test'

# kaggle otto with skf_test_size = 0.5
# server_port = '8081'
# dest_path = '/mnt/datacamp/databoard_04_8081_test'

# kaggle otto with skf_test_size = 0.2
# server_port = '8095'
# dest_path = '/mnt/datacamp/databoard_04_8095_test'

# variable star
# server_port = '8080'
# dest_path = '/mnt/datacamp/databoard_03_8080_test'

# debug_server = 'http://' + "localhost:{}".format(server_port)
# train_server = 'http://' + socket.gethostname() + ".lal.in2p3.fr:{}".format(
#    server_port)
# server_name = debug_server if local_deployment else train_server

vd_server = 'onevm-132.lal.in2p3.fr'
reims_server = 'romeo1.univ-reims.fr'
vd_root = '/mnt/datacamp'
local_root = '/tmp/databoard_local'  # for local publishing / testing


class RampConfig(object):
    def __init__(self,
                 ramp_name,
                 # for naming the library where the data and specific.py is
                 train_server,  # the server for training
                 train_user,  # the username on the train_server
                 train_root,  # the root dir of databoard on the train_server
                 num_cpus,  # number of cpus on the train_server
                 web_server,
                 # the server for the web site (and possibly leaderboard)
                 web_user,  # the username on the web_server
                 web_root,  # the root dir of databoard on the web_server
                 server_port,  # the server port on the web_server
                 cv_test_size,
                 random_state,
                 cv_bag_size=0,
                 ):
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
        self.cv_bag_size = cv_bag_size
        self.random_state = random_state

    def get_destination_path(self, root):
        destination_path = os.path.join(
            root, "databoard_" + self.ramp_name + "_" + self.server_port)
        return destination_path

    @property
    def train_destination_path(self):
        return self.get_destination_path(self.train_root)

    @property
    def web_destination_path(self):
        return self.get_destination_path(self.web_root)

    @property
    def specific(self):
        return import_module(
            '.specific', 'databoard.ramps.' + self.ramp_name)

    def get_deployment_target(self, mode='web'):
        deployment_target = ''
        if mode == 'web':
            user, server, root = 'web_user', 'web_server', 'web_root'
        elif mode == 'train':
            user = self.train_user
            server = self.train_server
            root = self.train_root
        else:
            raise ValueError('mode ???')
        if self.train_server != 'localhost':
            deployment_target += user + '@' + server + ':'
        deployment_target += self.get_destination_path(root)
        print deployment_target
        return deployment_target

    def is_same_web_and_train_servers(self):
        return ((self.web_server == self.train_server)
                and (self.web_user == self.train_user)
                and (self.web_root == self.train_root))


ramps_configs = dict()

local_kwargs = dict(
    train_server='localhost',
    train_user='',
    train_root=local_root,
    num_cpus=2,
    web_server='localhost',
    web_user='',
    web_root=local_root,
    server_port='8080',
    cv_test_size=0.2,
    random_state=57,
)

vd_kwargs = dict(
    train_server=vd_server,
    train_user='root',
    train_root=vd_root,
    web_server=vd_server,
    web_user='root',
    web_root=vd_root,
)

reims_kwargs = dict(
    train_server=reims_server,
    train_user='mcherti',
    train_root='/home/mcherti/ramp_pollenating_insects',
    web_server=vd_server,
    web_user='root',
    web_root=vd_root,
)

ramps_configs['iris_local'] = RampConfig(
    ramp_name='iris', **local_kwargs)
ramps_configs['boston_housing_local'] = RampConfig(
    ramp_name='boston_housing', **local_kwargs)
ramps_configs['amadeus_local'] = RampConfig(
    ramp_name='amadeus', **local_kwargs)
ramps_configs['el_nino_bagged_cv_future_local'] = RampConfig(
    ramp_name='el_nino_bagged_cv_future', **local_kwargs)
ramps_configs['el_nino_block_cv_local'] = RampConfig(
    ramp_name='el_nino_block_cv', cv_bag_size=0.5, **local_kwargs)
ramps_configs['kaggle_otto_local'] = RampConfig(
    ramp_name='kaggle_otto', **local_kwargs)
ramps_configs['mortality_prediction_local'] = RampConfig(
    ramp_name='mortality_prediction', **local_kwargs)
ramps_configs['pollenating_insects_local'] = RampConfig(
    ramp_name='pollenating_insects', **local_kwargs)
ramps_configs['variable_stars_local'] = RampConfig(
    ramp_name='variable_stars', **local_kwargs)

ramps_configs['iris_remote'] = RampConfig(
    ramp_name='iris',
    num_cpus=32,
    server_port='2171',
    cv_test_size=0.2,
    random_state=57,
    **vd_kwargs
)

ramps_configs['mortality_prediction_remote'] = RampConfig(
    ramp_name='mortality_prediction',
    num_cpus=10,
    server_port='2172',
    cv_test_size=0.2,
    random_state=57,
    **vd_kwargs
)

with open("ramp_index.txt") as f:
    ramp_index = f.readline()

config_object = ramps_configs[ramp_index]
