import os
import socket
from git import Repo
from multiprocessing import cpu_count

# root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_path = '.'

tag_len_limit = 40

# temporary trick to detect if whether
# this is debug mode
local_deployment = 'onevm' not in socket.gethostname()
n_processes = 3 if local_deployment else cpu_count()

# paths
repos_path = os.path.join(root_path, 'teams_repos')
ground_truth_path = os.path.join(root_path, 'ground_truth')
data_path = os.path.join(root_path, 'data')
raw_data_path = os.path.join(data_path, 'raw')
public_data_path = os.path.join(data_path, 'public')
private_data_path = os.path.join(data_path, 'private')
# output_path = os.path.join(root_path, 'output')
models_path = os.path.join(root_path, 'models')

cachedir = '.'

# Open ports in Stratuslab
# 22, 80, 389, 443, 636, 2135, 2170, 2171, 2172, 2811, 3147, 5001, 5010, 5015, 
# 8080, 8081, 8095, 8188, 8443, 8444, 9002, 10339, 10636, 15000, 15001, 15002, 
# 15003, 15004, 20000-25000.

# el nino
server_port = '8188'
dest_path = '/mnt/datacamp/databoard_05_8188_test'

# kaggle otto with skf_test_size = 0.5
#server_port = '8081'
#dest_path = '/mnt/datacamp/databoard_04_8081_test'

# kaggle otto with skf_test_size = 0.2
#server_port = '8095'
#dest_path = '/mnt/datacamp/databoard_04_8095_test'

# variable star
#server_port = '8080'
#dest_path = '/mnt/datacamp/databoard_03_8080_test'

debug_server = 'http://' + "localhost:{}".format(server_port) 
deploy_server = 'http://' + socket.gethostname() + ".lal.in2p3.fr:{}".format(server_port)
server_name = debug_server if local_deployment else deploy_server


notification_recipients = []
notification_recipients.append("djalel.benbouzid@gmail.com")
notification_recipients.append("balazs.kegl@gmail.com")
notification_recipients.append("alexandre.gramfort@gmail.com")

if local_deployment:
    try:
        user_mail = Repo('.').config_reader().get_value('user', 'email')
        notification_recipients = [user_mail]
    except:
        pass

assert repos_path != 'models' 