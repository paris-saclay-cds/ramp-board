import os
import socket
from git import Repo
from multiprocessing import cpu_count

root_path = "."
tag_len_limit = 40

# temporary trick to detect if whether
# this is debug mode
local_deployment = 'onevm' not in socket.gethostname()
n_processes = 3 if local_deployment else cpu_count()

# paths
repos_path = os.path.join(root_path, 'TeamsRepos')
ground_truth_path = os.path.join(root_path, 'ground_truth')
data_path = os.path.join(root_path, 'data')
raw_data_path = os.path.join(data_path, 'raw')
public_data_path = os.path.join(data_path, 'public')
private_data_path = os.path.join(data_path, 'private')
output_path = os.path.join(root_path, 'output')
models_path = os.path.join(root_path, 'models')

cachedir = '.'
serve_port = 8080

debug_server = 'http://' + "localhost:{}".format(serve_port) 
deploy_server = 'http://' + socket.gethostname() + ".lal.in2p3.fr:{}".format(serve_port)
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