import socket
from multiprocessing import cpu_count

root_path = "."
test_size = 0.5
random_state = 57

# temporary trick to detect if whether
# this is debug mode
debug_mode = 'onevm' not in socket.gethostname()

n_processes = 3 if debug_mode else cpu_count()
n_CV = 2 if debug_mode else 5 * n_processes

repos_path = "./TeamsRepos"
cachedir = '.'
serve_port = 8080

debug_server = 'http://' + "localhost:{}".format(serve_port) 
deploy_server = 'http://' + socket.gethostname() + ".lal.in2p3.fr:{}".format(serve_port)
server_name = debug_server if debug_mode else deploy_server
