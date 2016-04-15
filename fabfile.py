import sys
if '' not in sys.path:  # Balazs bug
    sys.path.insert(0, '')

import os
import logging
import time
from distutils.util import strtobool

logger = logging.getLogger('databoard')


def publish_local_test():
    destination_path = '/tmp/databoard_test'
    os.system('rm -rf ' + destination_path)
    os.mkdir(destination_path)
    os.system('rsync -rRultv problems/iris ' + destination_path)
    os.system('rsync -rRultv problems/boston_housing ' + destination_path)
    os.system('rsync -rultv fabfile.py ' + destination_path)


def test_setup():
    from databoard.remove_test_db import recreate_test_db
    import databoard.db_tools as db_tools
    import databoard.config as config

    if not os.path.exists(config.submissions_path):
        os.mkdir(config.submissions_path)
    if not os.path.exists(config.db_path):
        os.mkdir(config.db_path)
    open(os.path.join(config.submissions_path, '__init__.py'), 'a').close()

    recreate_test_db()
    db_tools.setup_workflows()
    db_tools.add_problem('iris')
    db_tools.add_event('iris_test')
    db_tools.add_problem('boston_housing')
    db_tools.add_event('boston_housing_test')

    db_tools.create_user(
        name='kegl', password='wine fulcra kook homy',
        lastname='Kegl', firstname='Balazs',
        email='balazs.kegl@gmail.com', access_level='admin')
    db_tools.create_user(
        name='agramfort', password='Fushun helium pigeon radon',
        lastname='Gramfort', firstname='Alexandre',
        email='alexandre.gramfort@gmail.com', access_level='admin')
    db_tools.create_user(
        name='akazakci', password='Sept. bestir Ottawa seven',
        lastname='Akin', firstname='Kazakci',
        email='osmanakin@gmail.com', access_level='admin')
    db_tools.create_user(
        name='mcherti', password='blown ashcan manful dost', lastname='Cherti',
        firstname='Mehdi', email='mehdicherti@gmail.com',
        access_level='admin')
    db_tools.create_user(
        name='camille_marini', password='therm pasha tootle stoney',
        lastname='Marini', firstname='Camille',
        email='camille.marini@gmail.com', access_level='admin')

    db_tools.create_user(
        name='test_user', password='test',
        lastname='Test', firstname='User',
        email='test.user@gmail.com', access_level='user')
    db_tools.create_user(
        name='test_iris_admin', password='test',
        lastname='Admin', firstname='Iris',
        email='iris.admin@gmail.com', access_level='user')

    db_tools.sign_up_team('iris_test', 'kegl')
    db_tools.sign_up_team('boston_housing_test', 'kegl')
    db_tools.sign_up_team('iris_test', 'agramfort')
    db_tools.sign_up_team('boston_housing_test', 'agramfort')
    db_tools.sign_up_team('iris_test', 'akazakci')
    db_tools.sign_up_team('boston_housing_test', 'akazakci')
    db_tools.sign_up_team('iris_test', 'mcherti')
    db_tools.sign_up_team('boston_housing_test', 'mcherti')
    db_tools.sign_up_team('iris_test', 'camille_marini')
    db_tools.sign_up_team('boston_housing_test', 'camille_marini')
    db_tools.sign_up_team('iris_test', 'test_user')

    db_tools.make_event_admin('iris_test', 'test_iris_admin')

    db_tools.make_submission_and_copy_files(
        'iris_test', 'kegl', 'rf',
        'problems/iris/deposited_submissions/kegl/rf')
    db_tools.make_submission_and_copy_files(
        'iris_test', 'kegl', 'rf2',
        'problems/iris/deposited_submissions/kegl/rf2')

    db_tools.make_submission_and_copy_files(
        'boston_housing_test', 'camille_marini', 'rf',
        'problems/boston_housing/deposited_submissions/kegl/rf')
    db_tools.make_submission_and_copy_files(
        'boston_housing_test', 'camille_marini', 'rf2',
        'problems/boston_housing/deposited_submissions/kegl/rf2')

    # send data to datarun
    host_url = os.environ.get('DATARUN_URL')
    username = os.environ.get('DATARUN_USERNAME')
    userpassd = os.environ.get('DATARUN_PASSWORD')
    data_id_iris = db_tools.send_data_datarun('iris', host_url, username,
                                              userpassd)
    data_id_boston = db_tools.send_data_datarun('boston_housing', host_url,
                                                username, userpassd)
    # send submissions to datarun
    print('**** TRAIN-TEST DATARUN ****')
    from databoard.db_tools import get_submissions
    list_data = [data_id_iris, data_id_boston]
    list_event = ['iris_test', 'boston_housing_test']
    for data_id, event_name in zip(list_data, list_event):
        submissions = get_submissions(event_name=event_name)
        submissions = [sub for sub in submissions if sub.name != 'sandbox']
        db_tools.train_test_submissions_datarun(data_id, host_url,
                                                username, userpassd,
                                                submissions=submissions,
                                                force_retrain_test=True,
                                                priority='L')
        time.sleep(228)
        db_tools.get_trained_tested_submissions_datarun(submissions, host_url,
                                                        username, userpassd)
        db_tools.compute_contributivity(event_name)

    # compare results with local train and test
    print('**** TRAIN-TEST LOCAL ****')
    db_tools.train_test_submissions()
    db_tools.compute_contributivity('iris_test')
    db_tools.compute_contributivity('boston_housing_test')


def sign_up_team(e, t):
    from databoard.db_tools import sign_up_team
    sign_up_team(event_name=e, team_name=t)


def approve_user(u):
    from databoard.db_tools import approve_user
    approve_user(user_name=u)


def serve():
    from databoard import app
    import databoard.views  # noqa
    import databoard.config as config

    server_port = int(config.server_port)
    app.run(
        debug=False,
        port=server_port,
        use_reloader=False,
        host='0.0.0.0',
        processes=1000)


def send_data_datarun(problem_name, host_url, username, userpassd):
    """
    Send data to datarun and prepare data (split train test)

    :param problem_name: name of the problem
    :param host_url: host url of datarun
    :param username: username for datarun
    :param userpassd: user password for datarun

    :type problem_name: string
    :type host_url: string
    :type username: string
    :type userpassd: string
    """
    from databoard.db_tools import send_data_datarun
    send_data_datarun(problem_name, host_url, username, userpassd)


def train_test_datarun(data_id, host_url, username, userpassd, e=None, t=None,
                       s=None, state=None, force='False', priority='L'):
    """Train and test submission using datarun.

    :param data_id: id of the associated dataset on datarun platform
    :param host_url: host url of datarun
    :param username: username for datarun
    :param userpassd: user password for datarun
    :param priority: training priority of the submissions on datarun,\
        'L' for low and 'H' for high

    :type data_id: integer
    :type host_url: string
    :type username: string
    :type userpassd: string
    :type priority: string
     """
    force = strtobool(force)

    from databoard.db_tools import train_test_submissions_datarun,\
        get_submissions, get_submissions_of_state

    if state is not None:
        submissions = get_submissions_of_state(state)
    else:
        submissions = get_submissions(
            event_name=e, team_name=t, submission_name=s)
    print submissions
    train_test_submissions_datarun(data_id, host_url, username, userpassd,
                                   submissions, force_retrain_test=force,
                                   priority=priority)
    compute_contributivity(event_name=e)


def get_trained_tested_datarun(host_url, username, userpassd,
                               e=None, t=None, s=None):
    """
    Get submissions from datarun and save predictions to databoard database

    :param host_url: host url of datarun
    :param username: username for datarun
    :param userpassd: user password for datarun

    :type host_url: string
    :type username: string
    :type userpassd: string
    """
    from databoard.db_tools import get_trained_tested_submissions_datarun
    from databoard.db_tools import get_submissions
    submissions = get_submissions(event_name=e, team_name=t, submission_name=s)
    print(submissions)
    get_trained_tested_submissions_datarun(submissions, host_url,
                                           username, userpassd)


def train_test(e=None, t=None, s=None, state=None, force='False'):
    force = strtobool(force)

    from databoard.db_tools import train_test_submissions,\
        get_submissions, get_submissions_of_state

    if state is not None:
        submissions = get_submissions_of_state(state)
    else:
        submissions = get_submissions(
            event_name=e, team_name=t, submission_name=s)
    print submissions
    train_test_submissions(submissions, force_retrain_test=force)
    compute_contributivity(event_name=e)


def compute_contributivity(event_name):
    from databoard.db_tools import compute_contributivity
    compute_contributivity(event_name)


def print_submissions(e=None, t=None, s=None):
    from databoard.db_tools import print_submissions
    print_submissions(event_name=e, team_name=t, submission_name=s)


software = [
    'fabfile.py',
    'setup.py',
    'requirements.txt',
    'README.md',
    'databoard/__init__.py',
    'databoard/base_prediction.py',
    'databoard/config.py',
    'databoard/db_tools.py',
    'databoard/fetch.py',
    'databoard/forms.py',
    'databoard/machine_parallelism.py',
    'databoard/model.py',
    'databoard/multiclass_prediction.py',
    'databoard/regression_prediction.py',
    'databoard/remove_test_db.py',
    'databoard/scores.py',
    'databoard/views.py',
    'databoard/post_api.py',
    'databoard/specific/__init__.py',
    'databoard/specific/workflows/__init__.py',
    'databoard/specific/workflows/regressor_workflow.py',
    'databoard/specific/workflows/classifier_workflow.py',
    'databoard/specific/problems/__init__.py',
    'databoard/specific/problems/boston_housing.py',
    'databoard/specific/problems/iris.py',
    'databoard/specific/events/__init__.py',
    'databoard/specific/events/boston_housing_test.py',
    'databoard/specific/events/iris_test.py',
    'databoard/tests/__init__.py',
    'databoard/tests/test_model.py',
    'databoard/tests/test_multiclass_predictions.py',
    'databoard/tests/test_regression_predictions.py',
    'databoard/tests/test_scores.py',
    'databoard/static',
    'databoard/templates',
]

deployment = [
    'fabfile.py',
    'problems/iris/data/raw/iris.csv',
    'problems/iris/sandbox/classifier.py',
    'problems/iris/deposited_submissions/kegl/rf/classifier.py',
    'problems/iris/deposited_submissions/kegl/rf2/classifier.py',
    'problems/boston_housing/data/raw/boston_housing.csv',
    'problems/boston_housing/sandbox/regressor.py',
    'problems/boston_housing/deposited_submissions/kegl/rf/regressor.py',
    'problems/boston_housing/deposited_submissions/kegl/rf2/regressor.py',
]


def publish_software():
    from databoard.config import vd_server, vd_root

    command = "rsync -pthrRvz -c "
    command += "--rsh=\'ssh -i " + os.path.expanduser("~")
    command += "/.ssh/datacamp/id_rsa -p 22\' "
    for file in software:
        command += file + ' '
    command += 'root@' + vd_server + ':' + vd_root + '/code/'
    print command
    os.system(command)


def publish_deployment():
    from databoard.config import vd_server, vd_root

    command = "rsync -pthrRvz -c "
    command += "--rsh=\'ssh -i " + os.path.expanduser("~")
    command += "/.ssh/datacamp/id_rsa -p 22\' "
    for file in deployment:
        command += file + ' '
    command += 'root@' + vd_server + ':' + vd_root + '/databoard_test/'
    print command
    os.system(command)

