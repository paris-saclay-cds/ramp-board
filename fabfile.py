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
    db_tools.setup_score_types()
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
    if host_url is None or username is None or userpassd is None:
        sys.exit('**** Configure your datarun authentication parameters ****')
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


def serve(port=None):
    from databoard import app
    import databoard.views  # noqa
    import databoard.config as config

    if port is None:
        port = config.server_port
    server_port = int(port)
    app.run(
        debug=False,
        port=server_port,
        use_reloader=False,
        host='0.0.0.0',
        processes=1000)


def add_workflow_element_type(name, type):
    from databoard.db_tools import add_workflow_element_type

    add_workflow_element_type(name, type)


def add_workflow(name, *element_type_names):
    from databoard.db_tools import add_workflow

    add_workflow(name, element_type_names)


def add_score_type(name, is_lower_the_better, minimum, maximum):
    from databoard.db_tools import add_score_type

    add_score_type(
        name, is_lower_the_better, float(minimum), float(maximum))


def add_problem(name, force='False'):
    """Add new problem. If force=True, deletes problem (with all events) if exists."""
    force = strtobool(force)

    from databoard.db_tools import add_problem

    add_problem(name, force)


def add_event(name, force='False'):
    """Add new event. If force=True, deletes event (with all submissions) if exists."""
    force = strtobool(force)

    from databoard.db_tools import add_event

    add_event(name, force)


def make_event_admin(e, u):
    from databoard.db_tools import make_event_admin

    make_event_admin(event_name=e, admin_name=u)


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
    from databoard.config import sandbox_d_name

    force = strtobool(force)

    from databoard.db_tools import train_test_submissions_datarun,\
        get_submissions, get_submissions_of_state

    if state is not None:
        submissions = get_submissions_of_state(state)
    else:
        submissions = get_submissions(
            event_name=e, team_name=t, submission_name=s)
    submissions = [submission for submission in submissions
                   if submission.name != sandbox_d_name]

    print submissions
    train_test_submissions_datarun(data_id, host_url, username, userpassd,
                                   submissions, force_retrain_test=force,
                                   priority=priority)


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
    compute_contributivity(event_name=e)


def train_test(e, t=None, s=None, state=None, force='False'):
    force = strtobool(force)

    from databoard.db_tools import train_test_submissions,\
        get_submissions, get_submissions_of_state
    from databoard.config import sandbox_d_name

    if state is not None:
        submissions = get_submissions_of_state(state)
    else:
        submissions = get_submissions(
            event_name=e, team_name=t, submission_name=s)
    submissions = [submission for submission in submissions
                   if submission.name != sandbox_d_name]
    train_test_submissions(submissions, force_retrain_test=force)
    compute_contributivity(event_name=e)


def compute_contributivity(event_name):
    from databoard.db_tools import compute_contributivity
    from databoard.db_tools import compute_historical_contributivity
    compute_contributivity(event_name)
    compute_historical_contributivity(event_name)


def compute_contributivity_and_save_leaderboards(event_name):
    from databoard.db_tools import compute_contributivity_and_save_leaderboards
    compute_contributivity_and_save_leaderboards(event_name)


def print_submissions(e=None, t=None, s=None):
    from databoard.db_tools import print_submissions
    print_submissions(event_name=e, team_name=t, submission_name=s)


def print_submission_similaritys():
    from databoard.db_tools import print_submission_similaritys
    print_submission_similaritys()


def delete_submission(e, t, s):
    from databoard.db_tools import delete_submission
    delete_submission(event_name=e, team_name=t, submission_name=s)


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
    'databoard/mixed_prediction.py',
    'databoard/multiclass_prediction.py',
    'databoard/regression_prediction.py',
    'databoard/remove_test_db.py',
    'databoard/views.py',
    'databoard/post_api.py',
    'databoard/specific/__init__.py',
    'databoard/specific/events/__init__.py',
    'databoard/specific/events/air_passengers_dssp4.py',
    'databoard/specific/events/boston_housing_test.py',
    'databoard/specific/events/drug_spectra.py',
    'databoard/specific/events/epidemium2_cancer_mortality.py',
    'databoard/specific/events/iris_test.py',
    'databoard/specific/events/HEP_detector_anomalies.py',
    'databoard/specific/events/variable_stars.py',
    'databoard/specific/problems/__init__.py',
    'databoard/specific/problems/air_passengers.py',
    'databoard/specific/problems/boston_housing.py',
    'databoard/specific/problems/drug_spectra.py',
    'databoard/specific/problems/epidemium2_cancer_mortality.py',
    'databoard/specific/problems/iris.py',
    'databoard/specific/problems/HEP_detector_anomalies.py',
    'databoard/specific/problems/variable_stars.py',
    'databoard/specific/score_types/__init__.py',
    'databoard/specific/score_types/accuracy.py',
    'databoard/specific/score_types/auc.py',
    'databoard/specific/score_types/error.py',
    'databoard/specific/score_types/error_mare_mixed.py',
    'databoard/specific/score_types/error_mixed.py',
    'databoard/specific/score_types/mare.py',
    'databoard/specific/score_types/mare_mixed.py',
    'databoard/specific/score_types/negative_log_likelihood.py',
    'databoard/specific/score_types/relative_rmse.py',
    'databoard/specific/score_types/rmse.py',
    'databoard/specific/workflows/__init__.py',
    'databoard/specific/workflows/classifier_workflow.py',
    'databoard/specific/workflows/feature_extractor_classifier_calibrator_workflow.py',
    'databoard/specific/workflows/feature_extractor_regressor_workflow.py',
    'databoard/specific/workflows/feature_extractor_classifier_regressor_workflow.py',
    'databoard/specific/workflows/feature_extractor_regressor_with_external_data_workflow.py',
    'databoard/specific/workflows/regressor_workflow.py',
    'databoard/tests/__init__.py',
    'databoard/tests/test_model.py',
    'databoard/tests/test_multiclass_predictions.py',
    'databoard/tests/test_regression_predictions.py',
    'databoard/static',
    'databoard/templates',
]


def publish_software(target='test'):
    from databoard.config import test_server, production_server
    from databoard.config import test_root, production_root

    if target == 'test':
        server = test_server
    else:
        server = production_server

    command = "rsync -pthrRvz -c "
    command += "--rsh=\'ssh -i " + os.path.expanduser("~")
    command += "/.ssh/datacamp/id_rsa -p 22\' "
    for file in software:
        command += file + ' '
    if target == 'test':
        command += 'root@' + server + ':/home/code/'
    else:
        command += 'root@' + server + ':/home/code/'
    print command
    os.system(command)


deployment = [
    'fabfile.py',
    'problems/iris/data/raw/iris.csv',
    'problems/iris/deposited_submissions/kegl/rf/classifier.py',
    'problems/iris/deposited_submissions/kegl/rf2/classifier.py',
    'problems/iris/description.html',
    'problems/iris/starting_kit',
    'problems/iris/starting_kit.zip',
    'problems/boston_housing/data/raw/boston_housing.csv',
    'problems/boston_housing/deposited_submissions/kegl/rf/regressor.py',
    'problems/boston_housing/deposited_submissions/kegl/rf2/regressor.py',
    'problems/boston_housing/description.html',
    'problems/boston_housing/starting_kit',
    'problems/boston_housing/starting_kit.zip',
]

# zip -r starting_kit.zip starting_kit


# def publish_deployment(target='test'):
#     from databoard.config import vd_server, production_server, vd_root

#     command = "rsync -pthrRvz -c "
#     if target == 'test':
#         server = vd_server
#         root = vd_root + '/databoard_test/'
#     else:
#         server = production_server
#         root = vd_root + '/databoard/'
#     command += "--rsh=\'ssh -i " + os.path.expanduser("~")
#     command += "/.ssh/datacamp/id_rsa -p 22\' "
#     for file in deployment:
#         command += file + ' '
#     command += 'root@' + server + ':' + root
#     print command
#     os.system(command)


def publish_problem(problem_name, target='local'):
    from databoard.config import test_server, production_server
    from databoard.config import test_root, production_root

    command = "rsync -pthrRvz -c "
    if target == 'local':
        command += 'problems/' + problem_name + ' /tmp/databoard_test/'
    else:
        if target == 'test':
            server = test_server
            root = test_root + '/databoard_test/'
        else:
            server = production_server
            root = production_root + '/databoard/'
        command += "--rsh=\'ssh -i " + os.path.expanduser("~")
        command += "/.ssh/datacamp/id_rsa -p 22\' problems/" + problem_name
        command += ' root@' + server + ':' + root
    print command
    os.system(command)
