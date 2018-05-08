from __future__ import print_function, absolute_import, unicode_literals
import os
import time
import logging
from subprocess import call
from subprocess import check_output
from glob import glob
import re
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine.url import URL

import boto3  # amazon api

from rampbkd.model import Base
from rampbkd.query import select_submissions_by_id
from rampbkd.api import set_predictions
from rampbkd.api import set_submission_state
from rampbkd.api import get_submissions
from rampbkd.api import get_submission_state
from rampbkd.api import get_submission_by_id
from rampbkd.api import set_submission_max_ram
from rampbkd.api import score_submission
from rampbkd.api import set_submission_error_msg

__all__ = [
    'train_loop',
    'launch_ec2_instance_and_train',
    'train_on_existing_ec2_instance',
    'launch_ec2_instances',
    'terminate_ec2_instance',
    'list_ec2_instance_ids',
    'status_of_ec2_instance',
    'upload_submission',
    'download_log',
    'download_predictions',
    'launch_train',
    'abort_training',
]

# we disable all the other loggers partly because loggers
# from boto3 are too verbose
for name, l in logging.Logger.manager.loggerDict.items():
    l.disabled = True
logging.basicConfig(
    #format='%(asctime)s ## %(message)s',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%m/%d/%Y,%I:%M:%S')
logger = logging.getLogger('ramp_aws')


# configuration fields
AWS_CONFIG_SECTION = 'aws'
AMI_IMAGE_ID_FIELD = 'ami_image_id'
AMI_USER_NAME_FIELD = 'ami_user_name'
INSTANCE_TYPE_FIELD = 'instance_type'
KEY_PATH_FIELD = 'key_path'
KEY_NAME_FIELD = 'key_name'
SECURITY_GROUP_FIELD = 'security_group'
REMOTE_RAMP_KIT_FOLDER_FIELD = 'remote_ramp_kit_folder'
LOCAL_RAMP_KIT_FOLDER_FIELD = 'local_ramp_kit_folder'
LOCAL_PREDICTIONS_FOLDER_FIELD = 'local_predictions_folder'
CHECK_STATUS_INTERVAL_SECS_FIELD = 'check_status_interval_secs'
CHECK_FINISHED_TRAINING_INTERVAL_SECS_FIELD = (
    'check_finished_training_interval_secs')
LOCAL_LOG_FOLDER_FIELD = 'local_log_folder'
TRAIN_LOOP_INTERVAL_SECS_FIELD = 'train_loop_interval_secs'
MEMORY_PROFILING_FIELD = 'memory_profiling'

# constants
RAMP_AWS_BACKEND_TAG = 'ramp_aws_backend_instance'
SUBMISSIONS_FOLDER = 'submissions'


def train_loop(config, event_name):
    """
    This function starts a training loop for a given event
    The loop waits for any submission with the state 'new' then
    create an ec2 instance to train the submission on it.

    Parameters
    ----------

    event_name : str
        event name
    """
    conf = config[AWS_CONFIG_SECTION]
    secs = conf[TRAIN_LOOP_INTERVAL_SECS_FIELD]
    while True:
        # Launch new instances for new submissions
        submissions = get_submissions(config, event_name, 'new')
        for submission_id, _ in submissions:
            submission = get_submission_by_id(config, submission_id)
            if submission.is_sandbox:
                continue
            instance, = launch_ec2_instances(config, nb=1)
            _tag_instance_by_submission(instance.id, submission)
            logger.info('Launched instance "{}" for submission "{}"'.format(
                instance.id, submission))
            set_submission_state(config, submission.id, 'sent_to_training')
        # Score tested submissions
        submissions = get_submissions(config, event_name, 'tested')
        for submission_id, _ in submissions:
            label = _get_submission_label_by_id(config, submission_id)
            logger.info('Scoring submission : {}'.format(label))
            score_submission(config, submission_id)
        # Get running instances and process events
        instance_ids = list_ec2_instance_ids(config)
        for instance_id in instance_ids:
            if not _is_ready(config, instance_id):
                continue
            tags = _get_tags(instance_id)
            if 'submission_id' not in tags:
                continue
            label = tags['Name']
            submission_id = int(tags['submission_id'])
            state = get_submission_state(config, submission_id)
            if state == 'sent_to_training':
                exit_status = upload_submission(
                    config, instance_id, submission_id)
                if exit_status != 0:
                    logger.error(
                        'Cannot upload submission "{}"'
                        ', an error occured'.format(label))
                    continue
                # start training HERE
                exit_status = launch_train(config, instance_id, submission_id)
                if exit_status != 0:
                    logger.error(
                        'Cannot start training of submission "{}"'
                        ', an error occured.'.format(label))
                    continue
                set_submission_state(config, submission_id, 'training')
            elif state == 'training':
                # in any case (successful training or not)
                # download the log 
                download_log(config, instance_id, submission_id)
                if _training_finished(config, instance_id, submission_id):                   
                    logger.info(
                        'Training of "{}" finished, checking '
                        'if successful or not...'.format(label))
                    if _training_successful(
                            config,
                            instance_id,
                            submission_id):
                        logger.info('Training of "{}" was successful'.format(label))
                        if conf.get(MEMORY_PROFILING_FIELD):
                            logger.info('Download max ram usage info of "{}"'.format(label))
                            download_mprof_data(config, instance_id, submission_id)
                            max_ram = _get_submission_max_ram(config, submission_id)
                            logger.info('Max ram usage of "{}": {}MB'.format(label, max_ram))
                            set_submission_max_ram(config, submission_id, max_ram)
                            
                        logger.info('Downloading the predictions of "{}"'.format(label))
                        path = download_predictions(
                            config, instance_id, submission_id)
                        set_predictions(config, submission_id, path, ext='npz')
                        set_submission_state(config, submission_id, 'tested')
                    else:
                        logger.info('Training of "{}" failed'.format(label))
                        set_submission_state(
                            config, submission_id, 'training_error')
                        error_msg = _get_traceback(
                            _get_log_content(config, submission_id)
                        )
                        set_submission_error_msg(
                            config, submission_id, error_msg)
                    # training finished, so terminate the instance
                    terminate_ec2_instance(config, instance_id)
        time.sleep(secs)


def launch_ec2_instance_and_train(config, submission_id):
    """
    This function does the following steps:

    1) launch a new ec2 instance
    2) upload the submission into the ec2 the instance
    3) train the submission
    4) get back the predictions and the log
    5) terminate the ec2 instance.

    Parameters
    ----------

    config : dict
        configuration

    submission_id : int
        submission id

    """
    instance, = launch_ec2_instances(config, nb=1)
    set_submission_state(config, submission_id, 'sent_to_training')
    _wait_until_ready(config, instance.id)
    train_on_existing_ec2_instance(config, instance.id, submission_id)
    terminate_ec2_instance(config, instance.id)


def _wait_until_ready(config, instance_id):
    """
    Wait until an ec2 instance is ready.

    Parameters
    ----------

    config : dict
        configuration

    instance_id : str

    """
    logger.info('Waiting until instance "{}" is ready...'.format(instance_id))
    conf = config[AWS_CONFIG_SECTION]
    secs = int(conf[CHECK_STATUS_INTERVAL_SECS_FIELD])
    while not _is_ready(config, instance_id):
        time.sleep(secs)


def train_on_existing_ec2_instance(config, instance_id, submission_id):
    """
    Train a submission on a ready ec2 instance
    the steps followed by this function are the following:
        1) upload the submission code to the instance
        2) launch training in a screen
        3) wait until training is finished
        4) download the predictions
        5) download th log
        6) set the predictions in the database
        7) score the submission
    """
    conf = config[AWS_CONFIG_SECTION]
    upload_submission(config, instance_id, submission_id)
    launch_train(config, instance_id, submission_id)
    set_submission_state(config, submission_id, 'training')
    _wait_until_train_finished(config, instance_id, submission_id)
    download_log(config, instance_id, submission_id)
    
    label = _get_submission_label_by_id(config, submission_id)
    if _training_successful(config, instance_id, submission_id):

        logger.info('Training of "{}" was successful'.format(
            label, instance_id))
        if conf[MEMORY_PROFILING_FIELD]:
            logger.info('Download max ram usage info of "{}"'.format(label))
            download_mprof_data(config, instance_id, submission_id)
            max_ram = _get_submission_max_ram(config, submission_id)
            logger.info('Max ram usage of "{}": {}MB'.format(label, max_ram))
            set_submission_max_ram(config, submission_id, max_ram)
            
        logger.info('Downloading predictions of : "{}"'.format(label))
        predictions_folder_path = download_predictions(
            config, instance_id, submission_id)
        set_predictions(config, submission_id,
                        predictions_folder_path, ext='npz')
        set_submission_state(config, submission_id, 'tested')
        logger.info('Scoring "{}"'.format(label))
        score_submission(config, submission_id)
    else:
        logger.info('Training of "{}" in "{}" failed'.format(
            label, instance_id))
        set_submission_state(config, submission_id, 'training_error')
        error_msg = _get_traceback(
            _get_log_content(config, submission_id))
        set_submission_error_msg(config, submission_id, error_msg)


def _wait_until_train_finished(config, instance_id, submission_id):
    """
    Wait until the training of a submission is finished in an ec2 instance.
    To check whether the training is finished, we check whether
    the screen is still active. If the screen is not active anymore,
    then we consider that the training has either finished or failed.
    """
    logger.info('Wait until training of submission "{}" is '
                'finished on instance "{}"...'.format(submission_id,
                                                      instance_id))
    conf = config[AWS_CONFIG_SECTION]
    secs = int(conf[CHECK_FINISHED_TRAINING_INTERVAL_SECS_FIELD])
    while not _training_finished(config, instance_id, submission_id):
        time.sleep(secs)


def launch_ec2_instances(config, nb=1):
    """
    Launch new ec2 instance(s)
    """
    conf = config[AWS_CONFIG_SECTION]
    ami_image_id = conf[AMI_IMAGE_ID_FIELD]
    instance_type = conf[INSTANCE_TYPE_FIELD]
    key_name = conf[KEY_NAME_FIELD]
    security_group = conf[SECURITY_GROUP_FIELD]

    logger.info('Launching {} new ec2 instance(s)...'.format(nb))

    # tag all instances using RAMP_AWS_BACKEND_TAG to be able
    # to list all instances later
    tags = [{
        'ResourceType': 'instance',
        'Tags': [
            {'Key': RAMP_AWS_BACKEND_TAG, 'Value': '1'},
        ]
    }]
    resource = boto3.resource('ec2')
    instances = resource.create_instances(
        ImageId=ami_image_id,
        MinCount=nb,
        MaxCount=nb,
        InstanceType=instance_type,
        KeyName=key_name,
        TagSpecifications=tags,
        SecurityGroups=[security_group],
    )
    return instances


def terminate_ec2_instance(config, instance_id):
    """
    Terminate an ec2 instance

    Parameters
    ----------

    config : dict
        configuration

    instance_id : str
        instance id
    """
    resource = boto3.resource('ec2')
    logger.info('Killing the instance {}...'.format(instance_id))
    return resource.instances.filter(InstanceIds=[instance_id]).terminate()


def list_ec2_instance_ids(config):
    """
    List all running instances ids

    Parameters
    ----------

    config : dict
        configuration

    Returns
    -------

    list of str
    """
    client = boto3.client('ec2')
    instances = client.describe_instances(
        Filters=[
            {'Name': 'tag:' + RAMP_AWS_BACKEND_TAG, 'Values': ['1']},
            {'Name': 'instance-state-name', 'Values': ['running']},
        ]
    )
    instance_ids = [
        inst['Instances'][0]['InstanceId']
        for inst in instances['Reservations']
    ]
    return instance_ids


def status_of_ec2_instance(config, instance_id):
    """
    Get the status of an ec2 instance

    Parameters
    ----------

    config : dict
        configuration

    instance_id : str
        instance id

    Returns
    -------

    dict with instance status or None
    if can return None if the instance has just been launched and is
    not even ready to give the status.
    """
    client = boto3.client('ec2')
    responses = client.describe_instance_status(
        InstanceIds=[instance_id])['InstanceStatuses']
    if len(responses) == 1:
        return responses[0]
    else:
        return None


def upload_submission(config, instance_id, submission_id):
    """
    Upload a submission on an ec2 instance

    Parameters
    ----------

    config : dict
        configuration

    instance_id : str
        instance id

    submission_id : int
        submission id
    """
    conf = config[AWS_CONFIG_SECTION]
    ramp_kit_folder = conf[REMOTE_RAMP_KIT_FOLDER_FIELD]
    dest_folder = os.path.join(ramp_kit_folder, SUBMISSIONS_FOLDER)
    submission_path = _get_submission_path(config, submission_id)
    return _upload(config, instance_id, submission_path, dest_folder)


def download_log(config, instance_id, submission_id, folder=None):
    """
    Download the log file from an ec2 instance to a local folder `folder`.
    If `folder` is not given, then the log file is downloaded on
    the value in config corresponding to `LOCAL_LOG_FOLDER_FIELD`.
    If `folder` is given, then the log file is put in `folder`.

    Parameters
    ----------

    config : dict
        configuration

    instance_id : str
        instance id

    submission_id : int
        submission id

    folder : str or None
        folder where to download the log
    """
    conf = config[AWS_CONFIG_SECTION]
    ramp_kit_folder = conf[REMOTE_RAMP_KIT_FOLDER_FIELD]
    submission_folder_name = _get_submission_folder_name(submission_id)
    source_path = os.path.join(
        ramp_kit_folder, SUBMISSIONS_FOLDER, submission_folder_name, 'log')
    if folder is None:
        dest_path = os.path.join(
            conf[LOCAL_LOG_FOLDER_FIELD], submission_folder_name, 'log')
    else:
        dest_path = folder
    try:
        os.makedirs(os.path.dirname(dest_path))
    except OSError:
        pass
    return _download(config, instance_id, source_path, dest_path)


def _get_log_content(config, submission_id):
    """
    Get the content of the log file.
    The log file must have been downloaded locally with `download_log`
    for this to work.

    Returns
    -------
    
    a str with the content of the log file
    """
    conf = config[AWS_CONFIG_SECTION]
    ramp_kit_folder = conf[REMOTE_RAMP_KIT_FOLDER_FIELD]
    submission_folder_name = _get_submission_folder_name(submission_id)
    path = os.path.join(
        conf[LOCAL_LOG_FOLDER_FIELD], 
        submission_folder_name, 
        'log')
    try: 
        content = open(path).read()
        content = _filter_colors(content)
        return content
    except IOError:
        logger.error('Could not open log file of "{}" when trying to get log content'.format(submission_id))
        return ''


def _get_traceback(content):
    """
    get the traceback part from the content where content
    is a str containing the standard error/output of 
    a python process. It is used to get the traceback
    of `ramp_test_submission` when there is an error.

    Returns
    -------

    str with the traceback
    """
    if not content:
        return ''
    #cut_exception_text = content.rfind('--->') 
    # was like commented line above in ramp-board
    # but there is no ---> in logs when we use
    # ramp_test_submission, so we just search for the
    # first occurence of 'Traceback'.
    cut_exception_text = content.find('Traceback')
    if cut_exception_text > 0:
        content = content[cut_exception_text:]
    return content
 

def _filter_colors(content):
    # filter linux colors from a string
    # check (https://pypi.org/project/colored/)
    return re.sub(r'(\x1b\[)([\d]+;[\d]+;)?[\d]+m', '', content)


def download_mprof_data(config, instance_id, submission_id, folder=None):
    """
    Download the dat file for memory profiling from an ec2 instance to a local folder `folder`.
    If `folder` is not given, then the dat file is downloaded on
    the value in config corresponding to `LOCAL_LOG_FOLDER_FIELD`.
    If `folder` is given, then the dat file is put in `folder`.
    IMPORTANT: memory_profiler >= 0.52.0 should be installed in the remote instances.
    
    Parameters
    ----------

    config : dict
        configuration

    instance_id : str
        instance id

    submission_id : int
        submission id

    folder : str or None
        folder where to download the log
    """
    conf = config[AWS_CONFIG_SECTION]
    ramp_kit_folder = conf[REMOTE_RAMP_KIT_FOLDER_FIELD]
    submission_folder_name = _get_submission_folder_name(submission_id)
    source_path = os.path.join(
        ramp_kit_folder,
        SUBMISSIONS_FOLDER,
        submission_folder_name,
        'mprof.dat')
    if folder is None:
        dest_path = os.path.join(
            conf[LOCAL_LOG_FOLDER_FIELD], submission_folder_name) + os.sep
    else:
        dest_path = folder
    return _download(config, instance_id, source_path, dest_path)


def _get_submission_max_ram(config, submission_id):
    conf = config[AWS_CONFIG_SECTION]
    ramp_kit_folder = conf[REMOTE_RAMP_KIT_FOLDER_FIELD]
    submission_folder_name = _get_submission_folder_name(submission_id)
    dest_path = os.path.join(conf[LOCAL_LOG_FOLDER_FIELD], submission_folder_name)
    filename = os.path.join(dest_path, 'mprof.dat')
    max_mem = 0.
    for line in open(filename).readlines()[1:]:
        _, mem, _ = line.split()
        max_mem = max(max_mem, float(mem))
    return max_mem


def download_predictions(config, instance_id, submission_id, folder=None):
    """
    Download the predictions from an ec2 instance into a local folder `folder`.
    If `folder` is not given, then the predictions are downloaded on
    the value in config corresponding to `LOCAL_PREDICTIONS_FOLDER_FIELD`.
    If `folder` is given, then the are put in `folder`.

    Parameters
    ----------

    config : dict
        configuration

    instance_id : str
        instance id

    submission_id : int
        submission id

    folder : str or None
        folder where to download the predictions

    Returns
    -------

    path of the folder of `training_output` containing the predictions
    """
    conf = config[AWS_CONFIG_SECTION]
    submission_folder_name = _get_submission_folder_name(submission_id)
    source_path = _get_remote_training_output_folder(
        config, instance_id, submission_id) + '/'
    if folder is None:
        dest_path = os.path.join(
            conf[LOCAL_PREDICTIONS_FOLDER_FIELD], submission_folder_name)
    else:
        dest_path = folder
    _download(config, instance_id, source_path, dest_path)
    return dest_path


def _get_remote_training_output_folder(config, instance_id, submission_id):
    """
    Get remote training output folder for a submission in an instance.
    For instance, it returns something like :
    ~/ramp-kits/iris/submissions/submission_000001/training_output.
    """
    conf = config[AWS_CONFIG_SECTION]
    ramp_kit_folder = conf[REMOTE_RAMP_KIT_FOLDER_FIELD]
    submission_folder_name = _get_submission_folder_name(submission_id)
    path = os.path.join(ramp_kit_folder, SUBMISSIONS_FOLDER,
                        submission_folder_name, 'training_output')
    return path


def launch_train(config, instance_id, submission_id):
    """
    Launch the training of a submission on an ec2 instance.
    A screen named `submission_folder_name` (see below)
    is created and within that screen `ramp_test_submission`
    is launched.

    Parameters
    ----------

    instance_id : str
        instance id

    submission_id : int
        submission id
    """
    conf = config[AWS_CONFIG_SECTION]
    ramp_kit_folder = conf[REMOTE_RAMP_KIT_FOLDER_FIELD]
    submission_folder_name = _get_submission_folder_name(submission_id)
    submission = get_submission_by_id(config, submission_id)
    values = {
        'ramp_kit_folder': ramp_kit_folder,
        'submission': submission_folder_name,
        'submission_folder': os.path.join(ramp_kit_folder, SUBMISSIONS_FOLDER,
                                          submission_folder_name),
        'log': os.path.join(ramp_kit_folder, SUBMISSIONS_FOLDER,
                            submission_folder_name, 'log')
    }
    # we use python -u so that standard input/output are flushed
    # and thus we can retrieve the log file live during training
    # without waiting for the process to finish.
    # We use an espace character around "$" because it is interpreted
    # before being run remotely and leads to an empty string
    run_cmd = r"python -u \$(which ramp_test_submission) --submission {submission} --save-y-preds "
    if conf.get(MEMORY_PROFILING_FIELD):
        run_cmd = "mprof run --output={submission_folder}/mprof.dat --include-children " + run_cmd
    cmd = (
        "screen -dm -S {submission} sh -c '. ~/.profile;"+
        "cd {ramp_kit_folder};"+
        "rm -fr {submission_folder}/training_output;"+
        "rm -f {submission_folder}/log;"+
        "rm -f {submission_folder}/mprof.dat;"+
        run_cmd+">{log} 2>&1'"
    )
    cmd = cmd.format(**values)
    # tag the ec2 instance with info about submission
    _tag_instance_by_submission(instance_id, submission)
    label = _get_submission_label(submission)
    logger.info('Launch training of {}..'.format(label))
    return _run(config, instance_id, cmd)


def abort_training(config, instance_id, submission_id):
    """
    Stop training a submission.
    This is done by killing the screen where
    the training process is.

    Parameters
    ----------

    instance_id : str
        instance id

    submission_id : int
        submission id
    """
    cmd = 'screen -S {} -X quit'.format(submission_id)
    return _run(config, instance_id, cmd)


def _get_submission_folder_name(submission_id):
    return 'submission_{:09d}'.format(submission_id)


def _get_submission_path(config, submission_id):
    # Get local submission path
    submission = get_submission_by_id(config, submission_id)
    return submission.path


def _get_submission_label_by_id(config, submission_id):
    submission = get_submission_by_id(config, submission_id)
    return _get_submission_label(submission)


def _get_submission_label(submission):
    # Submissions in AWS are tagged by the label
    label = '{}_{}'.format(submission.id, submission.name)
    return label


def _upload(config, instance_id, source, dest):
    """
    Upload a file to an ec2 instance

    Parameters
    ----------

    instance_id : str
        instance id

    source : str
        local file or folder

    dest : str
        remote file or folder
    """
    dest = '{user}@{ip}:' + dest
    return _rsync(config, instance_id, source, dest)


def _download(config, instance_id, source, dest):
    """
    Download a file from an ec2 instance

    Parameters
    ----------

    instance_id : str
        instance id

    source : str
        remote file or folder

    dest : str
        local file or folder

    """
    source = '{user}@{ip}:' + source
    return _rsync(config, instance_id, source, dest)


def _rsync(config, instance_id, source, dest):
    """
    Run rsync from/to an ec2 instance

    Parameters
    ----------

    config : dict
        configuration

    instance_id : str
        instance id

    source : str
        source file or folder

    dest : str
        dest file or folder

    """
    conf = config[AWS_CONFIG_SECTION]
    key_path = conf[KEY_PATH_FIELD]
    ami_username = conf[AMI_USER_NAME_FIELD]

    resource = boto3.resource('ec2')
    inst = resource.Instance(instance_id)
    ip = inst.public_ip_address
    fmt = {'user': ami_username, 'ip': ip}
    values = {
        'user': ami_username,
        'ip': ip,
        'cmd': "ssh -o 'StrictHostKeyChecking no' -i " + key_path,
        'source': source.format(**fmt),
        'dest': dest.format(**fmt),
    }
    cmd = "rsync -e \"{cmd}\" -avzP {source} {dest}".format(**values)
    logger.debug(cmd)
    return call(cmd, shell=True)


def _run(config, instance_id, cmd, return_output=False):
    """
    Run a shell command remotely on an ec2 instance and
    return either the exit status if `return_output` is False
    or the standard output if `return_output` is True

    Parameters
    ----------

    config : dict
        configuration

    instance_id : str
        instance id

    cmd : str
        shell command

    return_output : bool
        whether to return the standard output

    Returns
    -------

    If `return_output` is True, then a str containing
    the standard output.
    If `return_output` is False, then an int containing
    the exit status of the command.
    """
    conf = config[AWS_CONFIG_SECTION]
    key_path = conf[KEY_PATH_FIELD]
    ami_username = conf[AMI_USER_NAME_FIELD]

    resource = boto3.resource('ec2')
    inst = resource.Instance(instance_id)
    ip = inst.public_ip_address
    values = {
        'user': ami_username,
        'ip': ip,
        'ssh': "ssh -o 'StrictHostKeyChecking no' -i " + key_path,
        'cmd': cmd,
    }
    cmd = "{ssh} {user}@{ip} \"{cmd}\"".format(**values)
    logger.debug(cmd)
    if return_output:
        return check_output(cmd, shell=True)
    else:
        return call(cmd, shell=True)


def _is_ready(config, instance_id):
    """
    Return True if an instance is ready to be used
    """
    st = status_of_ec2_instance(config, instance_id)
    if st:
        check = st['InstanceStatus']['Details'][0]['Status']
        return check == 'passed'
    else:
        return False


def _training_finished(config, instance_id, submission_id):
    """
    Return True if a submission has finished training
    """
    submission_folder_name = _get_submission_folder_name(submission_id)
    return not _has_screen(config, instance_id, submission_folder_name)


def _training_successful(config, instance_id, submission_id):
    """
    Return True if a finished submission have been trained successfully.
    If the folder training_output exists and each fold directory contains
    .npz prediction files we consider that the training was successful.
    """
    folder = _get_remote_training_output_folder(
        config, instance_id, submission_id)
    
    cmd = "ls -l {}|grep fold_|wc -l".format(folder)
    nb_folds = int(_run(config, instance_id, cmd, return_output=True))

    cmd = "find {}|egrep 'fold.*/y_pred_train.npz'|wc -l".format(folder)
    nb_train_files = int(_run(config, instance_id, cmd, return_output=True))

    cmd = "find {}|egrep 'fold.*/y_pred_test.npz'|wc -l".format(folder)
    nb_test_files = int(_run(config, instance_id, cmd, return_output=True))
    
    if nb_folds == 0 or nb_train_files == 0 or nb_test_files == 0:
        return False
    return nb_folds == nb_train_files == nb_test_files


def _folder_exists(config, instance_id, folder):
    """
    Return True if a folder exists remotely in an instance
    """
    cmd = '[ -d {} ] && echo "1" || echo "0"'
    exists = bool(_run(config, instance_id, cmd, return_output=True))
    return exists


def _has_screen(config, instance_id, screen_name):
    """
    Return True if a screen named `screen_name` exists on
    the ec2 instance `instance_id`
    """
    cmd = "screen -ls|awk '{{print $1}}' | cut -d. -f2 | grep {}| wc -l"
    cmd = cmd.format(screen_name)
    nb = int(_run(config, instance_id, cmd, return_output=True))
    return nb > 0


def _tag_instance_by_submission(instance_id, submission):
    """
    Add tags to an instance with infos from the submission to know which
    submission is being trained on the instance.
    """
    _add_or_update_tag(instance_id, 'submission_id', str(submission.id))
    _add_or_update_tag(instance_id, 'submission_name', submission.name)
    _add_or_update_tag(instance_id, 'event_name', submission.event.name)
    _add_or_update_tag(instance_id, 'team_name', submission.team.name)
    name = _get_submission_label(submission)
    _add_or_update_tag(instance_id, 'Name', name)


def _add_or_update_tag(instance_id, key, value):
    client = boto3.client('ec2')
    tags = [
        {'Key': key, 'Value': value},
    ]
    return client.create_tags(Resources=[instance_id], Tags=tags)


def _get_tags(instance_id):
    client = boto3.client('ec2')
    filters = [
        {'Name': 'resource-id', 'Values': [instance_id]}
    ]
    response = client.describe_tags(Filters=filters)
    for t in response['Tags']:
        t['Key'], t['Value']
    return {t['Key']: t['Value'] for t in response['Tags']}


def _delete_tag(instance_id, key):
    client = boto3.client('ec2')
    tags = [{'Key': key}]
    return client.delete_tags(Resources=[instance_id], Tags=tags)
