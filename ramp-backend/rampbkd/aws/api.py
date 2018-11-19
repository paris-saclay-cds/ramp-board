"""
Backend support for Amazon EC2 instances.
The goal of this module is to provide a set of
helper functions to train a submission on an ec2 instance.
The first step is to add a section 'aws' on the configuration
file as the following.

## Configuration details

aws:
    ami_image_id : ami-0bc19972
    ami_user_name : ubuntu
    instance_type : t2.micro
    key_name: key
    key_path: /home/user/.ssh/id_rsa
    security_group : launch-wizard-1
    remote_ramp_kit_folder : ~/ramp/iris
    local_predictions_folder : ./predictions
    local_log_folder : ./logs
    check_status_interval_secs : 60
    check_finished_training_interval_secs : 60
    train_loop_interval_secs : 60

`ami_image_id` is the id of the image to use, it should start with 'ami-'.
The AMI should contain a folder `remote_ramp_kit_folder` (see below)
which contains the ramp kit.

`ami_user_name` is the username to connect with remotely on ec2 instances.

`instance_type` is the instance type (check https://ec2instances.info/).

`key_name` is the name of the key to connect with, so `key_name` should
exist im amazon. It can be created using their web app, or manually via
`aws` like this :
> aws ec2 import-key-pair --key-name <put key name here>
  --public-key-material "<put public key here>"

`security_group` is the name of the security group to use.
Security groups control which ports are accepted/blocked inbound or outbound.
They can be created in the web app of amazon. Use `default`
to use the default one.

`remote_ramp_kit_folder` is the folder in the ec2 instance
where the ramp-kit will reside. It should
be possible to launch `ramp_test_submission` in that folder.

`local_predictions_folder` is the local folder where the predictions are
downloaded (from the ec2 instance).

`local_log_folder` is the local folder where the logs are downloaded
(from the ec2 instance). The logs contain the standard output obtained
from running `ramp_test_submission` for a given submission.

`check_status_interval_secs` is the number of secs to wait until we
recheck whether an ec2 instance is ready to be used.

`check_finished_training_interval_secs` is the number of secs to wait
until we recheck whether the training of a submission in an ec2
instance is finished.

`train_loop_interval_secs` is the number of secs to wait each time we
process new events in `train_loop`

## Using the API

Once configuration is ready, the most straighforward way to use the API is
to use the function `launch_ec2_instance_and_train`. It does the full pipeline
in one pass. That is, it launches an ec2 instance, waits until it is
ready, upload submission, starts training, wait for training to
finish, download the predictions and logs, store the predictions on the
database, then terminate the ec2 instance.

from rampbkd.config import read_backend_config
conf = read_backend_config('config.yml')
launch_ec2_instance_and_train(conf, submission_id)

Another way to use the API is to run a loop that listens for new
submissions and  run them.

from rampbkd.config import read_backend_config
conf = read_backend_config('config.yml')
train_loop(conf)

The other available functions can be used to do something more custom.
One could imagine to launch a pool of ec2 instances first, then have
a training loop which waits for submissions and run them in an ec2 instance
(nothing prevents us to train multiple submissions in the same ec2 instance).
The pool size could be adapted automatically to the need. Also, different
submissions could need different types of machines (GPU vs CPU).
"""
from __future__ import print_function, absolute_import, unicode_literals
import os
import time
import logging
from subprocess import call
from subprocess import check_output

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

logging.basicConfig(
    format='%(asctime)s ## %(message)s',
    level=logging.INFO,
    datefmt='%m/%d/%Y,%I:%M:%S')
logger = logging.getLogger(__name__)


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
            instance, = launch_ec2_instances(config, nb=1)
            _tag_instance_by_submission(instance.id, submission)
            logger.info('Launched instance "{}" for submission "{}"'.format(
                instance.id, submission))
            set_submission_state(config, submission.id, 'sent_to_training')

        # Get running instances and process events
        instance_ids = list_ec2_instance_ids(config)
        for instance_id in instance_ids:
            if not _is_ready(config, instance_id):
                continue
            tags = _get_tags(instance_id)
            if 'submission_id' not in tags:
                continue
            name = tags['Name']
            submission_id = int(tags['submission_id'])
            state = get_submission_state(config, submission_id)
            if state == 'sent_to_training':
                exit_status = upload_submission(
                    config, instance_id, submission_id)
                if exit_status != 0:
                    logger.error(
                        'Cannot upload submission "{}"'
                        ', an error occured'.format(name))
                    continue
                # start training HERE
                exit_status = launch_train(config, instance_id, submission_id)
                if exit_status != 0:
                    logger.error(
                        'Cannot start training of submission "{}"'
                        ', an error occured.'.format(name))
                    continue
                set_submission_state(config, submission_id, 'training')
            elif state == 'training':
                if _training_finished(config, instance_id, submission_id):
                    logger.info(
                        'Training of "{}" finished, checking '
                        'if successful or not...'.format(name))
                    if _training_successful(
                            config,
                            instance_id,
                            submission_id):
                        logger.info('Training of "{}" successful'.format(name))
                        path = download_predictions(
                            config, instance_id, submission_id)
                        set_predictions(config, submission_id, path, ext='npz')
                        set_submission_state(config, submission_id, 'tested')
                    else:
                        logger.info('Training of "{}" failed'.format(name))
                        set_submission_state(
                            config, submission_id, 'training_error')
                    # training finished, so terminate the instance
                    terminate_ec2_instance(config, instance_id)
                # in any case download the log
                download_log(config, instance_id, submission_id)
            elif state == 'tested':
                # TODO scoring
                pass
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
    """
    upload_submission(config, instance_id, submission_id)
    launch_train(config, instance_id, submission_id)
    set_submission_state(config, submission_id, 'training')
    _wait_until_train_finished(config, instance_id, submission_id)
    download_log(config, instance_id, submission_id)

    if _training_successful(config, instance_id, submission_id):
        logger.info('Training of "{}" in "{}" was success'.format(
            submission_id, instance_id))
        predictions_folder_path = download_predictions(
            config, instance_id, submission_id)
        set_predictions(config, submission_id,
                        predictions_folder_path, ext='npz')
        set_submission_state(config, submission_id, 'tested')
    else:
        logger.info('Training of "{}" in "{}" failed'.format(
            submission_id, instance_id))
        set_submission_state(config, submission_id, 'training_error')


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
            conf[LOCAL_LOG_FOLDER_FIELD], submission_folder_name)
    else:
        dest_path = folder
    return _download(config, instance_id, source_path, dest_path)


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
    cmd = (
        "screen -dm -S {submission} sh -c '. ~/.profile;"
        "cd {ramp_kit_folder};"
        "rm -fr {submission_folder}/training_output;"
        "ramp_test_submission --submission {submission} --save-y-preds "
        "2>&1 > {log}"
        "'".format(**values))
    # tag the ec2 instance with info about submission
    _tag_instance_by_submission(instance_id, submission)
    logger.info('Launch training of {}..'.format(submission))
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
    submission = get_submission_by_id(config, submission_id)
    return submission.path


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
    If the folder training_output exists we consider that training
    is successfully finished.
    """
    folder = _get_remote_training_output_folder(
        config, instance_id, submission_id)
    return _folder_exists(config, instance_id, folder)


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
    name = '{}_{}'.format(submission.id, submission.name)
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
