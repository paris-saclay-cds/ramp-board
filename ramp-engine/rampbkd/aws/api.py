from __future__ import print_function, absolute_import, unicode_literals
import os
import time
import logging
from subprocess import call
from subprocess import check_output
import re
import codecs

# amazon api
import botocore  # noqa
import boto3

from rampbkd.api import get_submission_by_id
from rampbkd.api import get_event_nb_folds


__all__ = [
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

# we disable the boto3 loggers because they are too verbose
for k in (logging.Logger.manager.loggerDict.keys()):
    if 'boto' in k:
        logging.getLogger(k).disabled = True

logger = logging.getLogger('ramp_aws')
logger.setLevel(logging.DEBUG)

# configuration fields
AWS_CONFIG_SECTION = 'aws'
PROFILE_NAME_FIELD = 'profile_name'
ACCESS_KEY_ID_FIELD = 'access_key_id'
SECRET_ACCESS_KEY_FIELD = 'secret_access_key'
REGION_NAME_FIELD = 'region_name'
AMI_IMAGE_ID_FIELD = 'ami_image_id'
AMI_IMAGE_NAME_FIELD = 'ami_image_name'
AMI_USER_NAME_FIELD = 'ami_user_name'
INSTANCE_TYPE_FIELD = 'instance_type'
KEY_PATH_FIELD = 'key_path'
KEY_NAME_FIELD = 'key_name'
SECURITY_GROUP_FIELD = 'security_group'
REMOTE_RAMP_KIT_FOLDER_FIELD = 'remote_ramp_kit_folder'
LOCAL_PREDICTIONS_FOLDER_FIELD = 'local_predictions_folder'
CHECK_STATUS_INTERVAL_SECS_FIELD = 'check_status_interval_secs'
CHECK_FINISHED_TRAINING_INTERVAL_SECS_FIELD = (
    'check_finished_training_interval_secs')
LOCAL_LOG_FOLDER_FIELD = 'local_log_folder'
TRAIN_LOOP_INTERVAL_SECS_FIELD = 'train_loop_interval_secs'
MEMORY_PROFILING_FIELD = 'memory_profiling'

HOOKS_SECTION = 'hooks'
HOOK_START_TRAINING = 'start_training'
HOOK_SUCCESSFUL_TRAINING = 'successful_training'
HOOK_FAILED_TRAINING = 'failed_training'
HOOKS = [
    HOOK_START_TRAINING,
    HOOK_SUCCESSFUL_TRAINING,
    HOOK_FAILED_TRAINING,
]
ALL_FIELDS = [
    PROFILE_NAME_FIELD,
    ACCESS_KEY_ID_FIELD,
    SECRET_ACCESS_KEY_FIELD,
    REGION_NAME_FIELD,
    AMI_IMAGE_ID_FIELD,
    AMI_IMAGE_NAME_FIELD,
    AMI_USER_NAME_FIELD,
    INSTANCE_TYPE_FIELD,
    KEY_PATH_FIELD,
    KEY_NAME_FIELD,
    SECURITY_GROUP_FIELD,
    REMOTE_RAMP_KIT_FOLDER_FIELD,
    LOCAL_PREDICTIONS_FOLDER_FIELD,
    CHECK_STATUS_INTERVAL_SECS_FIELD,
    CHECK_FINISHED_TRAINING_INTERVAL_SECS_FIELD,
    LOCAL_LOG_FOLDER_FIELD,
    TRAIN_LOOP_INTERVAL_SECS_FIELD,
    MEMORY_PROFILING_FIELD,
    HOOKS_SECTION,
]
ALL_FIELDS = set(ALL_FIELDS)
REQUIRED_FIELDS = ALL_FIELDS - set([HOOKS_SECTION])

# constants
RAMP_AWS_BACKEND_TAG = 'ramp_aws_backend_instance'
SUBMISSIONS_FOLDER = 'submissions'


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
    logger.info('Training of submission "{}" is '
                'finished on instance "{}".'.format(submission_id,
                                                    instance_id))


def launch_ec2_instances(config, nb=1):
    """
    Launch new ec2 instance(s)
    """
    conf = config[AWS_CONFIG_SECTION]
    ami_image_id = conf.get(AMI_IMAGE_ID_FIELD)
    ami_name = conf.get(AMI_IMAGE_NAME_FIELD)
    if ami_image_id and ami_name:
        raise ValueError(
            'The fields ami_image_id and ami_image_name cannot be both'
            'specified at the same time. Please specify either ami_image_id'
            'or ami_image_name')
    if ami_name:
        ami_image_id = _get_image_id(config, ami_name)

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
    sess = _get_boto_session(config)
    resource = sess.resource('ec2')
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


def _get_image_id(config, image_name):
    sess = _get_boto_session(config)
    client = sess.client('ec2')
    result = client.describe_images(Filters=[
        {
            'Name': 'name',
            'Values': [
                image_name
            ]
        }
    ])
    images = result['Images']
    if len(images) == 0:
        raise ValueError(
            'No image corresponding to the name "{}"'.format(image_name))
    elif len(images) > 1:
        raise ValueError(
            'Multiple images corresponding to the name "{}".'
            ' Please fix that'.format(image_name))
    else:
        image = images[0]
        image_id = image['ImageId']
        return image_id


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
    sess = _get_boto_session(config)
    resource = sess.resource('ec2')
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
    sess = _get_boto_session(config)
    client = sess.client('ec2')
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
    sess = _get_boto_session(config)
    client = sess.client('ec2')
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
    # submission_path = _get_submission_path(config, submission_id)
    submission_path = submission_id
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
        content = codecs.open(path, encoding='utf-8').read()
        content = _filter_colors(content)
        return content
    except IOError:
        logger.error('Could not open log file of "{}" when trying to get '
                     'log content'.format(submission_id))
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
    # cut_exception_text = content.rfind('--->')
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
    Download the dat file for memory profiling from an ec2 instance to a
    local folder `folder`.
    If `folder` is not given, then the dat file is downloaded on
    the value in config corresponding to `LOCAL_LOG_FOLDER_FIELD`.
    If `folder` is given, then the dat file is put in `folder`.
    IMPORTANT: memory_profiler >= 0.52.0 should be installed in the
    remote instances.

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
    dest_path = os.path.join(
        conf[LOCAL_LOG_FOLDER_FIELD], submission_folder_name)
    filename = os.path.join(dest_path, 'mprof.dat')
    max_mem = 0.
    for line in codecs.open(filename, encoding='utf-8').readlines()[1:]:
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
    # submission = get_submission_by_id(config, submission_id)
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
    run_cmd = (r"python -u \$(which ramp_test_submission) "
               r"--submission {submission} --save-y-preds ")
    if conf.get(MEMORY_PROFILING_FIELD):
        run_cmd = (
            "mprof run --output={submission_folder}/mprof.dat "
            "--include-children " + run_cmd)
    cmd = (
        "screen -dm -S {submission} sh -c '. ~/.profile;"
        "cd {ramp_kit_folder};"
        "rm -fr {submission_folder}/training_output;"
        "rm -f {submission_folder}/log;"
        "rm -f {submission_folder}/mprof.dat;"
        + run_cmd + ">{log} 2>&1'"
    )
    cmd = cmd.format(**values)
    # tag the ec2 instance with info about submission
    _tag_instance_by_submission(config, instance_id, submission_folder_name)
    label = submission_folder_name  # _get_submission_label(submission)
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
    return os.path.split(submission_id)[1]
    # return 'submission_{:09d}'.format(submission_id)


def _get_submission_path(config, submission_id):
    # Get local submission path
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

    sess = _get_boto_session(config)
    resource = sess.resource('ec2')
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

    sess = _get_boto_session(config)
    resource = sess.resource('ec2')
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
    # submission = get_submission_by_id(config, submission_id)
    # actual_nb_folds = get_event_nb_folds(config, submission.event.name)
    return nb_folds == nb_train_files == nb_test_files  # == actual_nb_folds


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


def _tag_instance_by_submission(config, instance_id, submission):
    """
    Add tags to an instance with infos from the submission to know which
    submission is being trained on the instance.
    """
    # _add_or_update_tag(
    #     config, instance_id, 'submission_id', str(submission.id))
    # _add_or_update_tag(
    #     config, instance_id, 'submission_name', submission.name)
    # _add_or_update_tag(
    #     config, instance_id, 'event_name', submission.event.name)
    # _add_or_update_tag(
    #     config, instance_id, 'team_name', submission.team.name)
    # name = _get_submission_label(submission)
    name = submission
    _add_or_update_tag(config, instance_id, 'Name', name)


def _add_or_update_tag(config, instance_id, key, value):
    sess = _get_boto_session(config)
    client = sess.client('ec2')
    tags = [
        {'Key': key, 'Value': value},
    ]
    return client.create_tags(Resources=[instance_id], Tags=tags)


def _get_tags(config, instance_id):
    sess = _get_boto_session(config)
    client = sess.client('ec2')
    filters = [
        {'Name': 'resource-id', 'Values': [instance_id]}
    ]
    response = client.describe_tags(Filters=filters)
    for t in response['Tags']:
        t['Key'], t['Value']
    return {t['Key']: t['Value'] for t in response['Tags']}


def _delete_tag(config, instance_id, key):
    sess = _get_boto_session(config)
    client = sess.client('ec2')
    tags = [{'Key': key}]
    return client.delete_tags(Resources=[instance_id], Tags=tags)


def _get_boto_session(config):
    conf = config[AWS_CONFIG_SECTION]
    if PROFILE_NAME_FIELD in conf:
        sess = boto3.session.Session(
            profile_name=conf[PROFILE_NAME_FIELD],
            region_name=conf[REGION_NAME_FIELD],
        )
        return sess
    elif ACCESS_KEY_ID_FIELD in conf and SECRET_ACCESS_KEY_FIELD in conf:
        sess = boto3.session.Session(
            aws_access_key_id=conf[ACCESS_KEY_ID_FIELD],
            aws_secret_access_key=conf[SECRET_ACCESS_KEY_FIELD],
            region_name=conf[REGION_NAME_FIELD],
        )
        return sess
    else:
        raise ValueError(
            'Please specify either "{}" or both of "{}" and "{}"'.format(
                PROFILE_NAME_FIELD, ACCESS_KEY_ID_FIELD,
                SECRET_ACCESS_KEY_FIELD))


def validate_config(config):
    """
    Check whether configuration is correct
    raises ValueError if it is not correct.
    """
    if AWS_CONFIG_SECTION not in config:
        raise ValueError(
            'Expects "{}" section in config'.format(AWS_CONFIG_SECTION))
    conf = config[AWS_CONFIG_SECTION]
    for k in conf.keys():
        if k not in ALL_FIELDS:
            raise ValueError('Invalid field : "{}"'.format(k))
    required_fields_ = REQUIRED_FIELDS - set([
        AMI_IMAGE_NAME_FIELD,
        AMI_IMAGE_ID_FIELD,
        PROFILE_NAME_FIELD,
        ACCESS_KEY_ID_FIELD,
        SECRET_ACCESS_KEY_FIELD,
    ])
    for k in required_fields_:
        if k not in conf:
            raise ValueError(
                'Required field "{}" missing from config'.format(k))
    if AMI_IMAGE_NAME_FIELD in conf and AMI_IMAGE_ID_FIELD in conf:
        raise ValueError(
            'The fields "{}" and "{}" cannot be both '
            'specified at the same time. Please specify only '
            'one of them'.format(AMI_IMAGE_NAME_FIELD, AMI_IMAGE_ID_FIELD))
    if AMI_IMAGE_NAME_FIELD not in conf and AMI_IMAGE_ID_FIELD not in conf:
        raise ValueError(
            'Please specify either  "{}" or "{}" in config.'.format(
                AMI_IMAGE_NAME_FIELD, AMI_IMAGE_ID_FIELD))
    if (PROFILE_NAME_FIELD in conf
            and (ACCESS_KEY_ID_FIELD in conf
                 or SECRET_ACCESS_KEY_FIELD in conf)):
        raise ValueError(
            'Please specify either "{}" or both of "{}" and "{}"'.format(
                PROFILE_NAME_FIELD, ACCESS_KEY_ID_FIELD,
                SECRET_ACCESS_KEY_FIELD))
    if (PROFILE_NAME_FIELD not in conf
            and not (ACCESS_KEY_ID_FIELD in conf
                     and SECRET_ACCESS_KEY_FIELD in conf)):
        raise ValueError('Please specify both "{}" and "{}"'.format(
            ACCESS_KEY_ID_FIELD, SECRET_ACCESS_KEY_FIELD,
        ))
    hooks = conf.get(HOOKS_SECTION)
    if hooks:
        for hook_name in hooks.keys():
            if hook_name not in HOOKS:
                hook_names = ','.join(HOOKS)
                raise ValueError(
                    'Invalid hook name : {}, hooks should be one of '
                    'these : {}'.format(hook_name, hook_names))
