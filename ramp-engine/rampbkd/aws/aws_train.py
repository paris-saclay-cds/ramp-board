import logging
import os
from subprocess import call
import time

import botocore  # amazon api
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
from rampbkd.api import get_event_nb_folds

from rampbkd.aws.api import (
    AWS_CONFIG_SECTION,
    TRAIN_LOOP_INTERVAL_SECS_FIELD,
    HOOKS_SECTION,
    HOOK_SUCCESSFUL_TRAINING, HOOK_START_TRAINING, HOOK_FAILED_TRAINING,
    CHECK_STATUS_INTERVAL_SECS_FIELD,
    MEMORY_PROFILING_FIELD, LOCAL_LOG_FOLDER_FIELD,
    launch_ec2_instances, terminate_ec2_instance,
    _tag_instance_by_submission, _add_or_update_tag,
    list_ec2_instance_ids, _is_ready, _get_tags,
    upload_submission, launch_train, download_log,
    _training_finished, _training_successful,
    _get_submission_max_ram, download_mprof_data, download_predictions,
    _get_log_content, _get_traceback, _get_submission_folder_name,
    _wait_until_train_finished)


logger = logging.getLogger('ramp_aws')


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
            try:
                instance, = launch_ec2_instances(config, nb=1)
            except botocore.exceptions.ClientError as ex:
                logger.info('Exception when launching a new instance : "{}"'.format(ex))
                logger.info('Skipping...')
                continue
            nb_trials = 0
            while nb_trials < conf.get('new_instance_nb_trials', 20):
                if instance.state.get('name') == 'running':
                    break
                nb_trials += 1
                time.sleep(conf.get('new_instance_check_interval', 6))
            
            _tag_instance_by_submission(config, instance.id, submission)
            _add_or_update_tag(config, instance.id, 'train_loop', '1')
            logger.info('Launched instance "{}" for submission "{}"'.format(
                instance.id, submission))
            set_submission_state(config, submission.id, 'sent_to_training')
        # Score tested submissions
        submissions = get_submissions(config, event_name, 'tested')
        for submission_id, _ in submissions:
            label = _get_submission_label_by_id(config, submission_id)
            logger.info('Scoring submission : {}'.format(label))
            score_submission(config, submission_id)
            _run_hook(config, HOOK_SUCCESSFUL_TRAINING, submission_id)
        # Get running instances and process events
        instance_ids = list_ec2_instance_ids(config)
        for instance_id in instance_ids:
            if not _is_ready(config, instance_id):
                continue
            tags = _get_tags(config, instance_id)
            # Filter instances that were not launched
            # by the training loop API
            if 'submission_id' not in tags:
                continue
            if tags.get('event_name') != event_name:
                continue
            if 'train_loop' not in tags:
                continue
            # Process each instance
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
                _run_hook(config, HOOK_START_TRAINING, submission_id)
 
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
                        _run_hook(config, HOOK_FAILED_TRAINING, submission_id)
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
    _run_hook(config, HOOK_START_TRAINING, submission_id)
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
        _run_hook(config, HOOK_SUCCESSFUL_TRAINING, submission_id)
    else:
        logger.info('Training of "{}" in "{}" failed'.format(
            label, instance_id))
        set_submission_state(config, submission_id, 'training_error')
        error_msg = _get_traceback(
            _get_log_content(config, submission_id))
        set_submission_error_msg(config, submission_id, error_msg)
        _run_hook(config, HOOK_FAILED_TRAINING, submission_id)


def _run_hook(config, hook_name, submission_id):
    """
    run hooks corresponding to hook_name
    """
    conf = config[AWS_CONFIG_SECTION]
    hooks = conf.get(HOOKS_SECTION)
    if not hooks:
        return
    if hook_name in hooks:
        submission = get_submission_by_id(config, submission_id)
        submission_folder_name = _get_submission_folder_name(submission_id)
        submission_folder = os.path.join(
            conf[LOCAL_LOG_FOLDER_FIELD], 
            submission_folder_name)
        env = {
            'RAMP_AWS_SUBMISSION_ID': str(submission_id),
            'RAMP_AWS_SUBMISSION_NAME': submission.name,
            'RAMP_AWS_EVENT': submission.event.name,
            'RAMP_AWS_TEAM': submission.team.name,
            'RAMP_AWS_HOOK': hook_name,
            'RAMP_AWS_SUBMISSION_FOLDER': submission_folder
        }
        env.update(os.environ)
        cmd = hooks[hook_name]
        if type(cmd) == list:
            cmd = ';'.join(cmd)
        logger.info('Running "{}" for hook {}'.format(cmd, hook_name))
        return call(cmd, shell=True, env=env)


def _get_submission_label_by_id(config, submission_id):
    submission = get_submission_by_id(config, submission_id)
    return _get_submission_label(submission)


def _get_submission_label(submission):
    # Submissions in AWS are tagged by the label
    label = '{}_{}'.format(submission.id, submission.name)
    return label
