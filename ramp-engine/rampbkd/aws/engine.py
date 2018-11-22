import logging
import os
import sys

from . import api as aws


logger = logging.getLogger('ramp_aws_engine')
logger.setLevel(logging.DEBUG)


class AWSEngine:

    def __init__(self, config, conda_env='base', submission='starting_kit',
                 ramp_kit_dir='.', ramp_data_dir='.'):
        self.config = config
        self.conda_env = conda_env
        self.submission = submission
        self.ramp_kit_dir = ramp_kit_dir
        self.ramp_data_dir = ramp_data_dir
        self.submission_path = os.path.join(
            self.ramp_kit_dir, 'submissions', self.submission)
        self.status = 'initialized'

    def setup(self):
        self.instance, = aws.launch_ec2_instances(self.config)
        exit_status = aws.upload_submission(
            self.config, self.instance.id, self.submission_path)
        if exit_status != 0:
            logger.error(
                'Cannot upload submission "{}"'
                ', an error occured'.format(self.submission))
        else:
            logger.info("Uploaded submission '{}'".format(self.submission))
            self.status = 'setup'

    def launch_submission(self):
        if self.status == 'running':
            raise RuntimeError("Cannot launch submission: one is already "
                               "started")
        exit_status = aws.launch_train(
            self.config, self.instance.id, self.submission_path)
        if exit_status != 0:
            logger.error(
                'Cannot start training of submission "{}"'
                ', an error occured.'.format(self.submission))
        else:
            self.status = 'running'
        return exit_status

    @property
    def status(self):
        status = self._status
        if status == 'running':
            if aws._training_finished(
                    self.config, self.instance.id, self.submission_path):
                status = 'finished'
                self._status = status
        return status

    @status.setter
    def status(self, status):
        self._status = status

    def collect_submission(self):
        aws._wait_until_train_finished(
            self.config, self.instance.id, self.submission_path)
        self.status = 'finished'

        aws.download_log(self.config, self.instance.id, self.submission_path)

        if aws._training_successful(
                self.config, self.instance.id, self.submission_path):
            predictions_folder_path = aws.download_predictions(
                self.config, self.instance.id, self.submission_path)
        else:
            print("problem!")

    def teardown(self):
        aws.terminate_ec2_instance(self.config, self.instance.id)
