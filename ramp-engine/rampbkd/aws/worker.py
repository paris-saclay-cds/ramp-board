import logging
import os
import sys

from ..base import BaseWorker
from . import api as aws


logger = logging.getLogger('ramp_aws_engine')
logger.setLevel(logging.DEBUG)


class AWSWorker(BaseWorker):
    """
    Run RAMP submissions on Amazon.

    """

    def __init__(self, config, submission, ramp_kit_dir):
        super(AWSWorker, self).__init__(config, submission)
        self.submission_path = os.path.join(
            ramp_kit_dir, 'submissions', self.submission)

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

    def _is_submission_finished(self):
        return aws._training_finished(
            self.config, self.instance.id, self.submission_path)

    def collect_results(self):
        if self.status == 'running':
            aws._wait_until_train_finished(
                self.config, self.instance.id, self.submission_path)
            self.status = 'finished'
        if self.status != 'finished':
            raise ValueError("Cannot collect results if worker is not"
                             "'running' or 'finished'")

        aws.download_log(self.config, self.instance.id, self.submission_path)

        if aws._training_successful(
                self.config, self.instance.id, self.submission_path):
            _ = aws.download_predictions(
                self.config, self.instance.id, self.submission_path)
            self.status = 'collected'
        else:
            # TODO deal with failed training
            print("problem!")

    def teardown(self):
        aws.terminate_ec2_instance(self.config, self.instance.id)
