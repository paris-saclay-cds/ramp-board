import logging

from ..base import BaseWorker
from . import api as aws


logger = logging.getLogger('RAMP-AWS')
logger.setLevel(logging.DEBUG)


class AWSWorker(BaseWorker):
    """
    Run RAMP submissions on Amazon.

    """

    def __init__(self, config, submission):
        super(AWSWorker, self).__init__(config, submission)
        self.submissions_path = self.config['submissions_dir']

    def setup(self):
        """Set up the worker.

        This will launch an instance on Amazon, and copy the submission
        to the instance.
        """
        # sanity check for the configuration variable
        for required_param in ('instance_type', 'access_key_id'):
            self._check_config_name(self.config, required_param)

        self.instance, = aws.launch_ec2_instances(self.config)
        exit_status = aws.upload_submission(
            self.config, self.instance.id, self.submission,
            self.submissions_path)
        if exit_status != 0:
            logger.error(
                'Cannot upload submission "{}"'
                ', an error occured'.format(self.submission))
        else:
            logger.info("Uploaded submission '{}'".format(self.submission))
            self.status = 'setup'

    def launch_submission(self):
        """Launch the submission.

        Basically, this runs ``ramp_test_submission`` inside the
        Amazon instance.
        """
        if self.status == 'running':
            raise RuntimeError("Cannot launch submission: one is already "
                               "started")
        exit_status = aws.launch_train(
            self.config, self.instance.id, self.submission)
        if exit_status != 0:
            logger.error(
                'Cannot start training of submission "{}"'
                ', an error occured.'.format(self.submission))
        else:
            self.status = 'running'
        return exit_status

    def _is_submission_finished(self):
        return aws._training_finished(
            self.config, self.instance.id, self.submission)

    def collect_results(self):
        super(AWSWorker, self).collect_results()
        if self.status == 'running':
            aws._wait_until_train_finished(
                self.config, self.instance.id, self.submission)
            self.status = 'finished'
        if self.status != 'finished':
            raise ValueError("Cannot collect results if worker is not"
                             "'running' or 'finished'")

        aws.download_log(self.config, self.instance.id, self.submission)

        if aws._training_successful(
                self.config, self.instance.id, self.submission):
            _ = aws.download_predictions(  # noqa
                self.config, self.instance.id, self.submission)
            self.status = 'collected'
            return 0, ''
        else:
            error_msg = aws._get_traceback(
                aws._get_log_content(self.config, self.submission))
            self.status = 'collected'
            return 1, error_msg

    def teardown(self):
        """Terminate the Amazon instance"""
        aws.terminate_ec2_instance(self.config, self.instance.id)
        super(AWSWorker, self).teardown()
