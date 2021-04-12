import logging
import subprocess

from ..base import BaseWorker, _get_traceback
from . import api as aws


logger = logging.getLogger('RAMP-AWS')

log_file = 'aws_worker.log'
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')  # noqa
fileHandler = logging.FileHandler(log_file, mode='a')
fileHandler.setFormatter(formatter)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)

logger.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)


class AWSWorker(BaseWorker):
    """
    Run RAMP submissions on Amazon.

    Parameters
    ----------
    config : dict
        Configuration dictionary to set the worker. The required parameters
        are listed in the user guide.
    submission : str
        Name of the RAMP submission to be handle by the worker.

    Attributes
    ----------
    status : str
        The status of the worker. It should be one of the following state:

            * 'initialized': the worker has been instantiated.
            * 'setup': the worker has been set up.
            * 'error': setup failed / training couldn't be started
            * 'running': the worker is training the submission.
            * 'finished': the worker finished to train the submission.
            * 'collected': the results of the training have been collected.

    """

    def __init__(self, config, submission):
        super().__init__(config, submission)
        self.submissions_path = self.config['submissions_dir']
        self.instance = None

    def setup(self):
        """Set up the worker.

        This will launch an instance on Amazon, and copy the submission
        to the instance.
        """
        # sanity check for the configuration variable
        for required_param in ('instance_type', 'access_key_id'):
            self._check_config_name(self.config, required_param)

        logger.info("Setting up AWSWorker for submission '{}'".format(
            self.submission))
        _instances, status = aws.launch_ec2_instances(self.config)

        if not _instances:
            if status == 'retry':
                # there was a timeout error, put this submission back in the
                # queue and try again later
                logger.warning("Unable to launch instance for submission "
                               f"{self.submission}. Adding it back to the "
                               "queue and will try again later")
                self.status = 'retry'
            else:
                logger.error("Unable to launch instance for submission "
                             f"{self.submission}. An error occured: {status}")
                self.status = 'error'
            return
        else:
            logger.info("Instance launched for submission '{}'".format(
                        self.submission))
            self.instance, = _instances

        for _ in range(5):
            # try uploading the submission a few times, as this regularly fails
            exit_status = aws.upload_submission(
                self.config, self.instance.id, self.submission,
                self.submissions_path)
            if exit_status == 0:
                break
            else:
                logger.info("Uploading submission failed, retrying ...")
        if exit_status != 0:
            logger.error(
                'Cannot upload submission "{}"'
                ', an error occured'.format(self.submission))
            self.status = 'error'
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
        if self.status == 'error':
            raise RuntimeError("Cannot launch submission: the setup failed")
        try:
            exit_status = aws.launch_train(
                self.config, self.instance.id, self.submission)
        except Exception as e:
            logger.error(f'Unknown error occurred: {e}')
            exit_status = 1

        if exit_status != 0:
            logger.error(
                'Cannot start training of submission "{}"'
                ', an error occured.'.format(self.submission))
            self.status = 'error'
        else:
            self.status = 'running'
        return exit_status

    def _is_submission_finished(self):
        try:
            return aws._training_finished(
                self.config, self.instance.id, self.submission)
        except subprocess.CalledProcessError as e:
            # it is no longer possible to connect to the instance
            # possibly it was terminated from outside. restart the submission
            logger.warning("Unable to connect to the instance for submission "
                           f"{self.submission}. Adding the submission back to"
                           " the queue and will try again later")
            raise e

    def _is_submission_interrupted(self):
        """Check if spot instance has been marked as to be terminated by
        AWS."""
        return aws.is_spot_terminated(self.config, self.instance.id)

    def collect_results(self):
        super().collect_results()
        # Fail safe that is only used when worker used alone (not
        # with dispatcher).
        # The event config: 'check_finished_training_interval_secs'
        # is used here, but again only when worker used alone.
        if self.status == 'running':
            aws._wait_until_train_finished(
                self.config, self.instance.id, self.submission)
            self.status = 'finished'
        if self.status != 'finished':
            raise ValueError("Cannot collect results if worker is not"
                             "'running' or 'finished'")

        logger.info("Collecting submission '{}'".format(self.submission))
        exit_status = 0
        try:
            _ = aws.download_log(self.config,
                                 self.instance.id, self.submission)
        except Exception as e:
            logger.error("Error occurred when downloading the logs"
                         f" from the submission: {e}")
            exit_status = 2
            error_msg = str(e)
            self.status = 'error'
        if exit_status == 0:
            if aws._training_successful(
                    self.config, self.instance.id, self.submission):

                try:
                    _ = aws.download_predictions(self.config,
                                                 self.instance.id,
                                                 self.submission)
                except Exception as e:
                    logger.error("Downloading the prediction failed with"
                                 f"error {e}")
                    self.status = 'error'
                    exit_status, error_msg = 1, str(e)
                else:
                    self.status = 'collected'
                    exit_status, error_msg = 0, ''
            else:
                error_msg = _get_traceback(
                    aws._get_log_content(self.config, self.submission))
                self.status = 'collected'
                exit_status = 1
        logger.info(repr(self))
        return exit_status, error_msg

    def teardown(self):
        """Terminate the Amazon instance"""
        # Only terminate if instance is running
        if self.instance:
            instance_status = aws.check_instance_status(
                self.config, self.instance.id
            )
            if instance_status == 'running':
                aws.terminate_ec2_instance(self.config, self.instance.id)
        super().teardown()
