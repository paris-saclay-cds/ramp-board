import argparse

import os
import time
from datetime import datetime
import logging
from subprocess import call
from subprocess import check_output

from databoard.model import Submission
from databoard.model import db
from databoard.db_tools import get_earliest_new_submission
from databoard.db_tools import train_test_submission
from databoard.db_tools import update_leaderboards
from databoard.db_tools import update_all_user_leaderboards
from databoard.db_tools import compute_contributivity
from databoard.db_tools import compute_historical_contributivity
from databoard.db_tools import score_submission

import boto3 # amazon api

logging.basicConfig(
    format='%(asctime)s ## %(message)s',
    level=logging.INFO,
    datefmt='%m/%d/%Y,%I:%M:%S')
logger = logging.getLogger(__name__)


def train_loop(event_name='pollenating_insects_3_JNI_2017', 
               ami_image_id='ami-e5d72a9d', 
               ami_username='ubuntu',
               instance_type='g3.4xlarge', 
               key_name='ramp.studio', 
               ssh_key='/root/.ssh/amazon/rampstudio.pem',
               security_group='launch-wizard-74',
               sleep_time_secs=60, 
               timeout_secs=60*5,
               db_host='134.158.74.188',
               db_url=None,
               pgversion='9.3'):
        
    """
    Training loop for launching submissions to amazon ec2 through amazon API.

    Parameters
    ==========

    event_name : str
        Name of the event
    ami_image_id : str
        AMI image id that will be used for the training machines
    ami_username : str
        username that will be used to log into training machines
    instance_type : str
        type of the instance to use
    key_name : str
        SSL key name (look at key pairs in ec2 dashboard)
    ssh_key : str
        path of the key corresponding to `key_name` locally
    security_group : str
        security group in amazon
    sleep_time_secs : int
        number of secs to sleep between each iteration of the infinite loop
    timeout_secs : int
        number of secs before the training process is stopped and relaunched
        if its state is `sent_to_training` instead `training`
    db_host : str
        host of the database that will be used by training machines
    db_url : str, optional
        database url. if None, the local environment variable DATABOARD_DB_URL is used and
        `localhost` is replaced `db_host`
    pgversion : str
        postgresql version (9.3/9.5)
    """
    if db_url is None:
        db_url = os.getenv('DATABOARD_DB_URL').replace('localhost', db_host)

    ec2_resource = boto3.resource('ec2')
    ec2_client = boto3.client('ec2')
    while True:
        # Listen to new events
        new_submission = get_earliest_new_submission(event_name)
        if new_submission:
            logging.info('Got new submission : "{}"'.format(new_submission))
            instances = ec2_client.describe_instances(
                Filters=[
                    {
                        'Name': 'tag:event_name',
                        'Values':[event_name],
                    },
                    {
                        'Name': 'tag:submission_id',
                        'Values': [str(new_submission.id)],
                    }
                ]
            )
            nb_instances = len(instances['Reservations'])
            instance_ids = [inst['Instances'][0]['InstanceId'] 
                            for inst in instances['Reservations']]
            nb_running =  sum([ec2_resource.Instance(instance_id).state['Name'] == 'running' 
                               for instance_id in instance_ids])
            if nb_running > 1:
                logging.info(
                    'There is more than one instance for the submission "{}". '
                    'This should not happen. Please kill all except one of them.'.format(new_submission))
                logging.info(instance_ids)
            elif nb_running == 1:
                logging.info(
                    'There is already an instance for the submission "{}"' 
                    'so I will not launch a new amazon instance.'.format(new_submission))
            else:
                # nb_running is 0
                logging.info('Launching a new amazon instance for submission "{}"...'.format(new_submission))
                tags = [
                    {
                        'ResourceType': 'instance',
                        'Tags': [
                            {'Key': 'event_name', 'Value': event_name},
                            {'Key': 'submission_id', 'Value': str(new_submission.id)},
                            {'Key': 'Name', 'Value': str(new_submission.id) + '_' + new_submission.name}
                        ]
                    }
                ]
                instance, = ec2_resource.create_instances(
                    ImageId=ami_image_id, 
                    MinCount=1, MaxCount=1, 
                    InstanceType=instance_type, 
                    KeyName=key_name,
                    TagSpecifications=tags,
                    SecurityGroups=[security_group],
                )
                new_submission.state = 'sent_to_training' 
                db.session.commit()

                logging.info(
                    'Launched the instance, the instance id is {}, '
                    'launch time is : {}, Submission is "{}"'.format(instance.id, instance.launch_time, new_submission))

        # Process events

        # get list of `running` aws instances corresponding to `event_name`
        instances = ec2_client.describe_instances(
            Filters=[
                {'Name': 'tag:event_name', 'Values':[event_name]},
                {'Name': 'instance-state-name', 'Values': ['running']}
            ]
        )
        # get `ids` of instances
        instance_ids = [inst['Instances'][0]['InstanceId'] for inst in instances['Reservations']]
        # get `status` of instances
        instance_statuses = ec2_client.describe_instance_status(InstanceIds=instance_ids)['InstanceStatuses']
        # process each instance, depending on its state
        for instance_id, status in zip(instance_ids, instance_statuses):
            check_status = status['SystemStatus']['Details'][0]['Status']
            if check_status != 'passed':
                continue
            # check_status is 'passed', it means the aws instance can be used.
            # at this level, the submission can be either `new`, `sent_to_training`, `tested`, `training` or has an error
            inst = ec2_resource.Instance(instance_id)
            tags = dict((tag['Key'], tag['Value']) for tag in inst.tags)
            submission_id = int(tags['submission_id'])
            submission = Submission.query.filter_by(id=submission_id).one()
            if submission.state == 'sent_to_training':
                if _is_screen_launched(user=ami_username, ip=inst.public_ip_address, ssh_key=ssh_key):
                    # if there is already a launched training screen and `timeout_secs` is reached, kill the screen.
                    # This is used to prevent from the hanging problems that occurs in the db
                    # in the beginning, before the state becomes `training`.
                    delta = (datetime.now() - submission.sent_to_training_timestamp)
                    delta_secs = delta.total_seconds()
                    logging.info('Submission "{}", {:.3f}secs before timeout'.format(submission, timeout_secs - delta_secs))
                    if delta_secs >= timeout_secs:
                        logging.info('Timeout on submission "{}" on {} after {:.3f}secs, relaunching training'.format(submission, inst.public_ip_address, delta_secs))
                        cmd = "ssh -i {ssh_key} {user}@{ip} screen -S train -X quit".format(user=ami_username, ip=inst.public_ip_address, ssh_key=ssh_key)
                        call(cmd, shell=True)
                else:
                    # no training screen is running, so rsync submission code (only needed the first time) 
                    # and then launch a training screen
                    exit_status = _rsync_submission(
                        user=ami_username, 
                        ip=inst.public_ip_address, 
                        ssh_key=ssh_key,
                        submission_path=submission.path,
                    )
                    if exit_status != 0:
                        logging.info('Exit status not zero : problem in rsync submission for "{}"'.format(submission))
                        continue
                    logging.info('Launching training for the submission "{}"'.format(submission))
                    exit_status = _train_test(
                        user=ami_username,
                        ip=inst.public_ip_address,
                        submission=submission,
                        ssh_key=ssh_key,
                        db_url=db_url,
                    )
                    if exit_status != 0:
                        logging.info('Exit status not zero : problem in launching training for "{}"'.format(submission))
                        continue
                    # checkpoint for sent_to_training_timestamp
                    submission.sent_to_training_timestamp = datetime.now()
                    db.session.commit()
                    logging.info('Successfuly launched training the submission "{}" in {}'.format(submission, inst.public_ip_address))
            elif submission.state == 'tested':
                # Steps
                # 1) rsync latest log file
                # 2) kill instance
                # 3) compute scores
                # 4) update leaderboard

                # rsync log
                logging.info('Rsync the log of "{}"...'.format(submission))
                _rsync_log(user=ami_username, ip=inst.public_ip_address, ssh_key=ssh_key, submission=submission)
                # kill instance
                logging.info('Killing the instance {}...'.format(instance_id))
                ec2_resource.instances.filter(InstanceIds=[instance_id]).terminate()
                # compute score
                logging.info('Computing the score...')
                score_submission(submission)
                # update leaderboard
                logging.info('Updating the leaderboard...')
                update_leaderboards(submission.event.name)
                update_all_user_leaderboards(submission.event.name)
                compute_contributivity(event_name)
                compute_historical_contributivity(event_name)
                logging.info('Successfully finished training and testing the submission "{}"'.format(submission))
            elif submission.is_error:
                # Steps
                # 1) rsync the latest log file
                # 2) kill instance
                logging.info('Submission "{}" has finished training with an error.'.format(submission))
                # rsync log
                logging.info('Rsync the log of "{}"...'.format(submission))
                _rsync_log(user=ami_username, ip=inst.public_ip_address, ssh_key=ssh_key, submission=submission)
                # kill
                logging.info('Killing the instance {}...'.format(instance_id))
                ec2_resource.instances.filter(InstanceIds=[instance_id]).terminate()
            else:
                # the submission is training, so just rsync the log
                logging.info('Rsync the log of "{}"...'.format(submission))
                _rsync_log(user=ami_username, ip=inst.public_ip_address, ssh_key=ssh_key, submission=submission)
        db.session.close()
        time.sleep(sleep_time_secs)


def _add_postgresql_rule(ip, pgversion=9.3):
    rule = 'host all mrramp {ip}/32 md5 # amazon'.format(ip=ip)
    pghba = "/etc/postgresql/{pgversion}/main/pg_hba.conf".format(pgversion=pgversion)
    
    # if the rule already exists, dont do anything
    if rule + '\n' in open(pghba).readlines():
        return
    cmd = "sudo sh -c \"echo '{rule}'>>{pghba};\"".format(rule=rule, pghba=pghba)
    logging.debug(cmd)
    exit_status = call(cmd, shell=True)
    if exit_status != 0:
        logging.info('Exit status not zero : problem in adding postgresql access rule for "{}"'.format(submission))
        return
    cmd = "sudo -u postgres /usr/lib/postgresql/{pgversion}/bin/pg_ctl -D /etc/postgresql/{pgversion}/main reload".format(pgversion=pgversion)
    logging.debug(cmd)
    exist_status = call(cmd, shell=True)
    if exit_status != 0:
        logging.info('Exit status not zero : problem in reloading postgresql for "{}"'.format(submission))
        return


def _rsync_submission(user, ip, submission_path, ssh_key):
    values = {
        'user': user,
        'ip': ip,
        'submission' : submission_path,
        'cmd': "ssh -o 'StrictHostKeyChecking no' -i " + ssh_key
    }
    cmd = "rsync -e \"{cmd}\" -avzP {submission} {user}@{ip}:~/backend/submissions".format(**values)
    logging.debug(cmd)
    return call(cmd, shell=True)


def _is_screen_launched(user, ip, ssh_key):
    cmd = "ssh -o 'StrictHostKeyChecking no' -i {ssh_key} {user}@{ip} screen -ls|grep train|wc -l".format(user=user, ip=ip, ssh_key=ssh_key)
    logging.debug(cmd)
    nb = int(check_output(cmd, shell=True))
    return nb > 0

def _train_test(user, ip, ssh_key, submission, db_url):
    values = {
        'user': user,
        'ip': ip,
        'e': submission.event.name,
        't': submission.team.name,
        's': submission.name,
        'ssh_key': ssh_key,
        'db': db_url,
    }
    cmd = ("ssh -i {ssh_key} {user}@{ip} \""
           "screen -dm -S train sh -c '. ~/.profile;"
           "export DATABOARD_DB_URL={db};"
           "cd ~/backend;"
           "fab train_test_on_server:e={e},t={t},s={s} 2>&1|tee log'\"".format(**values))
    logging.debug(cmd)
    exit_status = call(cmd, shell=True)
    return  exit_status


def _rsync_log(user, ip, ssh_key, submission):
    values = {
        'user': user,
        'ip': ip,
        'cmd': "ssh -o 'StrictHostKeyChecking no' -i " + ssh_key,
        'submission' : str(submission.id) + '_' + submission.name,
    }
    cmd = "rsync -e \"{cmd}\" -avzP {user}@{ip}:~/backend/log logs/{submission}".format(**values)
    call(cmd, shell=True)


if __name__ == '__main__':
    for k in (logging.Logger.manager.loggerDict.keys()):
        if 'boto' in k:
            logging.getLogger(k).disabled = True
    profiles = {
        'prod':dict(
    	      event_name='pollenating_insects_3_JNI_2017',
              ami_image_id='ami-25d32f5d',
              ami_username='ubuntu',
              instance_type='g3.4xlarge', 
              key_name='ramp.studio', 
              ssh_key='/root/.ssh/amazon/rampstudio.pem',
              security_group='launch-wizard-74',
              sleep_time_secs=60, 
              timeout_secs=60*15,
              db_host='ramp.studio',
              pgversion='9.3',
        ),
        'test':dict(
              event_name='pollenating_insects_3',
              ami_image_id='ami-25d32f5d', 
              ami_username='ubuntu',
              instance_type='g3.4xlarge', 
              key_name='test_server', 
              ssh_key='/home/ubuntu/.ssh/amazon/test_server.pem',
              security_group='launch-wizard-74',
              sleep_time_secs=60, 
              timeout_secs=60*15,
              db_host='134.158.74.188',
              pgversion='9.5',
        )
    }
    parser = argparse.ArgumentParser(description='AWS Training loop')
    parser.add_argument('--profile', default='test', help='prod/test')
    args = parser.parse_args()
    config = profiles[args.profile]
    train_loop(**config)
