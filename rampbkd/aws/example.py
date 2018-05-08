from rampbkd.config import read_backend_config
import logging
from api import launch_ec2_instance_and_train
from api import train_on_existing_ec2_instance
from api import launch_ec2_instances

if __name__ == '__main__':
    for k in logging.Logger.manager.loggerDict.keys():
        logging.getLogger(k).disabled = True
    logging.getLogger('api').disabled = False
    conf = read_backend_config('config.yml')
    #launch_ec2_instances(conf, nb=1)
    submission_id = 1
    instance_id = 'i-0deb25f46ae28f06f'
    train_on_existing_ec2_instance(conf, instance_id, submission_id)
