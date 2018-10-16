import os

# There is a dangerous rm -rf test_deployment_path operation
# in local test. To avoid deleting the deployment directory
# on the production server accidentally, the local test
# deletes test_deployment_path. It's important to set these
# to different directories on the production server.
local_test_deployment_path = '/tmp/databoard'  # edit this!
deployment_path = '/tmp/databoard'  # edit this!
ramp_kits_path = os.path.join(deployment_path, 'ramp-kits')
ramp_data_path = os.path.join(deployment_path, 'ramp-data')
submissions_d_name = 'submissions'
submissions_path = os.path.join(deployment_path, submissions_d_name)
sandbox_d_name = 'starting_kit'

is_parallelize = True  # make it False if parallel training is not working

server_port = 8080
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 587
MAIL_USERNAME = os.environ.get('DATABOARD_MAIL_USERNAME')
MAIL_PASSWORD = os.environ.get('DATABOARD_MAIL_PASSWORD')
MAIL_DEFAULT_SENDER = ('RAMP admin', os.environ.get('DATABOARD_MAIL_SENDER'))
MAIL_RECIPIENTS = ''  # notification_recipients
ADMIN_MAILS = os.environ.get('DATABOARD_ADMIN_MAILS')
DATABASE_QUERY_TIMEOUT = 0.5  # slow database query threshold (in seconds)
databoard_path = os.environ.get('DATABOARD_PATH', '/tmp')


class Config(object):
    # abs max upload file size, to throw 413, before saving it
    MAX_CONTENT_LENGTH = 1024 * 1024 * 1024
    LOG_FILENAME = None  # if None, output to screen
    WTF_CSRF_ENABLED = True
    SECRET_KEY = 'eroigudsfojbn;lk'
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///' + db_f_name
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABOARD_DB_URL')
    TESTING = False


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABOARD_DB_URL_TEST')
