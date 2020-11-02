from .aws import AWSWorker
from .dispatcher import Dispatcher
from .local import CondaEnvWorker
from .remote import RemoteWorker

from ._version import __version__

available_workers = {'conda': CondaEnvWorker,
                     'aws': AWSWorker,
                     'remote': RemoteWorker}

__all__ = [
    'AWSWorker',
    'CondaEnvWorker',
    'Dispatcher',
    'available_workers',
    '__version__'
]
