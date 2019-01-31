from .dispatcher import Dispatcher
from .local import CondaEnvWorker

from ._version import __version__

available_workers = {'conda': CondaEnvWorker}

__all__ = [
    'CondaEnvWorker',
    'Dispatcher',
    'available_workers',
    '__version__'
]
