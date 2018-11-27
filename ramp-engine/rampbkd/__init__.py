"""
 ____      _    __  __ ____    _                _                  _
|  _ \    / \  |  \/  |  _ \  | |__   __ _  ___| | _____ _ __   __| |
| |_) |  / _ \ | |\/| | |_) | | '_ \ / _` |/ __| |/ / _ \ '_ \ / _` |
|  _ <  / ___ \| |  | |  __/  | |_) | (_| | (__|   <  __/ | | | (_| |
|_| \_\/_/   \_\_|  |_|_|     |_.__/ \__,_|\___|_|\_\___|_| |_|\__,_|

Toolkit for interacting with the RAMP database

"""
from ._version import get_versions

__version__ = get_versions()['version']

del get_versions
