# All the following functions may be used with dask, so
# they should include imports inside functions.
from typing import Dict


def _conda_info_envs() -> Dict:
    import json
    import subprocess

    proc = subprocess.Popen(
        ["conda", "info", "--envs", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, _ = proc.communicate()
    conda_info = json.loads(stdout)
    return conda_info


def _get_conda_env_path(conda_info: Dict, env_name: str, worker=None) -> str:
    """Get path for a python executable of a conda env
    """
    import os

    if env_name == 'base':
        return os.path.join(conda_info['envs'][0], 'bin')
    else:
        envs_path = conda_info['envs'][1:]
        if not envs_path:
            worker.status = 'error'
            raise ValueError('Only the conda base environment exist. You '
                             'need to create the "{}" conda environment '
                             'to use it.'.format(env_name))
        is_env_found = False
        for env in envs_path:
            if env_name == os.path.split(env)[-1]:
                is_env_found = True
                return os.path.join(env, 'bin')
                break
        if not is_env_found:
            worker.status = 'error'
            raise ValueError(f'The specified conda environment {env_name} '
                             f'does not exist. You need to create it.')
