# All the following functions may be used with dask, so
# they should include imports inside functions.
from typing import Dict


def _conda_info_envs() -> Dict:
    import json
    import subprocess

    proc = subprocess.Popen(
        ["conda", "info", "--envs", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = proc.communicate()
    conda_info = json.loads(stdout)
    return conda_info


def _get_conda_env_path(conda_info: Dict, env_name: str, worker=None) -> str:
    """Get path for a python executable of a conda env"""
    import os

    if env_name == "base":
        return os.path.join(conda_info["envs"][0], "bin")
    else:
        envs_path = conda_info["envs"][1:]
        if not envs_path:
            worker.status = "error"
            raise ValueError(
                "Only the conda base environment exist. You "
                'need to create the "{}" conda environment '
                "to use it.".format(env_name)
            )
        for env in envs_path:
            if env_name == os.path.split(env)[-1]:
                return os.path.join(env, "bin")
        worker.status = "error"
        raise ValueError(
            f"The specified conda environment {env_name} "
            f"does not exist. You need to create it."
        )


def _conda_ramp_test_submission(
    config: Dict,
    submission: str,
    cmd_ramp: str,
    log_dir: str,
    wait: bool = False,
):
    import os
    import subprocess

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = open(os.path.join(log_dir, "log"), "wb+")
    proc = subprocess.Popen(
        [
            # Make sure the process has larger niceness than
            # the server.
            "nice",
            "-5",
            cmd_ramp,
            "--submission",
            submission,
            "--ramp-kit-dir",
            config["kit_dir"],
            "--ramp-data-dir",
            config["data_dir"],
            "--ramp-submission-dir",
            config["submissions_dir"],
            "--save-output",
            "--ignore-warning",
        ],
        stdout=log_file,
        stderr=log_file,
    )
    if wait:
        # Wait until process completes. In particular, we need this call
        # to be blocking with dask, since dask has another mechanism for
        # cancelling tasks.
        proc.communicate()
        return proc.returncode
    else:
        return proc, log_file
