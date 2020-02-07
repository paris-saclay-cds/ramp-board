import os
import subprocess
import zipfile

from ramp_database.testing import _delete_line_from_file
from ramp_database.testing import setup_files_extension_type
from ramp_database.testing import setup_ramp_kit_ramp_data
from ramp_database.tools.event import add_event
from ramp_database.tools.event import add_problem
from ramp_database.tools.event import get_problem
from ramp_database.utils import session_scope

from .config_parser import read_config
from .ramp import generate_ramp_config


def deploy_ramp_event(config, event_config, setup_ramp_repo=True, force=False):
    """Deploy a RAMP event using a configuration file.

    This utility is in charge of creating the kit and data repository for a
    given RAMP event. It will also setup the database.

    Parameters
    ----------
    config : str
        The path to the YAML file containing the database information.
    event_config : str
        The path to the YAML file containing the RAMP infomation.
    setup_ramp_repo : bool, default is True
        Whether or not to setup the RAMP kit and data repositories.
    force : bool, default is False
        Whether or not to potentially overwrite the repositories, problem and
        event in the database.
    """
    database_config = read_config(config, filter_section='sqlalchemy')
    ramp_config = generate_ramp_config(event_config, config)

    with session_scope(database_config) as session:
        setup_files_extension_type(session)
        if setup_ramp_repo:
            setup_ramp_kit_ramp_data(
                ramp_config, ramp_config['problem_name'], force
            )
        else:
            # we do not clone the repository but we need to convert the
            # notebook to html
            current_directory = os.getcwd()
            problem_kit_path = ramp_config['ramp_kit_dir']
            os.chdir(problem_kit_path)
            subprocess.check_output(["jupyter", "nbconvert", "--to", "html",
                                     "{}_starting_kit.ipynb"
                                     .format(ramp_config['problem_name'])])
            # delete this line since it trigger in the front-end
            # (try to open execute "custom.css".)
            _delete_line_from_file(
                "{}_starting_kit.html".format(ramp_config['problem_name']),
                '<link rel="stylesheet" href="custom.css">\n'
            )
            os.chdir(current_directory)
        # check if the repository exists
        problem = get_problem(session, ramp_config['problem_name'])
        if problem is None:
            add_problem(session, ramp_config['problem_name'],
                        ramp_config['ramp_kit_dir'],
                        ramp_config['ramp_data_dir'])
        else:
            if ((ramp_config['ramp_kit_dir'] != problem.path_ramp_kit or
                 ramp_config['ramp_data_dir'] != problem.path_ramp_data) and
                    not force):
                raise ValueError(
                    'The RAMP problem already exists in the database. The path'
                    ' to the kit or to the data is different. You need to set'
                    ' "force=True" if you want to overwrite these parameters.'
                )
            if setup_ramp_repo:
                setup_ramp_kit_ramp_data(
                    ramp_config, ramp_config['problem_name'], force
                )
            if force:
                add_problem(session, ramp_config['problem_name'],
                            ramp_config['ramp_kit_dir'],
                            ramp_config['ramp_data_dir'],
                            force)

        if not os.path.exists(ramp_config['ramp_submissions_dir']):
            os.makedirs(ramp_config['ramp_submissions_dir'])

        # create a folder in the ramp-kit directory to store the archive
        archive_dir = os.path.abspath(os.path.join(
            ramp_config['ramp_kit_dir'], 'events_archived'
        ))
        if not os.path.exists(archive_dir):
            os.makedirs(archive_dir)
        zip_filename = os.path.join(
            archive_dir, ramp_config["event_name"] + ".zip"
        )
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(ramp_config['ramp_kit_dir']):
                if archive_dir not in os.path.abspath(root):
                    for f in files:
                        path_file = os.path.join(root, f)
                        zipf.write(
                            path_file,
                            os.path.relpath(
                                path_file, start=ramp_config["ramp_kit_dir"]
                            )
                        )

        add_event(session, ramp_config['problem_name'],
                  ramp_config['event_name'],
                  ramp_config['event_title'],
                  ramp_config['sandbox_name'],
                  ramp_config['ramp_submissions_dir'],
                  ramp_config['event_is_public'],
                  force)
