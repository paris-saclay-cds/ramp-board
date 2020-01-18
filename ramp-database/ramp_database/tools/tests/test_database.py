import shutil

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import Extension
from ramp_database.model import Model
from ramp_database.model import SubmissionFileType
from ramp_database.model import SubmissionFileTypeExtension

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope

from ramp_database.testing import create_test_db

from ramp_database.tools.database import add_extension
from ramp_database.tools.database import add_submission_file_type
from ramp_database.tools.database import add_submission_file_type_extension
from ramp_database.tools.database import get_extension
from ramp_database.tools.database import get_submission_file_type
from ramp_database.tools.database import get_submission_file_type_extension


@pytest.fixture
def session_scope_function(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_test_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_check_extension(session_scope_function):
    extension_name = 'cpp'
    add_extension(session_scope_function, extension_name)
    extension = get_extension(session_scope_function, extension_name)
    assert extension.name == extension_name
    assert isinstance(extension, Extension)
    extension = get_extension(session_scope_function, None)
    assert len(extension) == 5
    assert isinstance(extension, list)


def test_check_submission_file_type(session_scope_function):
    name = 'my own type'
    is_editable = False
    max_size = 10 ** 5
    add_submission_file_type(session_scope_function, name, is_editable,
                             max_size)
    sub_file_type = get_submission_file_type(session_scope_function, name)
    assert sub_file_type.name == name
    assert sub_file_type.is_editable is is_editable
    assert sub_file_type.max_size == max_size
    assert isinstance(sub_file_type, SubmissionFileType)
    sub_file_type = get_submission_file_type(session_scope_function, None)
    assert len(sub_file_type) == 4
    assert isinstance(sub_file_type, list)


def test_check_submission_file_type_extension(session_scope_function):
    # create a new type and extension
    extension_name = 'cpp'
    add_extension(session_scope_function, extension_name)
    type_name = 'my own type'
    is_editable = False
    max_size = 10 ** 5
    add_submission_file_type(session_scope_function, type_name, is_editable,
                             max_size)

    add_submission_file_type_extension(session_scope_function, type_name,
                                       extension_name)
    sub_ext_type = get_submission_file_type_extension(session_scope_function,
                                                      type_name,
                                                      extension_name)
    assert sub_ext_type.file_type == type_name
    assert sub_ext_type.extension_name == extension_name
    assert isinstance(sub_ext_type, SubmissionFileTypeExtension)

    sub_ext_type = get_submission_file_type_extension(session_scope_function,
                                                      None, None)
    assert len(sub_ext_type) == 5
    assert isinstance(sub_ext_type, list)
