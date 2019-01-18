import shutil

import pytest

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.model import Extension
from rampdb.model import Model
from rampdb.model import SubmissionFileType
from rampdb.model import SubmissionFileTypeExtension

from rampdb.utils import setup_db
from rampdb.utils import session_scope

from rampdb.testing import create_test_db

from rampdb.tools.database import add_extension
from rampdb.tools.database import add_submission_file_type
from rampdb.tools.database import add_submission_file_type_extension
from rampdb.tools.database import get_extension
from rampdb.tools.database import get_submission_file_type
from rampdb.tools.database import get_submission_file_type_extension


@pytest.fixture(scope='module')
def database_config():
    return read_config(path_config_example(), filter_section='sqlalchemy')


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture
def session_scope_function(config):
    try:
        create_test_db(config)
        with session_scope(config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        db, _ = setup_db(config['sqlalchemy'])
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
