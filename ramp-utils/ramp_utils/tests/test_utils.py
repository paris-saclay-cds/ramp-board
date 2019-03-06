import six

from ramp_utils.utils import check_password
from ramp_utils.utils import encode_string
from ramp_utils.utils import hash_password
from ramp_utils.utils import import_module_from_source


def test_import_module_from_source():
    module_path = os.path.dirname(__file__)
    # import the local_module.py which consist of a single function.
    mod = import_module_from_source(
        os.path.join(module_path, 'local_module.py'), 'mod'
    )
    assert hasattr(mod, 'func_local_module')


def test_check_password():
    password = "hjst3789ep;ocikaqjw"
    hashed_password = hash_password(password)
    assert check_password(password, hashed_password)
    assert not check_password("hjst3789ep;ocikaqji", hashed_password)


def test_encode_string():
    if six.PY3:
        string = encode_string('a string')
        assert isinstance(string, bytes)
        string = encode_string(b'a string')
        assert isinstance(string, bytes)
    else:
        string = encode_string('a string')
        assert isinstance(string, bytes)
