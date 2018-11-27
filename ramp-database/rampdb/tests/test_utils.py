import os
from rampdb.utils import import_module_from_source


def test_import_module_from_source():
        module_path = os.path.dirname(__file__)
        # import the local_module.py which consist of a single function.
        mod = utils.import_module_from_source(
                os.path.join(module_path, 'local_module.py'), 'mod'
        )
        assert hasattr(mod, 'func_local_module')
