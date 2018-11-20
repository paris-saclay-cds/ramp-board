import os
import subprocess
import shutil

import pytest

from rampbkd.local import LocalEngine


def test_local_engine():
    module_path = os.path.dirname(__file__)
    ramp_kit_dir = os.path.join(module_path, 'kits', 'iris')
    ramp_data_dir = ramp_kit_dir
    engine = LocalEngine(conda_env='ramp',
                         ramp_data_dir=ramp_data_dir,
                         ramp_kit_dir=ramp_kit_dir)
    try:
        engine.setup()
        engine.launch_submission()
        # engine.launch_submission()
        while engine.collect_submission() is None:
            pass
        print(engine.collect_submission())
        engine.teardown()
    finally:
        output_training = os.path.join(ramp_kit_dir, 'submissions',
                                       'starting_kit', 'training_output')
        if os.path.exists(output_training):
            shutil.rmtree(output_training)


def test_local_engine_unknown_env():
    module_path = os.path.dirname(__file__)
    ramp_kit_dir = os.path.join(module_path, 'kits', 'iris')
    ramp_data_dir = ramp_kit_dir
    engine = LocalEngine(conda_env='xxx',
                         ramp_data_dir=ramp_data_dir,
                         ramp_kit_dir=ramp_kit_dir)
    msg_err = "The specified conda environment xxx does not exist."
    with pytest.raises(ValueError, match=msg_err):
        engine.setup()
