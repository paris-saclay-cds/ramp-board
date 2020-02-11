import os
import shutil
import tempfile

from ramp_utils.datasets import OSFRemoteMetaData
from ramp_utils.datasets import fetch_from_osf


def test_fetch_from_osf():
    archive = [OSFRemoteMetaData(filename="iris.csv", id="7vyah", revision=1)]
    tmp_dir = tempfile.mkdtemp()

    try:
        fetch_from_osf(path_data=tmp_dir, metadata=archive)
        assert os.path.exists(os.path.join(tmp_dir, archive[0].filename))
    finally:
        shutil.rmtree(tmp_dir)
