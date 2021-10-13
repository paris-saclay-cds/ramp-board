import os
from pathlib import Path

from ramp_engine.remote import _check_dask_workers_single_machine
from ramp_engine.remote import _serialize_folder, _deserialize_folder
import pytest


def test_dask_workers_single_machine():
    workers_nthreads = {
        "tcp://127.0.0.1:38901": 4,
        "tcp://127.0.0.1:43229": 4,
        "tcp://127.0.0.1:46663": 4,
    }

    assert _check_dask_workers_single_machine(workers_nthreads.keys()) is True

    workers_nthreads = {"tcp://127.0.0.1:38901": 4, "tcp://127.0.0.2:43229": 4}

    msg = "dask workers should .* on 1 machine, found 2:"
    with pytest.raises(ValueError, match=msg):
        _check_dask_workers_single_machine(workers_nthreads.keys())

    msg = "dask workers should .* on 1 machine, found 0:"
    with pytest.raises(ValueError, match=msg):
        _check_dask_workers_single_machine("some invalid string")


def test_serialize_deserialize_folder(tmpdir):
    with pytest.raises(FileNotFoundError):
        _serialize_folder("some_invalid_path")

    base_dir = Path(tmpdir)
    src_dir = base_dir / "src"
    src_dir.mkdir()
    with open(src_dir / "1.txt", "wt") as fh:
        fh.write("a")
    (src_dir / "dir2").mkdir()

    stream = _serialize_folder(src_dir)
    assert isinstance(stream, bytes)

    dest_dir = base_dir / "dest"
    _deserialize_folder(stream, dest_dir)

    assert sorted(os.listdir(src_dir)) == sorted(os.listdir(dest_dir))

    # create some other file objects in the destination dir and try
    # to deserialize a second time
    (dest_dir / "dir3").mkdir()
    _deserialize_folder(stream, dest_dir)
    assert sorted(os.listdir(src_dir)) == sorted(os.listdir(dest_dir))
