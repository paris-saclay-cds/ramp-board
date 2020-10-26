from ramp_engine.remote import _check_dask_workers_single_machine
import pytest


def test_dask_workers_single_machine():
    workers_nthreads = {
        'tcp://127.0.0.1:38901': 4,
        'tcp://127.0.0.1:43229': 4,
        'tcp://127.0.0.1:46663': 4
    }

    assert _check_dask_workers_single_machine(workers_nthreads.keys()) is True

    workers_nthreads = {
        'tcp://127.0.0.1:38901': 4,
        'tcp://127.0.0.2:43229': 4
    }

    msg = 'dask workers should .* on 1 machine, found 2:'
    with pytest.raises(ValueError, match=msg):
        _check_dask_workers_single_machine(workers_nthreads.keys())

    msg = 'dask workers should .* on 1 machine, found 0:'
    with pytest.raises(ValueError, match=msg):
        _check_dask_workers_single_machine('some invalid string')
