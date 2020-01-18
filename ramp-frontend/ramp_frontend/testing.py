"""The :mod:`ramp_frontend.testing` module contains all functions used to
easily test the frontend."""

import errno
import socket
from contextlib import contextmanager

import pytest


def login(client, username, password):
    """Simulate a log-in from a user.

    After the log-in request, the client will be redirect to the expected page.

    Parameters
    ----------
    client : :class:`flask.testing.FlaskClient`
        The testing client used for unit testing.
    username : str
        The user's name.
    password : str
        The user's password.

    Returns
    -------
    response : :class:`flask.wrappers.Response`
        The response of the client.
    """
    return client.post('/login', data=dict(
        user_name=username,
        password=password
    ), follow_redirects=True)


def logout(client):
    """Simulate a log-out.

    After the log-out request, the client will be redirected to the expected
    page.

    Parameters
    ----------
    client : :class:`flask.testing.FlaskClient`
        The testing client used for unit testing.

    Returns
    -------
    response : :class:`flask.wrappers.Response`
        The response of the client.
    """
    return client.get('/logout', follow_redirects=True)


@contextmanager
def login_scope(client, username, password):
    """Context manager to log-in during the ``with`` scope.

    Parameters
    ----------
    client : :class:`flask.testing.FlaskClient`
        The testing client used for unit testing.
    username : str
        The user's name.
    password : str
        The user's password.

    Returns
    -------
    client : :class:`flask.testing.FlaskClient`
        A client which is logged-in for the duration of the ``with`` scope.
    """
    login(client, username, password)
    yield client
    logout(client)


def _bind_smtp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port_in_use = False

    try:
        s.bind(("127.0.0.1", 8025))
    except socket.error as e:
        if e.errno == errno.EADDRINUSE:
            port_in_use = True

    s.close()
    return port_in_use


_fail_no_smtp_server = pytest.mark.skipif(
    not _bind_smtp_port(), reason="No smtp server in use."
)
