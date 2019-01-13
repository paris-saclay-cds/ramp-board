"""The :mod:`frontend.testing` module contains all functions used to easily
test the frontend."""


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
