import logging

from ..model import Extension
from ..model import SubmissionFileType
from ..model import SubmissionFileTypeExtension

from ._query import select_extension_by_name
from ._query import select_submission_file_type_by_name
from ._query import select_submission_type_extension_by_name

logger = logging.getLogger('RAMP-DATABASE')


# Add functions: add entries in the database
def add_extension(session, name):
    """Adding a new extension, e.g., 'py'.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str
        The name of the extension to add if it does not exist.
    """
    extension = select_extension_by_name(session, name)
    if extension is None:
        extension = Extension(name=name)
        logger.info('Adding {}'.format(extension))
        session.add(extension)
        session.commit()


def add_submission_file_type(session, name, is_editable, max_size):
    """Add a new submission file type, e.g., ('code', True, 10 ** 5).

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str
        The name of file type.
    is_editable: bool
        If the file type is editable.
    max_size : int
        The maximum size of the file.

    Notes
    -----
    Should be preceded by adding extensions.
    """
    submission_file_type = select_submission_file_type_by_name(session, name)
    if submission_file_type is None:
        submission_file_type = SubmissionFileType(
            name=name, is_editable=is_editable, max_size=max_size)
        logger.info('Adding {}'.format(submission_file_type))
        session.add(submission_file_type)
        session.commit()


def add_submission_file_type_extension(session, type_name, extension_name):
    """Adding a new submission file type extension, e.g., ('code', 'py').

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    type_name : str
        The file type.
    extension_name : str
        The extension name.

    Notes
    -----
    Should be preceded by adding submission file types and extensions.
    """
    type_extension = select_submission_type_extension_by_name(
        session, type_name, extension_name
    )
    if type_extension is None:
        submission_file_type = select_submission_file_type_by_name(session,
                                                                   type_name)
        extension = select_extension_by_name(session, extension_name)
        type_extension = SubmissionFileTypeExtension(
            type=submission_file_type,
            extension=extension
        )
        logger.info('Adding {}'.format(type_extension))
        session.add(type_extension)
        session.commit()


# Get functions: get information from the database
def get_extension(session, extension_name):
    """Get extension from the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    extension_name : str or None
        The name of the extension to query. If None, all the extensions will be
        queried.

    Returns
    -------
    extension : :class:`ramp_database.model.Extension` or list of \
:class:`ramp_database.model.Extension`
        The queried extension.
    """
    return select_extension_by_name(session, extension_name)


def get_submission_file_type(session, type_name):
    """Get submission file type from the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    type_name : str or None
        The name of the type to query. If None, all the file type will be
        queried.

    Returns
    -------
    extension : :class:`ramp_database.model.SubmissionFileType` or list of \
:class:`ramp_database.model.SubmissionFileType`
        The queried submission file type.
    """
    return select_submission_file_type_by_name(session, type_name)


def get_submission_file_type_extension(session, type_name, extension_name):
    """Get submission file type extension from the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    type_name : str or None
        The name of the type to query. If None, all the file type will be
        queried.
    extension_name : str or None
        The name of the extension to query. If None, all the extension will be
        queried.

    Returns
    -------
    extension : :class:`ramp_database.model.SubmissionFileTypeExtension` or \
list of :class:`ramp_database.model.SubmissionFileTypeExtension`
        The queried submission file type.
    """
    return select_submission_type_extension_by_name(
        session, type_name, extension_name
    )
