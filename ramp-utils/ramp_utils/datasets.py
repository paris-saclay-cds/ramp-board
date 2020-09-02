from collections import namedtuple
import hashlib
import json
import os
from urllib import request

OSFRemoteMetaData = namedtuple(
    "OSFRemoteMetaData",
    ["filename", "id", "revision"]
)


def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def fetch_from_osf(path_data, metadata, token=None):
    """Fetch files from OSF storage.

    If the file is inside a private project, you will need to specify your
    token.

    Parameters
    ----------
    path_data : str
        The path where to store the data which will be downloaded.
    metadata : list of OSFRemoteMetaData
        A list of OSF metadata for each file in the dataset.
    token : str
        The OSF token to be able to make the token in case the file is in a
        private project.
    """
    if not os.path.exists(path_data):
        os.makedirs(path_data)
    for file_info in metadata:
        file_info_url = (
            f"https://api.osf.io/v2/files/{file_info.id}/"
        )
        req = request.Request(file_info_url)
        if token is not None:
            req.add_header("Authorization", f"Bearer {token}")
        response = request.urlopen(req)
        info = json.loads(response.read())
        original_checksum = \
            info["data"]["attributes"]["extra"]["hashes"]["sha256"]
        filename = os.path.join(path_data, file_info.filename)
        if os.path.exists(filename):
            if _sha256(filename) == original_checksum:
                # already the same file, skip the downloading
                continue

        osf_url = (
            f"https://osf.io/download/"
            f"{file_info.id}/?revision={file_info.revision}"
        )
        req = request.Request(osf_url)
        if token is not None:
            req.add_header("Authorization", f"Bearer {token}")
        response = request.urlopen(req)
        if response.getcode() != 200:
            raise RuntimeError(response.read())
        with open(filename, "wb") as fid:
            fid.write(response.read())
        assert _sha256(filename) == original_checksum, \
            f"{filename} was corrupted during download"
