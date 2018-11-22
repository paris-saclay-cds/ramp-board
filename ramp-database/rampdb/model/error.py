__all__ = [
    'NameClashError',
    'MergeTeamError',
    'DuplicateSubmissionError',
    'TooEarlySubmissionError',
    'MissingExtensionError',
    'MissingSubmissionFileError',
]


class NameClashError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class MergeTeamError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class DuplicateSubmissionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class TooEarlySubmissionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class MissingSubmissionFileError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class MissingExtensionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
