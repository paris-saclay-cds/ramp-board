import requests
import zlib
import base64


def read_compress(file_name):
    with open(file_name, 'r') as ff:
        df = ff.read()
    return base64.b64encode(zlib.compress(df))


def url_post(url1, url2, username, password, data):
    url = url1 + url2
    url = url[0:9] + url[9::].replace('//', '/')
    return requests.post(url, auth=(username, password), json=data)


def url_get(url1, url2, username, password, r_id=None):
    url = url1 + url2
    url = url[0:9] + url[9::].replace('//', '/')
    if r_id:
        url = url + str(r_id) + '/'
    return requests.get(url, auth=(username, password))


def post_data(host_url, username, password,
              data_name, target_column, workflow_elements, data_file,
              extra_files=None):
    """
    To post data to the datarun api.\
    Data are compressed (with zlib) and base64-encoded before being posted.

    :param host_url: api host url, such as http://127.0.0.1:8000/ (localhost)
    :param username: username to be used for authentication
    :param password: password to be used for authentication
    :param data_name: name of the raw dataset
    :param target_column: name of the target column
    :param workflow_elements: workflow elements associated with this dataset,\
    e.g., feature_extractor, classifier
    :param data_file: name with absolute path of the dataset file
    :param extra_files: list of names with absolute path of extra files\
        (such as a specific.py)

    :type host_url: string
    :type username: string
    :type password: string
    :type data_name: string
    :type target_column: string
    :type workflow_elements: string
    :type data_file: string
    :type extra_files: list of string
    """

    data = {'name': data_name, 'target_column': target_column,
            'workflow_elements': workflow_elements}
    df = read_compress(data_file)
    short_name = data_file.split('/')[-1]
    data['files'] = {short_name: df}
    if extra_files:
        for filename in extra_files:
            df = read_compress(filename)
            short_name = filename.split('/')[-1]
            data['files'][short_name] = df

    return url_post(host_url, '/runapp/rawdata/', username, password,
                    data)


def post_split(host_url, username, password,
               held_out_test, raw_data_id, random_state=42):
    """
    To split data between train and test on datarun

    :param host_url: api host url, such as http://127.0.0.1:8000/ (localhost)
    :param username: username to be used for authentication
    :param password: password to be used for authentication
    :param held_out_test: ratio of data for the test set
    :param raw_data_id: id of the raw dataset on datarun
    :param random_state: random state to be used in the shuffle split

    :type host_url: string
    :type username: string
    :type password: string
    :type held_out_test: float (between 0 and 1)
    :type raw_data_id: integer
    :type random_state: integer
    """
    data = {'random_state': random_state, 'held_out_test': held_out_test,
            'raw_data_id': raw_data_id}
    return url_post(host_url, '/runapp/rawdata/split/', username, password,
                    data)


def custom_post_split(host_url, username, password, raw_data_id):
    """
    To split data between train and test on datarun using a specific
    prepare_data function sent by databoard

    :param host_url: api host url, such as http://127.0.0.1:8000/ (localhost)
    :param username: username to be used for authentication
    :param password: password to be used for authentication
    :param raw_data_id: id of the raw dataset on datarun

    :type host_url: string
    :type username: string
    :type password: string
    :type raw_data_id: integer
    """
    data = {'raw_data_id': raw_data_id}
    return url_post(host_url, '/runapp/rawdata/customsplit/', username,
                    password, data)


def post_submission_fold(host_url, username, password,
                         sub_id, sub_fold_id, train_is, test_is,
                         priority='L',
                         raw_data_id=None, list_submission_files=None,
                         force=None):
    """
    To post submission on cv fold and submission (if not already posted).\
    Submission files are compressed (with zlib) and base64-encoded before being\
    posted.

    :param host_url: api host url, such as http://127.0.0.1:8000/ (localhost)
    :param username: username to be used for authentication
    :param password: password to be used for authentication
    :param sub_id: id of the submission on databoard
    :param sub_fold_id: id of the submission on cv fold on databoard
    :param train_is: train indices for the cv fold
    :param test_is: test indices for the cv fold
    :param priority: priority level to train test the model: L for low\
    and H for high
    :param raw_data_id: id of the associated data, when submitting a submission
    :param list_submission_files: list of files of the submission,\
        when submitting a submission
    :param force: to force the submission even if ids already exist\
        force can be 'submission, submission_fold' to resubmit both\
        or 'submission, submission_fold' to resubmit only the submission\
        on cv fold. None by default.

    :type host_url: string
    :type username: string
    :type password: string
    :type sub_id: integer
    :type sub_fold_id: integer
    :type train_is: numpy array
    :type test_is: numpy array
    :type priority: string
    :type raw_data_id: integer
    :type list_submission_files: list
    :type force: string
    """
    # Compress train and test indices
    train_is = base64.b64encode(zlib.compress(train_is.tostring()))
    test_is = base64.b64encode(zlib.compress(test_is.tostring()))
    data = {'databoard_sf_id': sub_fold_id, 'databoard_s_id': sub_id,
            'train_is': train_is, 'test_is': test_is}
    # To force the submission even if ids already exist
    if force:
        data['force'] = force
    # If the submission does not exist, post info needed to save it in the db
    if raw_data_id and list_submission_files:
        data['raw_data'] = raw_data_id
        data['files'] = {}
        for ff in list_submission_files:
            data['files'][ff.split('/')[-1]] = read_compress(ff)
    return url_post(host_url, '/runapp/submissionfold/', username, password,
                    data)


def get_prediction_list(host_url, username, password,
                        list_submission_fold_id):
    """
    Get predictions given a list of submission on cv fold ids

    :param host_url: api host url, such as http://127.0.0.1:8000/ (localhost)
    :param username: username to be used for authentication
    :param password: password to be used for authentication
    :param list_submission_fold_id: list of submission on cv fold ids from \
    which we want the predictions

    :type host_url: string
    :type username: string
    :type password: string
    :type list_submission_fold_id: list
    """
    data = {'list_submission_fold': list_submission_fold_id}
    return url_post(host_url, '/runapp/testpredictions/list/', username,
                    password, data)


def get_prediction_new(host_url, username, password,
                       raw_data_id):
    """
    Get all new predictions given a raw data id

    :param host_url: api host url, such as http://127.0.0.1:8000/ (localhost)
    :param username: username to be used for authentication
    :param password: password to be used for authentication
    :param raw_data_id: id of a data set from which we want new predictions

    :type host_url: string
    :type username: string
    :type password: string
    :type raw_data_id: integer
    """
    data = {'raw_data_id': raw_data_id}
    return url_post(host_url, '/runapp/testpredictions/new/', username,
                    password, data)


def get_raw_data(host_url, username, password):
    """
    Get all raw data sets

    :param host_url: api host url, such as http://127.0.0.1:8000/ (localhost)
    :param username: username to be used for authentication
    :param password: password to be used for authentication

    :type host_url: string
    :type username: string
    :type password: string
    """
    return url_get(host_url, '/runapp/rawdata/', username, password)


def get_submission_fold_light(host_url, username, password):
    """
    Get all submissions on cv fold\
    only main info: id, associated submission id, state, and new

    :param host_url: api host url, such as http://127.0.0.1:8000/ (localhost)
    :param username: username to be used for authentication
    :param password: password to be used for authentication

    :type host_url: string
    :type username: string
    :type password: string
    """
    return url_get(host_url, '/runapp/submissionfold-light/', username,
                   password)


def get_submission_fold(host_url, username, password):
    """
    Get all submission on cv fold (all attributes)

    :param host_url: api host url, such as http://127.0.0.1:8000/ (localhost)
    :param username: username to be used for authentication
    :param password: password to be used for authentication

    :type host_url: string
    :type username: string
    :type password: string
    """
    return url_get(host_url, '/runapp/submissionfold/', username, password)


def get_submission_fold_detail(host_url, username, password,
                               submission_fold_id):
    """
    Get details about a submission on cv fold given its id

    :param host_url: api host url, such as http://127.0.0.1:8000/ (localhost)
    :param username: username to be used for authentication
    :param password: password to be used for authentication
    :param submission_fold_id: id of the submission on cv fold

    :type host_url: string
    :type username: string
    :type password: string
    :param submission_fold_id: integer
    """
    return url_get(host_url, '/runapp/submissionfold/', username, password,
                   r_id=submission_fold_id)
