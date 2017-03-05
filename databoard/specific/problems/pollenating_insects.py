import os
import sys
import pandas as pd
import subprocess

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import databoard.multiclass_prediction as prediction
from databoard.config import problems_path
from databoard.specific.workflows.batch_classifier_workflow import ArrayContainer

workflow_name = 'batch_classifier_workflow'

random_state = 42
test_ratio = 0.5
# Due to memory constraints, images are not loaded from disk into memory in one shot.
# Rather, only one chunk of size `chunk_size` is loaded from the disk each time.
# The size of the chunk is not necessarily the same than `batch_size`, the size
# of the mini-batch used to train neural nets. The chunk is typically bigger than batch_size.
# In parallel to training ( in another thread), the next `chunk_size` images are loaded
# into memory (it is parallelized over CPUs, the number of jobs is controlled by 'n_img_load_jobs') 
# and put into a queue. The neural net retrieves each time `batch_size` elements from the queue
# and updates its parameters using each mini-batch.
# Note that `batch_size` is controlled by the user, it is specified in `Classifier`
# whereas `chunk_size` is constrolled by the backend.
chunk_size = 1024
n_img_load_jobs = 8
# Due to memory constraints, it is not possible to predict the whole test data at 
# once, so the predictions are also done using mini-batches.
# The same `chunk_size` is used at test time. The size of the mini-batches in
# test time is controlled by `test_batch_size`, and it is set by the backend, not
# the user. Because there is no backprop in test time, `test_batch_size` can typically
# be larger than the one used in training.
test_batch_size = 256

problem_name = 'pollenating_insects'  # should be the same as the file name
# folder containing images to train or test on
img_folder = os.path.join(
    problems_path, problem_name, 'data', 'raw', 'imgs'
)
train_img_folder = os.path.join(
    problems_path, problem_name, 'data', 'raw', 'train_imgs'
)
full_filename_raw = os.path.join(
    problems_path, problem_name, 'data', 'raw', 'spipoll.txt'
)
full_filename = os.path.join(
    problems_path, problem_name, 'data', 'full.csv'
)
train_filename = os.path.join(
    problems_path, problem_name, 'data', 'train.csv')
test_filename = os.path.join(
    problems_path, problem_name, 'data', 'test.csv')
# These attributes, `attrs`, are assigned into the `X_array` to give
# to the batch_classifier_workflow some global variables
# which are necessary for training and testing.
attrs = {
    'chunk_size': chunk_size,
    'n_jobs': n_img_load_jobs,
    'test_batch_size': test_batch_size,
    'folder': img_folder,
    'n_classes': 18
}

def prepare_data():

    #1) split data into training and test
    _split(test_ratio=test_ratio, random_state=random_state)

    #2) download the images from urls
    _download()
    
    #3) Put links of training images in a  folder, the folder will be 
    #   given to users.
    X, y = get_train_data()
    _silent_mkdir(train_img_folder)
    for id_ in X:
        source = os.path.abspath(os.path.join(img_folder, _get_image_filename(id_)))
        dest = os.path.join(train_img_folder, _get_image_filename(id_))
        os.link(source, dest)


def get_train_data():
    return _get_data(filename=train_filename)


def get_test_data():
    return _get_data(filename=test_filename)


def _get_data(filename):
    df = pd.read_csv(filename)
    X = ArrayContainer(df['id'].values, attrs=attrs)
    y = df['class'].values
    return X, y


def _download():
    """
    donwload all images and put them in the folder img/.
    It requires the command 'wget' to exist.
    """
    _silent_mkdir(img_folder)
    df = pd.read_csv(full_filename)
    for _, cols in df.iterrows():
        filename = os.path.join(img_folder, _get_image_filename(cols['id']))
        if os.path.exists(filename):
            continue
        url = cols['picture_url']
        cmd = 'wget {} --output-document={}'.format(url, filename)
        subprocess.call(cmd, shell=True)


def _split(test_ratio=0.5, random_state=42):
    """
    Parameters
    ==========

    test_ratio : float between 0 and 1
    random_state : int
        seed used to shuffle the raw data
    """
    assert 0 <= test_ratio <= 1
    df = _load_full_raw(filename=full_filename_raw)
    df = shuffle(df, random_state=random_state)
    df.to_csv(full_filename, index_label='id')
    df = df.drop('our_unique_id', axis=1)
    nb_test = int(len(df) * test_ratio)
    nb_train = len(df) - nb_test
    df.iloc[0:nb_train].to_csv(train_filename, index_label='id')
    df.iloc[nb_train:].to_csv(test_filename, index_label='id')


def _load_full_raw(filename='spipoll.txt'):
    df = pd.read_table(filename)
    #remove duplicates in URL
    df = df.drop_duplicates(subset=['picture_url'], keep='first')
    df['taxa_code'] = df['taxa_code'].apply(_taxa_code_to_int)
    taxa_codes = df['taxa_code'].unique().tolist()
    code_index = {t: i for i, t in enumerate(taxa_codes)}
    df['class'] = df['taxa_code'].apply(lambda code:code_index[code])
    return df


def _get_image_filename(unique_id):
    return 'id_{}.jpg'.format(unique_id)


def _taxa_code_to_int(code):
    try:
        code = int(code)
    except ValueError:
        code = code.replace('{', '').replace('}', '')
        code = int(code)
    return code


def _silent_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
