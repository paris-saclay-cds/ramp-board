import os
import sys
import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import databoard.multiclass_prediction as prediction
from databoard.config import problems_path

random_state = 42
test_ratio = 0.5
chunk_size = 1024
n_img_load_jobs = 8
problem_name = 'pollenating_insects'  # should be the same as the file name
img_folder = os.path.join(
    problems_path, problem_name, 'data', 'raw', 'imgs'
)
full_filename = os.path.join(
    problems_path, problem_name, 'data', 'raw', 'spipoll.txt'
)
train_filename = os.path.join(
    problems_path, problem_name, 'data', 'train.csv')
test_filename = os.path.join(
    problems_path, problem_name, 'data', 'test.csv')

def prepare_data():
    #_download()
    _split(test_ratio=test_ratio, random_state=random_state)


def get_train_data():
    return _get_data(filename=train_filename)


def get_test_data():
    return _get_data(filename=test_filename)


def _get_data(filename):
    df = pd.read_csv(filename)
    X = df['our_unique_id'].values
    y = df['class'].values
    return X, y


def _download():
    """
    donwload all images and put them in the folder img/.
    It requires the command 'wget' to exist.
    """
    _silent_mkdir(img_folder)
    df = _load_full(full_filename)
    for _, cols in df.iterrows():
        filename = os.path.join(img_folder, _get_image_filename(cols['our_unique_id']))
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
    df = _load_full(filename=full_filename)
    df = shuffle(df, random_state=random_state)
    
    nb_test = int(len(df) * test_ratio)
    nb_train = len(df) - nb_test

    df.iloc[0:nb_train].to_csv(train_filename)
    df.iloc[nb_train:].to_csv(test_filename)


def _load_full(filename='spipoll.txt'):
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
