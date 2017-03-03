import sys
import argparse
import os
from shutil import copyfile
import subprocess

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from skimage.io import imread

def all():
    download()
    split()

def download():
    """
    donwload all images and put them in the folder img/.
    It requires the command 'wget' to exist.
    """
    _silent_mkdir('imgs')
    df = _load_full('spipoll.txt')
    for _, cols in df.iterrows():
        filename = os.path.join('imgs', _get_image_filename(cols['our_unique_id']))
        if os.path.exists(filename):
            continue
        url = cols['picture_url']
        cmd = 'wget {} --output-document={}'.format(url, filename)
        subprocess.call(cmd, shell=True)

def split(train_public=0.25, private_train=0.5, private_test=0.25, random_state=42):
    """
    Split the data (images + their labels) into three parts : 
        public train, private train and private test.
    It puts the images and labels if each in the corresponding folder : 'pub_train/'
    'priv_train' or 'priv_test'. It requires that the images have already been
    downloaded using the function "download()".

    Parameters
    ==========

    train_public : float between 0 and 1
        ratio of public training split
    private_train : float between 0 and 1
        ratio of private training split
    private_test : float between 0 and 1
        ratio of private test split
    random_state : int
        seed used to shuffle the raw data

    The three parameters 'train_public', 'private_train' and 'private_test' 
    have to sum to 1.
    """
    assert _float_equal(train_public + private_train + private_test, 1.0)
    df = _load_full(filename='spipoll.txt')
    df = shuffle(df, random_state=random_state)
    
    priv_train_start = int(len(df) * train_public)
    priv_test_start = priv_train_start + int(len(df) * (private_test))
    
    df.iloc[0:priv_train_start].to_csv('pub_train/data.csv')
    df.iloc[priv_train_start:priv_test_start].to_csv('priv_train/data.csv')
    df.iloc[priv_test_start:].to_csv('priv_test/data.csv')
    
    _silent_mkdir('pub_train')
    _silent_mkdir('priv_train')
    _silent_mkdir('priv_test')
 
    _copy_imgs(data='pub_train/data.csv', source='imgs', dest='pub_train')
    _copy_imgs(data='priv_train/data.csv', source='imgs', dest='priv_train')
    _copy_imgs(data='priv_test/data.csv', source='imgs', dest='priv_test')

def minibatch_img_iterator(df, batch_size=512, include_y=True, folder='imgs', n_jobs=8):
    """
    Generator function that yields minibatches of images, optionally with their labels.

    Parameters
    ==========

    df : pandas DataFrame
        dataframe containing the labels
    batch_size : int
        minibatch size
    include_y : bool
        whether to include labels in the yielded value
    n_jobs : int
        number of parallel jobs for simultanously load images into files

    Yields
    ======

    if include_y is True:
        it yields each time a tuple (X, y) where X is a list
        of numpy arrays of images and y is a list of ints (labels).
        The length of X and y is 'batch_size' at most (it can be smaller).
    if include_y is False: 
        it yields each time X where X is a list of numpy arrays
        of images. The length of X is 'batch_size' at most (it can be smaller).
    The shape of each element of X in both cases
    is (height, width, color), where color=3 and height/width
    vary according to examples (hence the fact that X is a list instead of numpy array).
    """
    df = df.set_index('our_unique_id')
    id_list = df.index.values
    for i in range(0, len(id_list), batch_size):
        id_list_cur = id_list[i:i + batch_size]
        filenames = map(_get_image_filename, id_list_cur)
        filenames = map(lambda filename:os.path.join(folder, filename), filenames)
        X = Parallel(n_jobs=n_jobs)(delayed(imread)(filename) for filename in filenames)
        if include_y:
            y = map(lambda id:df.loc[id]['taxa_code'], id_list)
            yield X, y
        else:
            yield X

def _load_full(filename='spipoll.txt'):
    df = pd.read_table(filename)
    #remove duplicates in URL
    df = df.drop_duplicates(subset=['picture_url'], keep='first')
    df['taxa_code'] = df['taxa_code'].apply(_taxa_code_to_int)
    return df

def _get_image_filename(unique_id):
    return 'id_{}.jpg'.format(unique_id)

def _copy_imgs(data, source, dest):
    df = pd.read_csv(data)
    for unique_id in df['our_unique_id']:
        filename = _get_image_filename(unique_id)
        copyfile(os.path.join(source, filename), os.path.join(dest, filename))

def _taxa_code_to_int(code):
    try:
        code = int(code)
    except ValueError:
        code = code.replace('{', '').replace('}', '')
        code = int(code)
    return code

def _silent_mkdir(path):
    if not os.path.exists(path):
        os.mdkir(path)

def _float_equal(x, y):
    return abs(x - y) <= 1e-10

if __name__ == '__main__':
    desc = """
    A tool to download and split insects data. It requires spipoll.txt to exist. 
    For doing the full setup at once, run : 
    >> python data.py --action all.
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--action', help='Action to perform. Possible actions : "all", "download" and "split"', required=True)
    args = parser.parse_args(sys.argv[1:])
    actions = {'all': all, 'download': download, 'split': split}
    if args.action in actions:
        action = actions[args.action]
        action()
    else:
        print('Unknown action. action should be "all", "download" or "split".')
