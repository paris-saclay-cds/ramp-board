import numpy as np

from tempfile import NamedTemporaryFile
import pandas as pd

import os
import subprocess
from sklearn.base import BaseEstimator

path = os.path.dirname(os.path.realpath(__file__))


class Classifier(BaseEstimator):
    indexes = True
    interface_folder = "/home/mcherti/ramp/interface/rampinterface/torch"

    def __init__(self,
                 data_filename=interface_folder+"/insects.t7",
                 labels_filename=interface_folder+"/labels",
                 nb_examples=25433):
        self.data_filename = data_filename
        self.labels_filename = labels_filename
        self.nb_examples = nb_examples

    def fit(self, X, y=None):
        train = X
        train = pd.DataFrame(train)
        test = pd.DataFrame(np.arange(self.nb_examples))

        temp_dir = '.'
        train_tmp_file = NamedTemporaryFile(delete=False, dir=temp_dir)
        train.to_csv(train_tmp_file, header=None, index=False)
        train_tmp_file.close()

        test_tmp_file = NamedTemporaryFile(delete=False, dir=temp_dir)
        test.to_csv(test_tmp_file, header=None, index=False)
        test_tmp_file.close()

        proba_tmp_file = NamedTemporaryFile(delete=False, dir=temp_dir)
        proba_tmp_file.close()

        targets_tmp_file = NamedTemporaryFile(delete=False, dir=temp_dir)
        targets_tmp_file.close()

        try:
            cmd = "cd {0}; luajit {1}/classifier.lua --X {2} --y {3} --train {4} --test {5} --outputproba {6} --outputtargets {7}"
            args = (path,
                    self.interface_folder,
                    self.data_filename, self.labels_filename,
                    train_tmp_file.name, test_tmp_file.name,
                    proba_tmp_file.name, targets_tmp_file.name)
            subprocess.call(cmd.format(*args), shell=True)
            proba = pd.read_csv(proba_tmp_file.name, header=None).values
            targets = pd.read_csv(targets_tmp_file.name, header=None).values
            if len(targets.shape) == 2:
                targets = targets[:, 0]
            self.targets = targets
            self.proba = proba
            print(self.targets.shape, self.proba.shape)
        except Exception, e:
            print str(e)

        finally:
            os.remove(train_tmp_file.name)
            os.remove(test_tmp_file.name)
            os.remove(proba_tmp_file.name)
            os.remove(targets_tmp_file.name)

    def predict(self, X):
        return self.targets[X]

    def predict_proba(self, X):
        return self.proba[X]


if __name__ == "__main__":
    clf = Classifier()
    clf.fit(np.arange(200))
    print(clf.predict([10, 11, 22]))
