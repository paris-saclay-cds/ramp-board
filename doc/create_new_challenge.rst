####################
Create New Challenge
####################
RAMP challenges should include two types of datasets: 
*  public dataset which is given to all of the competing users. This dataset
will be split for crossvalidation (train and test) and can be used for 
self-evaluation during the duration of the challenge
*  private dataset which is never given to the competing users and which is used 
to evaluate the score of the users after the challenge deadline has passed.

To write you own challenge for RAMP you need to create directory with the name 
of your challenge in the ramp-kits directory of your RAMP:

        ramp_deployment/ramp_kits/your_challenge

This directory tree should be of the following structure:

.. image:: _static/img/dir_structure.png

Where:

requirements.txt file
----------------------
This file should include all the libraries required to run your challenge
Its content might, for example, look as follows:

    numpy
    scikit-learn
    pandas
    matplotlib
    seaborn


problem.py Python file
----------------------
Here, you will need to define:

*  if the challenge is classification (e.g. 
.. `pollenating insects problem`: https://github.com/ramp-kits/pollenating_insects_3_simplified/blob/master/problem.py)
or regression (e.g.
.. `boston housing problem`: https://github.com/ramp-kits/boston_housing/blob/master/problem.py)
problem;
*  which RAMP workflow you want to use;
*  the score which will be used for evaluation
and include three functions: get_cv(), get_train_data() and get_test_data().

The basic Iris 
.. `problem.py`: https://github.com/ramp-kits/iris/blob/master/problem.py
(which is a classification problem) looks like that:

First you need to import all the required libraries:

    import os
    import pandas as pd
    import rampwf as rw
    from sklearn.model_selection import StratifiedShuffleSplit

Next, define name of the problem:

    problem_title = 'Iris classification'

Define which column in your .csv file is to be classified:
    _target_column_name = 'species'

Specify possible labels:

    _prediction_label_names = ['setosa', 'versicolor', 'virginica']

Choose which prediction class will be used. You can choose from 
.. `existing prediction classes`: https://github.com/paris-saclay-cds/ramp-workflow/tree/master/rampwf/prediction_types
or create your own:
    # A type (class) which will be used to create wrapper objects for y_pred
    Predictions = rw.prediction_types.make_multiclass(
        label_names=_prediction_label_names)

Specify which RAMP workflow will be used. You can use one of the 
.. `existing workflows`: https://github.com/paris-saclay-cds/ramp-workflow/tree/master/rampwf/workflows
or you can create your own:

    # An object implementing the workflow
    workflow = rw.workflows.Classifier()

Then, define which score types will be calculated. Score types defined in 
RAMP can be found 
.. `here`: https://github.com/paris-saclay-cds/ramp-workflow/tree/master/rampwf/score_types
but you can also add new ones:

    score_types = [
        rw.score_types.Accuracy(name='acc'),
        rw.score_types.ClassificationError(name='error'),
        rw.score_types.NegativeLogLikelihood(name='nll'),
        rw.score_types.F1Above(name='f1_70', threshold=0.7),
    ]

Finally you should create few functions which should include:

* get_cv() with input X (the training data) and y (test data)

    def get_cv(X, y):
        cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=57)
        return cv.split(X, y)


    def _read_data(path, f_name):
        data = pd.read_csv(os.path.join(path, 'data', f_name))
        y_array = data[_target_column_name].values
        X_array = data.drop([_target_column_name], axis=1).values
        return X_array, y_array


    def get_train_data(path='.'):
        f_name = 'train.csv'
        return _read_data(path, f_name)


    def get_test_data(path='.'):
        f_name = 'test.csv'
        return _read_data(path, f_name)





Submission directory
--------------------

Starting_kit directory and its content
......................................

Data directory
--------------

your_challenge_starting_kit.ipynb Jupiter notebook file
-------------------------------------------------------


For examples of code of existing challenges feel free to visit RAMP kits github account:

.. _https://github.com/ramp-kits/: https://github.com/ramp-kits/
