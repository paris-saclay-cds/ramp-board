####################
Create New Challenge
####################

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

The basic Iris problem.py (classification problem) looks like that:








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
