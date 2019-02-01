###############
RAMP bundle API
###############

This is the full API documentation of the different RAMP packages.

ramp-database
=============

:mod:`rampdb.model`: the database model
---------------------------------------

.. automodule:: rampdb.model
    :no-members:
    :no-inherited-members:

The database schema is summarized in the figure below:

.. image:: _static/img/schema_db.png
   :target: _static/img/schema_db.png

.. currentmodule:: rampdb

General tables
..............

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model.Extension
   model.Keyword
   model.UserInteraction

User-related tables
...................

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model.Team
   model.User

Event-related tables
....................

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model.CVFold
   model.Event
   model.Problem
   model.ScoreType
   model.Workflow

Submission-related tables
.........................

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model.HistoricalContributivity
   model.Submission
   model.SubmissionScore
   model.SubmissionFile
   model.SubmissionFileType
   model.SubmissionOnCVFold
   model.SubmissionSimilarity

Relationship tables
...................

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model.EventTeam
   model.EventAdmin
   model.EventScoreType
   model.ProblemKeyword
   model.SubmissionFileTypeExtension
   model.WorkflowElement
   model.WorkflowElementType

ramp-engine
===========

ramp-frontend
=============

ramp-utils
==========
