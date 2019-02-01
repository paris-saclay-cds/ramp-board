###############
RAMP bundle API
###############

This is the full API documentation of the different RAMP packages.

RAMP database
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

:mod:`rampdb.tools`: the tools to communicate with the database
---------------------------------------------------------------

.. automodule:: rampdb.tools
    :no-members:
    :no-inherited-members:

.. currentmodule:: rampdb

User-related database tools
...........................

**Functions to act on an entry in the database**

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.team.ask_sign_up_team
   tools.user.approve_user
   tools.team.sign_up_team

**Functions to add new entries in the database**

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.user.add_user
   tools.user.add_user_interaction

**Functions to get entries from the database**

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.team.get_event_team_by_name
   tools.user.get_user_by_name
   tools.user.get_team_by_name
   tools.user.get_user_interactions_by_name

**Functions to set an entry in the database**

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.user.set_user_by_instance

Event-related database tools
............................

**Functions to act on an entry in the database**

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.leaderboard.update_leaderboards
   tools.leaderboard.update_user_leaderboards
   tools.leaderboard.update_all_user_leaderboards

**Functions to add new entries in the database**

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.event.add_event
   tools.event.add_event_admin
   tools.event.add_keyword
   tools.event.add_problem
   tools.event.add_problem_keyword
   tools.event.add_workflow

**Functions to delete entries in the database**

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.event.delete_event
   tools.event.delete_problem
   tools.event.delete_submission_similarity

**Functions to get entries from the database**

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.event.get_event
   tools.event.get_event_admin
   tools.event.get_keyword_by_name
   tools.leaderboard.get_leaderboard
   tools.event.get_problem
   tools.event.get_problem_keyword_by_name
   tools.event.get_workflow

Submission-related database tools
.................................

**Functions to act on an entry in the database**

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.submission.score_submission
   tools.submission.submit_starting_kits

**Functions to add new entries in the database**

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.database.add_extension
   tools.submission.add_submission
   tools.database.add_submission_file_type
   tools.database.add_submission_file_type_extension
   tools.submission.add_submission_similarity

**Functions to get entries from the database**

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.submission.get_event_nb_folds
   tools.database.get_extension
   tools.submission.get_predictions
   tools.submission.get_scores
   tools.submission.get_source_submissions
   tools.submission.get_submissions
   tools.submission.get_submission_by_id
   tools.submission.get_submission_by_name
   tools.submission.get_submission_error_msg
   tools.database.get_submission_file_type
   tools.database.get_submission_file_type_extension
   tools.submission.get_submission_max_ram
   tools.submission.get_submission_state

**Functions to set an entry in the database**

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.submission.set_bagged_scores
   tools.submission.set_predictions
   tools.submission.set_scores
   tools.submission.set_submission_error_msg
   tools.submission.set_submission_max_ram
   tools.submission.set_submission_state
   tools.submission.set_time

Frontend-related database tools
...............................

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.frontend.is_admin
   tools.frontend.is_accessible_event
   tools.frontend.is_accessible_leaderboard
   tools.frontend.is_accessible_code
   tools.frontend.is_user_signed_up

:mod:`rampdb.exceptions`: type of errors raise by the database
--------------------------------------------------------------

.. automodule:: rampdb.exceptions
    :no-members:
    :no-inherited-members:

.. currentmodule:: rampdb

.. autosummary::
   :toctree: generated/
   :template: class.rst

   exceptions.DuplicateSubmissionError
   exceptions.MergeTeamError
   exceptions.MissingSubmissionFileError
   exceptions.MissingExtensionError
   exceptions.NameClashError
   exceptions.TooEarlySubmissionError
   exceptions.UnknownStateError

:mod:`rampdb.testing`: functionalities to test database model and tools
-----------------------------------------------------------------------

.. automodule:: rampdb.testing
    :no-members:
    :no-inherited-members:

.. currentmodule:: rampdb

.. autosummary::
   :toctree: generated/
   :template: function.rst

   testing.add_events
   testing.add_users
   testing.add_problems
   testing.create_test_db
   testing.create_toy_db
   testing.setup_toy_db
   testing.setup_ramp_kits_ramp_data
   testing.setup_files_extension_type
   testing.sign_up_teams_to_events
   testing.submit_all_starting_kits

:mod:`rampdb.utils`: setup and connect RAMP database
----------------------------------------------------

.. automodule:: rampdb.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: rampdb

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.setup_db
   utils.session_scope

RAMP engine
===========

RAMP frontend
=============

RAMP utils
==========
