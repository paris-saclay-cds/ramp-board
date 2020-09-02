########
RAMP API
########

This is the full API documentation of the different RAMP packages.

RAMP database
=============

:mod:`ramp_database.model`: the database model
----------------------------------------------

.. automodule:: ramp_database.model
    :no-members:
    :no-inherited-members:

The database schema is summarized in the figure below:

.. image:: _static/img/schema_db.png
   :target: _static/img/schema_db.png

.. currentmodule:: ramp_database

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
   model.SubmissionScoreOnCVFold
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

:mod:`ramp_database.tools`: the tools to communicate with the database
----------------------------------------------------------------------

.. automodule:: ramp_database.tools
    :no-members:
    :no-inherited-members:

.. currentmodule:: ramp_database

User-related database tools
...........................

**Functions to act on an entry in the database**

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.team.ask_sign_up_team
   tools.user.approve_user
   tools.user.make_user_admin
   tools.user.set_user_access_level
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

**Functions to delete an entry in the database**

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.user.delete_user
   tools.team.delete_event_team

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
   tools.frontend.is_user_sign_up_requested

:mod:`ramp_database.exceptions`: type of errors raise by the database
---------------------------------------------------------------------

.. automodule:: ramp_database.exceptions
    :no-members:
    :no-inherited-members:

.. currentmodule:: ramp_database

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

:mod:`ramp_database.testing`: functionalities to test database model and tools
------------------------------------------------------------------------------

.. automodule:: ramp_database.testing
    :no-members:
    :no-inherited-members:

.. currentmodule:: ramp_database

.. autosummary::
   :toctree: generated/
   :template: function.rst

   testing.add_events
   testing.add_users
   testing.add_problems
   testing.create_test_db
   testing.create_toy_db
   testing.setup_toy_db
   testing.setup_ramp_kit_ramp_data
   testing.setup_files_extension_type
   testing.sign_up_teams_to_events
   testing.submit_all_starting_kits

:mod:`ramp_database.utils`: setup and connect RAMP database
-----------------------------------------------------------

.. automodule:: ramp_database.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: ramp_database

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.check_password
   utils.hash_password
   utils.setup_db
   utils.session_scope

RAMP engine
===========

The RAMP engine is made of a dispatcher to orchestrate the training and
evaluation of submissions which are processed using workers. Since there is
a dispatcher for each RAMP event, we provide a daemon which will start
dispatcher for each open event.

RAMP Dispatcher
---------------

.. currentmodule:: ramp_engine

.. autosummary::
   :toctree: generated/
   :template: class.rst

   daemon.Daemon
   dispatcher.Dispatcher

RAMP Workers
------------

.. currentmodule:: ramp_engine

.. autosummary::
   :toctree: generated/
   :template: class.rst

   base.BaseWorker
   local.CondaEnvWorker
   aws.AWSWorker

RAMP frontend
=============

The RAMP frontend is the development of the website. It uses extensively Flask.

.. currentmodule:: ramp_frontend

.. autosummary::
   :toctree: generated/
   :template: function.rst

   create_app

:mod:`ramp_frontend.views`: views used on the frontend to interact with the user
--------------------------------------------------------------------------------

.. automodule:: ramp_frontend.views
    :no-members:
    :no-inherited-members:

General views
.............

.. currentmodule:: ramp_frontend.views

.. autosummary::
   :toctree: generated/
   :template: function.rst

   general.index
   general.ramp
   general.teaching
   general.data_science_themes
   general.keywords

Authentication views
....................

.. currentmodule:: ramp_frontend.views

.. autosummary::
   :toctree: generated/
   :template: function.rst

   auth.load_user
   auth.login
   auth.logout
   auth.reset_password
   auth.reset_with_token
   auth.sign_up
   auth.update_profile
   auth.user_confirm_email

Admin views
...........

.. currentmodule:: ramp_frontend.views

.. autosummary::
   :toctree: generated/
   :template: function.rst

   admin.approve_users
   admin.approve_single_user
   admin.approve_sign_up_for_event
   admin.update_event
   admin.user_interactions
   admin.dashboard_submissions

Leaderboard views
.................

.. currentmodule:: ramp_frontend.views

.. autosummary::
   :toctree: generated/
   :template: function.rst

   leaderboard.my_submissions
   leaderboard.leaderboard
   leaderboard.competition_leaderboard
   leaderboard.private_leaderboard
   leaderboard.private_competition_leaderboard

Submission views
................

.. currentmodule:: ramp_frontend.views

.. autosummary::
   :toctree: generated/
   :template: function.rst

   ramp.problems
   ramp.problem
   ramp.user_event
   ramp.sign_up_for_event
   ramp.sandbox
   ramp.ask_for_event
   ramp.credit
   ramp.event_plots
   ramp.view_model
   ramp.view_submission_error

Utilities
.........

.. currentmodule:: ramp_frontend.views

.. autosummary::
   :toctree: generated/
   :template: function.rst

   redirect.redirect_to_credit
   redirect.redirect_to_sandbox
   redirect.redirect_to_user
   visualization.score_plot

:mod:`ramp_frontend.forms`: forms used in the website
-----------------------------------------------------

.. automodule:: ramp_frontend.forms
    :no-members:
    :no-inherited-members:

.. currentmodule:: ramp_frontend

.. autosummary::
   :toctree: generated/
   :template: class.rst

   forms.LoginForm
   forms.UserUpdateProfileForm
   forms.UserCreateProfileForm
   forms.CodeForm
   forms.SubmitForm
   forms.UploadForm
   forms.EventUpdateProfileForm
   forms.MultiCheckboxField
   forms.ImportForm
   forms.CreditForm
   forms.AskForEventForm

:mod:`ramp_frontend.testing`: functionalities to test the frontend
------------------------------------------------------------------

.. automodule:: ramp_frontend.testing
    :no-members:
    :no-inherited-members:

.. currentmodule:: ramp_frontend

.. autosummary::
   :toctree: generated/
   :template: function.rst

   testing.login
   testing.logout
   testing.login_scope

:mod:`ramp_frontend.utils`: Utilities to ease sending email
-----------------------------------------------------------

.. automodule:: ramp_frontend.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: ramp_frontend

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.body_formatter_user
   utils.send_mail

RAMP utils
==========

The RAMP utils package is made of utilities used for managing configuration,
deploying RAMP events, and other utilities shared between the different RAMP
packages.

Configuration utilities
-----------------------

.. currentmodule:: ramp_utils

.. autosummary::
   :toctree: generated/
   :template: function.rst

   generate_flask_config
   generate_ramp_config
   generate_worker_config
   read_config
   testing.database_config_template
   testing.ramp_config_template

Datasets utilities
------------------

.. currentmodule:: ramp_utils

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.fetch_from_osf

.. autosummary::
   :toctree: generated/
   :template: class.rst

   datasets.OSFRemoteMetaData

Deployment utilities
--------------------

.. currentmodule:: ramp_utils

.. autosummary::
   :toctree: generated/
   :template: function.rst

   deploy.deploy_ramp_event

RAMP shared utilities
---------------------

.. currentmodule:: ramp_utils

.. autosummary::
   :toctree: generated/
   :template: function.rst

   import_module_from_source

