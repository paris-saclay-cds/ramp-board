"""
The :mod:`rampdb.tools` provides some routine to get and set information from
and in the database.
"""

from .api import get_event_nb_folds
from .api import get_predictions
from .api import get_scores
from .api import get_submission_by_id
from .api import get_submission_by_name
from .api import get_submission_state
from .api import get_submissions
from .api import get_time

from .api import set_predictions
from .api import set_scores
from .api import set_submission_error_msg
from .api import set_submission_max_ram
from .api import set_submission_state
from .api import set_time

from .api import score_submission

from .tools import combine_predictions_list
from .tools import get_active_user_event_team
from .tools import get_n_team_members
from .tools import get_n_user_teams
from .tools import get_next_best_single_fold
from .tools import get_team_members
from .tools import get_user_event_teams
from .tools import get_user_teams

__all__ = [
    'combine_predictions_list',
    'get_active_user_event_team',
    'get_event_nb_folds',
    'get_predictions',
    'get_n_team_members',
    'get_n_user_teams',
    'get_next_best_single_fold',
    'get_scores',
    'get_submission_by_id',
    'get_submission_by_name',
    'get_submission_state',
    'get_submissions',
    'get_team_members',
    'get_time',
    'get_user_event_teams',
    'get_user_teams',
    'set_predictions',
    'set_scores',
    'set_submission_error_msg',
    'set_submission_max_ram',
    'set_submission_state',
    'set_time',
    'score_submission'
]
